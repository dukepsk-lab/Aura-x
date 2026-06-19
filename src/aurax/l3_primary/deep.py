"""Deep time-series ensemble members (Roadmap v2 — "architecture matching").

The forecasting research's lesson is to match the model to the signal's
character, so these members slot into the regime-conditional ensemble:

* :class:`CNNSide`     — 1D-conv member (the v1 deep backbone);
* :class:`PatchTSTSide`— patch-based transformer, for **cyclical** regimes;
* :class:`SSMSide`     — diagonal state-space model, for **chaotic** regimes
  (a compact SSM in the Mamba family; a full selective-SSM can drop in here).

All implement :class:`~aurax.l3_primary.base.SideModel` and are weighted per
regime via ``primary.regime_overrides`` (Layer 2). Torch is a lazy, optional
dependency (the ``deep`` extra): the GBM/logistic members remain the fallback,
and a :class:`SequenceSideModel` raises a clear error if torch is absent.

A :class:`SequenceSideModel` base does standardisation, windowing, the weighted
training loop (Adam + early stopping) and 3-class probability alignment; each
architecture only supplies ``_build_net``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PROBA_COLUMNS, SideModel, align_proba, encode_labels
from .windows import make_sequences


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "deep members require PyTorch. Install with:  pip install -e '.[deep]'"
        ) from exc
    return torch, nn


class SequenceSideModel(SideModel):
    """Base for windowed sequence members (handles everything but the network)."""

    name = "sequence"

    def __init__(
        self,
        lookback: int = 24,
        hidden: int = 32,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 128,
        patience: int = 6,
        val_frac: float = 0.15,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.weight_decay = weight_decay
        self.seed = seed

    # subclasses implement this
    def _build_net(self, torch, nn, n_features: int, n_classes: int):  # noqa: ANN001
        raise NotImplementedError

    # --- helpers -------------------------------------------------------------
    def _standardize_windows(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
        z = (np.nan_to_num(X.to_numpy(dtype=float)) - self.mean_) / self.std_
        return make_sequences(pd.DataFrame(z, index=X.index), self.lookback)

    def fit(self, X, y, sample_weight=None) -> SequenceSideModel:
        torch, nn = _import_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        xv = np.nan_to_num(X.to_numpy(dtype=float))
        self.mean_, self.std_ = xv.mean(axis=0), xv.std(axis=0) + 1e-9
        seqs, idx = self._standardize_windows(X)
        if len(seqs) < 10:
            raise ValueError("not enough rows to build sequence windows")

        labels = encode_labels(y.reindex(idx))
        self.classes_ = sorted(set(labels.tolist()))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([cls_idx[v] for v in labels], dtype=np.int64)
        w = (
            np.ones(len(idx), dtype=float)
            if sample_weight is None
            else np.clip(sample_weight.reindex(idx).to_numpy(dtype=float), 0, None)
        )

        # time-ordered train/val split for early stopping
        n_val = max(1, int(len(seqs) * self.val_frac))
        cut = len(seqs) - n_val
        device = "cpu"
        to_t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        x_tr, x_val = to_t(seqs[:cut]), to_t(seqs[cut:])
        y_tr, y_val = to_t(y_enc[:cut], torch.long), to_t(y_enc[cut:], torch.long)
        w_tr, w_val = to_t(w[:cut]), to_t(w[cut:])

        net = self._build_net(torch, nn, seqs.shape[2], len(self.classes_)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss(reduction="none")

        best_val, best_state, bad = float("inf"), None, 0
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            net.train()
            order = rng.permutation(cut)
            for s in range(0, cut, self.batch_size):
                b = order[s : s + self.batch_size]
                opt.zero_grad()
                loss = (ce(net(x_tr[b]), y_tr[b]) * w_tr[b]).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = float((ce(net(x_val), y_val) * w_val).mean())
            if val_loss < best_val - 1e-5:
                best_val, best_state, bad = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self._net = net
        self._torch = torch
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        torch = self._torch
        seqs, idx = self._standardize_windows(X)
        out = pd.DataFrame(np.nan, index=X.index, columns=list(PROBA_COLUMNS))
        if len(seqs):
            with torch.no_grad():
                logits = self._net(torch.as_tensor(seqs, dtype=torch.float32))
                proba = torch.softmax(logits, dim=1).cpu().numpy()
            out.loc[idx, :] = align_proba(proba, self.classes_, idx).to_numpy()
        # warm-up rows (no full window) → neutral, so they don't bias the blend
        return out.fillna(1.0 / 3.0)


class CNNSide(SequenceSideModel):
    """1D-convolutional sequence member."""

    name = "cnn"

    def _build_net(self, torch, nn, n_features, n_classes):  # noqa: ANN001
        hidden = self.hidden

        class _CNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv1d(n_features, hidden, 3, padding=1), nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                )
                self.head = nn.Linear(hidden, n_classes)

            def forward(self, x):  # x: (B, L, F)
                return self.head(self.body(x.transpose(1, 2)))

        return _CNN()


class PatchTSTSide(SequenceSideModel):
    """Patch-based transformer member (cyclical regimes)."""

    name = "patchtst"

    def __init__(self, patch_len: int = 6, stride: int = 3, nhead: int = 4, n_layers: int = 2, **kw) -> None:
        super().__init__(**kw)
        self.patch_len = patch_len
        self.stride = stride
        self.nhead = nhead
        self.n_layers = n_layers

    def _build_net(self, torch, nn, n_features, n_classes):  # noqa: ANN001
        d_model, lookback = self.hidden, self.lookback
        patch_len = min(self.patch_len, lookback)
        stride = max(1, min(self.stride, patch_len))
        n_patches = (lookback - patch_len) // stride + 1
        nhead = self.nhead if d_model % self.nhead == 0 else 1  # nhead must divide d_model
        n_layers = self.n_layers

        class _PatchTST(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(patch_len * n_features, d_model)
                self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1
                )
                self.enc = nn.TransformerEncoder(layer, n_layers)
                self.head = nn.Linear(d_model, n_classes)

            def forward(self, x):  # x: (B, L, F)
                b = x.shape[0]
                p = x.unfold(1, patch_len, stride)              # (B, n_patches, F, patch_len)
                p = p.permute(0, 1, 3, 2).reshape(b, n_patches, patch_len * n_features)
                z = self.enc(self.embed(p) + self.pos)
                return self.head(z.mean(dim=1))

        return _PatchTST()


class SSMSide(SequenceSideModel):
    """Diagonal state-space (SSM) member (chaotic regimes; Mamba family)."""

    name = "ssm"

    def _build_net(self, torch, nn, n_features, n_classes):  # noqa: ANN001
        d_model = self.hidden

        class _DiagSSM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inp = nn.Linear(n_features, d_model)
                self.log_decay = nn.Parameter(torch.randn(d_model) * 0.1)
                self.gain = nn.Parameter(torch.randn(d_model) * 0.1)
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, n_classes)

            def forward(self, x):  # x: (B, L, F)
                u = self.inp(x)                       # (B, L, d)
                a = torch.sigmoid(self.log_decay)     # diagonal decay in (0, 1)
                h = torch.zeros(x.shape[0], u.shape[2], device=x.device)
                for t in range(u.shape[1]):           # short lookback → cheap scan
                    h = a * h + self.gain * u[:, t, :]
                return self.head(self.norm(h))

        return _DiagSSM()
