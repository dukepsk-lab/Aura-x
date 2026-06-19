# Aura-X MQL5 — Execution Agent (Layer 7)

The Python stack (L0–L6) makes the **decision**; this MQL5 Expert Advisor is the
**guarded hand at the point of contact**. The split keeps the heavy ML/validation
in Python and only the latency-sensitive, broker-facing execution in MT5.

## Layout

```
mql5/
├── Experts/AuraX_EA.mq5      # the EA: per-H4-bar guarded order placement
└── Include/AuraX/
    ├── Risk.mqh              # ATR sizing + confidence multiplier (mirrors L6)
    └── Guards.mqh            # spread / news / slippage guards (mirrors L7)
```

## Decision transport (the one TODO)

`FetchSignal()` in the EA is the seam between Python and MT5. Wire it to either:

1. **HTTP** — `WebRequest()` against the FastAPI service
   (`GET {InpDecisionUrl}/signal/<symbol>`). Add the URL to
   *Tools → Options → Expert Advisors → Allow WebRequest for listed URL*.
2. **File bridge** — the Python side writes an approved-signal JSON to
   `MQL5/Files/aurax_<symbol>.json`; the EA reads it on each new bar.

The signal the EA expects: `side ∈ {-1,0,+1}`, `meta_prob` (calibrated
`P(correct)`), and an `approved` flag (meta-gate + risk already cleared upstream).

## Behaviour (already implemented)

- Acts once per **completed H4 bar**.
- **Spread guard** before anything else; **news blackout** + **slippage** helpers ready.
- **ATR-scaled** stop/target (`InpAtrStopMult` ≈ 2.0–3.0×) and
  **constant-dollar-risk** sizing scaled by the **confidence multiplier**.
- Idempotent **magic number** so restarts don't double-fire.

## Deploy

1. Copy `Experts/AuraX_EA.mq5` → `<terminal>/MQL5/Experts/` and
   `Include/AuraX/` → `<terminal>/MQL5/Include/AuraX/`.
2. Compile in MetaEditor (F7).
3. Attach to an **EURUSD H4** (and **GBPUSD H4**) chart. **Demo first** — nothing
   goes live until the validation gate (§5) is cleared.
