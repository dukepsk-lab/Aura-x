//+------------------------------------------------------------------+
//|                                                      AuraX_EA.mq5 |
//|   Aura-X execution agent (Layer 7).                              |
//|                                                                  |
//|   The Python stack (L0-L6) makes the *decision*; this EA is the  |
//|   guarded hand at the point of contact. On each new H4 bar it    |
//|   pulls the latest approved signal from the decision service     |
//|   (FastAPI / file bridge), then sizes and places the order with  |
//|   local spread/news/slippage guards and ATR-scaled stops.        |
//|                                                                  |
//|   SKELETON: the transport (WebRequest/file read) is marked TODO; |
//|   everything else compiles as a working guarded order path.      |
//+------------------------------------------------------------------+
#property copyright "Aura-X"
#property version   "0.10"
#property strict

#include <Trade/Trade.mqh>
#include <AuraX/Risk.mqh>
#include <AuraX/Guards.mqh>

//--- inputs (defaults mirror config/default.yaml) -------------------
input double InpRiskPerTrade      = 0.005;   // fraction of equity risked per trade
input double InpAtrStopMult       = 2.5;     // H4 swing stop in [2.0, 3.0] x ATR
input double InpTpAtrMult         = 2.0;     // take-profit in ATR multiples
input int    InpAtrPeriod         = 14;
input double InpMetaTau           = 0.55;    // trade iff P(correct) >= tau
input double InpKellyFraction     = 0.5;
input double InpMaxConfMult       = 2.0;
input double InpMaxSpreadPoints   = 20.0;    // ~2.0 pips on a 5-digit symbol
input double InpMaxSlippagePoints = 15.0;
input int    InpNewsBlackoutMin   = 30;
input ulong  InpMagic             = 990011;  // idempotent strategy id
input string InpDecisionUrl       = "http://127.0.0.1:8000";  // FastAPI base

//--- globals --------------------------------------------------------
CTrade   trade;
int      atr_handle = INVALID_HANDLE;
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetTypeFillingBySymbol(_Symbol);

   atr_handle = iATR(_Symbol, PERIOD_H4, InpAtrPeriod);
   if(atr_handle == INVALID_HANDLE)
   {
      Print("AuraX: failed to create ATR handle");
      return INIT_FAILED;
   }
   PrintFormat("AuraX EA initialised on %s H4 (magic=%I64u)", _Symbol, InpMagic);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(atr_handle != INVALID_HANDLE)
      IndicatorRelease(atr_handle);
}

//+------------------------------------------------------------------+
//| Decision record returned by the Python service.                  |
//+------------------------------------------------------------------+
struct AuraSignal
{
   int    side;       // -1 short | 0 flat | +1 long
   double meta_prob;  // calibrated P(signal correct)
   bool   approved;   // passed meta gate + risk upstream
};

//--- TODO(v1): fetch the latest approved signal for `symbol`.
//    Wire to FastAPI via WebRequest(InpDecisionUrl + "/signal/<symbol>")
//    or a JSON file written to MQL5/Files by the bridge. Returns flat for now.
bool FetchSignal(const string symbol, AuraSignal &out)
{
   out.side = 0;
   out.meta_prob = 0.0;
   out.approved = false;
   return false; // no signal until transport is wired
}

//+------------------------------------------------------------------+
//| Act once per completed H4 bar.                                   |
//+------------------------------------------------------------------+
void OnTick()
{
   const datetime t = iTime(_Symbol, PERIOD_H4, 0);
   if(t == last_bar_time)
      return;            // only act on a fresh bar close
   last_bar_time = t;

   //--- pre-trade spread guard
   if(!SpreadOK(_Symbol, InpMaxSpreadPoints))
      return;

   AuraSignal sig;
   if(!FetchSignal(_Symbol, sig) || !sig.approved || sig.side == 0)
      return;

   //--- confidence-scaled, ATR-sized position
   const double atr = AtrValue(atr_handle, 0);
   if(atr <= 0.0)
      return;

   const double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   const double risk_cash   = equity * InpRiskPerTrade;
   const double conf        = ConfidenceMultiplier(sig.meta_prob, InpMetaTau,
                                                   InpKellyFraction, InpMaxConfMult);
   if(conf <= 0.0)
      return;

   double lots = PositionSizeATR(_Symbol, risk_cash, atr, InpAtrStopMult) * conf;
   if(lots <= 0.0)
      return;

   //--- ATR-scaled barriers
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   const double price = (sig.side > 0)
                        ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                        : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   const double sl_dist = InpAtrStopMult * atr;
   const double tp_dist = InpTpAtrMult  * atr;

   bool ok = false;
   if(sig.side > 0)
      ok = trade.Buy(lots, _Symbol, price, price - sl_dist, price + tp_dist, "AuraX");
   else
      ok = trade.Sell(lots, _Symbol, price, price + sl_dist, price - tp_dist, "AuraX");

   if(!ok)
      PrintFormat("AuraX: order failed retcode=%u (%s)", trade.ResultRetcode(),
                  trade.ResultRetcodeDescription());
}
//+------------------------------------------------------------------+
