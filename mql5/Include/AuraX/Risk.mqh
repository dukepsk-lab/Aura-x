//+------------------------------------------------------------------+
//|  AuraX/Risk.mqh — ATR position sizing + barriers (Layer 6/7)     |
//|  Mirrors aurax.l6_risk.sizing: constant-dollar-risk ATR sizing.  |
//+------------------------------------------------------------------+
#ifndef AURAX_RISK_MQH
#define AURAX_RISK_MQH

//--- Wilder ATR via the indicator handle (caller owns the handle).
double AtrValue(const int atr_handle, const int shift = 0)
{
   double buf[];
   if(CopyBuffer(atr_handle, 0, shift, 1, buf) <= 0)
      return 0.0;
   return buf[0];
}

//--- Constant-dollar-risk lot size:  lots = risk_cash / (stop_dist * tick_val/tick_size)
//    Auto-shrinks as ATR (hence stop distance) rises.
double PositionSizeATR(const string symbol, const double risk_cash,
                       const double atr, const double stop_multiplier)
{
   if(atr <= 0.0 || stop_multiplier <= 0.0) return 0.0;

   const double stop_distance = stop_multiplier * atr;
   const double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   const double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_size <= 0.0) return 0.0;

   const double money_per_lot = (stop_distance / tick_size) * tick_value;
   if(money_per_lot <= 0.0) return 0.0;

   double lots = risk_cash / money_per_lot;

   //--- normalise to broker volume constraints
   const double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   const double vmax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   const double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(vstep > 0.0) lots = MathFloor(lots / vstep) * vstep;
   lots = MathMax(vmin, MathMin(vmax, lots));
   return lots;
}

//--- Capped fractional-Kelly multiplier on the calibrated meta-probability.
double ConfidenceMultiplier(const double p, const double tau,
                            const double kelly_fraction, const double cap)
{
   if(p < tau) return 0.0;
   const double edge = 2.0 * p - 1.0;
   const double ref  = 2.0 * tau - 1.0;
   double scaled = (ref > 0.0) ? kelly_fraction * edge / ref : kelly_fraction;
   return MathMin(cap, MathMax(0.0, scaled));
}

#endif // AURAX_RISK_MQH
