//+------------------------------------------------------------------+
//|  AuraX/Guards.mqh — pre-trade execution guards (Layer 7)          |
//|  Mirrors aurax.l7_execution.guards: spread / news / slippage.    |
//+------------------------------------------------------------------+
#ifndef AURAX_GUARDS_MQH
#define AURAX_GUARDS_MQH

//--- Skip if the current spread exceeds budget (in points).
bool SpreadOK(const string symbol, const double max_spread_points)
{
   const double spread = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   return spread <= max_spread_points;
}

//--- Enforce max slippage between intended and fill price (in points).
bool SlippageOK(const double intended, const double fill,
                const double point, const double max_slippage_points)
{
   if(point <= 0.0) return false;
   return MathAbs(fill - intended) / point <= max_slippage_points;
}

//--- News blackout: true if now is within +/- minutes of any event time.
//    Event times are provided by the decision service (calendar feed).
bool InNewsWindow(const datetime now, const datetime &events[],
                  const int blackout_minutes)
{
   const int window = blackout_minutes * 60;
   for(int i = 0; i < ArraySize(events); i++)
      if(MathAbs((long)(now - events[i])) <= window)
         return true;
   return false;
}

#endif // AURAX_GUARDS_MQH
