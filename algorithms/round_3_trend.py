"""Round 3 MR + trend-bias variant on HYDROGEL.

Adds a trend nudge to the MR target so HYDROGEL can go short when
price drifts persistently down. Keeps VELVET MR unchanged.

Mechanism: anchor = slow EMA (span 500), fast = shorter EMA (span
200), div = fast - anchor.
  |div| <= DEADBAND        -> trend off, pure MR (fade the dip)
  DEADBAND < |div| < SAT   -> trend contributes; MR linearly attenuated
  |div| >= SAT             -> pure trend (MR off)

Status: on 4-day backtest this currently yields ~+13K vs pure MR's
+132K. The MR-dip-fade and short-the-drop signals are not cleanly
separable with a simple EMA-crossover detector: historical dips are
visible to this detector and triggering it costs the MR profit on
d0/d1/d2. Keeping this file as reference / for further tuning.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15           # wide spread (~15) needs matching skew to cross
    ANCHOR_SPAN = 500       # short span: anchor tracks drift quickly so a
                            # persistent trend doesnt pile one-sided longs
    MR_STRENGTH = 5         # target_pos per tick of (fair - anchor)
    FAST_SPAN = 200         # mid-length EMA; short enough to register a
                            # steady drift but long enough not to respond
                            # to oscillations around the anchor
    TREND_BIAS = 25         # target shift per unit of (fast - anchor)
                            # above deadband - overrides MR on drifts
    TREND_DEADBAND = 5      # |fast - anchor| must exceed this before the
                            # trend term contributes
    TREND_SATURATION = 12   # when |fast - anchor| reaches this, MR is
                            # fully disabled (linear attenuation between)


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3            # tight spread (~5) - gentle posting
    ANCHOR_SPAN = 5000      # VELVET mean-reverts cleanly on long horizon
    MR_STRENGTH = 10        # strong MR signal, let skew execute patiently
    FAST_SPAN = 0           # VELVET's long anchor already protects against
    TREND_BIAS = 0          # trend pile-up; no trend nudge needed
    TREND_DEADBAND = 0
    TREND_SATURATION = 0


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 0
    ANCHOR_SPAN = 0
    MR_STRENGTH = 0
    FAST_SPAN = 0
    TREND_BIAS = 0
    TREND_DEADBAND = 0
    TREND_SATURATION = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 0
    ANCHOR_SPAN = 0
    MR_STRENGTH = 0
    FAST_SPAN = 0
    TREND_BIAS = 0
    TREND_DEADBAND = 0
    TREND_SATURATION = 0


def _micro_mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    bv = od.buy_orders[bb]
    av = -od.sell_orders[ba]
    if bv + av <= 0:
        return (bb + ba) / 2.0
    return (bb * av + ba * bv) / (bv + av)


class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # EMA state: {symbol: current_ema_value}
        ema: Dict[str, float] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                ema = td.get("ema", {})
            except Exception:
                pass

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                result[HydrogelParams.SYMBOL] = self._mm_with_fair(
                    state, HydrogelParams, fv, ema)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                result[VelvetParams.SYMBOL] = self._mm_with_fair(
                    state, VelvetParams, velvet_mid, ema)

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    result[P.SYMBOL] = self._mm_with_fair(
                        state, P, fv, ema)

        return result, 0, json.dumps({"ema": ema})

    def _mm_with_fair(self, state: TradingState, P, fv: float,
                      ema: Dict[str, float]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders

        # EMA anchor update + MR target + trend nudge
        target_pos = 0
        if P.ANCHOR_SPAN > 0:
            prev = ema.get(P.SYMBOL)
            if prev is None:
                anchor = fv
            else:
                alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
                anchor = alpha * fv + (1 - alpha) * prev
            ema[P.SYMBOL] = anchor

            trend_signal = 0.0
            mr_weight = 1.0
            if P.FAST_SPAN > 0 and P.TREND_BIAS > 0:
                fast_key = P.SYMBOL + "_f"
                fprev = ema.get(fast_key)
                if fprev is None:
                    fast = fv
                else:
                    fa = 2.0 / (P.FAST_SPAN + 1.0)
                    fast = fa * fv + (1 - fa) * fprev
                ema[fast_key] = fast
                divergence = fast - anchor
                absdiv = abs(divergence)
                if absdiv > P.TREND_DEADBAND:
                    sign = 1.0 if divergence > 0 else -1.0
                    trend_signal = P.TREND_BIAS * sign * (absdiv - P.TREND_DEADBAND)
                    # Linearly attenuate MR between deadband and saturation.
                    # Past saturation MR is off entirely and only trend acts.
                    span = max(1.0, P.TREND_SATURATION - P.TREND_DEADBAND)
                    mr_weight = max(0.0, 1.0 - (absdiv - P.TREND_DEADBAND) / span)

            raw = mr_weight * (-P.MR_STRENGTH * (fv - anchor)) + trend_signal
            target_pos = max(-P.POS_LIMIT,
                             min(P.POS_LIMIT, int(round(raw))))

        # Skew relative to target. Long-vs-target lowers fv_eff (sell
        # pressure), short-vs-target raises it (buy pressure). When
        # flat and target is at limit, the shift crosses the book.
        skew = P.INV_SKEW * (pos - target_pos) / P.POS_LIMIT
        fv_eff = fv - skew

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        # Aggressive takes past fv_eff; directly buys / sells from
        # market quotes up to POS_LIMIT when MR signal is strong.
        for price in sorted(od.sell_orders.keys()):
            if price < fv_eff and buy_cap > 0:
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                buy_cap -= qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > fv_eff and sell_cap > 0:
                qty = min(od.buy_orders[price], sell_cap)
                orders.append(Order(P.SYMBOL, price, -qty))
                sell_cap -= qty

        # Posted quotes: penny-best shifted by skew.
        base_bid = best_bid + 1
        base_ask = best_ask - 1
        if base_bid >= base_ask:
            base_bid = best_bid
            base_ask = best_ask

        shift = int(round(-skew))
        our_bid = base_bid + shift
        our_ask = base_ask + shift

        our_bid = min(our_bid, int(fv_eff))
        our_ask = max(our_ask, int(fv_eff) + 1)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))

        return orders
