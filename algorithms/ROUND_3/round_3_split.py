"""Round 3 SPLIT - per-product MR vs breakout.

Evidence from 4-day backtests:
  HYDROGEL on day 3 (platform test day): MR=+780, breakout W=600=+5,895.
  VELVET   on day 3:                      MR=+2,678, breakout=-4,174.
  VELVET mean-reverts on all 4 days; HYDROGEL drifts persistently on
  day 3 and makes sense to trend on.

So:
  HYDROGEL -> breakout (Donchian W=600)
  VELVET   -> MR (EMA anchor + MR target + inventory skew)
  VEV_4000/4500 -> pure MM around S_mid - K

On historicals (d0/d1/d2), HYDROGEL breakout loses ~-80K/day. The bet
is that the live/test sample behaves like day 3 (drifty, non-MR) and
not like the historicals. If the live market turns out to mean-revert,
this underperforms round_3.py materially.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    MODE = "breakout"
    WINDOW = 600
    BREAKOUT_TOL = 0.0


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    MODE = "mr"
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 0
    MODE = "off"


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 0
    MODE = "off"


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

        ema: Dict[str, float] = {}
        hist: Dict[str, List[float]] = {}
        tgt: Dict[str, int] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                ema = td.get("ema", {})
                hist = td.get("hist", {})
                tgt = td.get("tgt", {})
            except Exception:
                pass

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                result[HydrogelParams.SYMBOL] = self._trade(
                    state, HydrogelParams, fv, ema, hist, tgt)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                result[VelvetParams.SYMBOL] = self._trade(
                    state, VelvetParams, velvet_mid, ema, hist, tgt)

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    result[P.SYMBOL] = self._trade(
                        state, P, fv, ema, hist, tgt)

        return result, 0, json.dumps({"ema": ema, "hist": hist, "tgt": tgt})

    def _trade(self, state: TradingState, P, fv: float,
               ema: Dict[str, float],
               hist: Dict[str, List[float]],
               tgt: Dict[str, int]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders

        target_pos = 0

        if P.MODE == "mr":
            prev = ema.get(P.SYMBOL)
            if prev is None:
                anchor = fv
            else:
                alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
                anchor = alpha * fv + (1 - alpha) * prev
            ema[P.SYMBOL] = anchor
            raw = -P.MR_STRENGTH * (fv - anchor)
            target_pos = max(-P.POS_LIMIT,
                             min(P.POS_LIMIT, int(round(raw))))

        elif P.MODE == "breakout":
            target_pos = tgt.get(P.SYMBOL, 0)
            h = hist.get(P.SYMBOL, [])
            if len(h) >= P.WINDOW // 4:
                hi = max(h)
                lo = min(h)
                if fv >= hi + P.BREAKOUT_TOL:
                    target_pos = P.POS_LIMIT
                elif fv <= lo - P.BREAKOUT_TOL:
                    target_pos = -P.POS_LIMIT
            h.append(fv)
            if len(h) > P.WINDOW:
                h = h[-P.WINDOW:]
            hist[P.SYMBOL] = h
            tgt[P.SYMBOL] = target_pos

        skew = P.INV_SKEW * (pos - target_pos) / P.POS_LIMIT
        fv_eff = fv - skew

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

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
