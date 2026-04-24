"""Round 3 DOUBLE-MA crossover.

No assumption that price has a fixed mean. Track rolling fast-SMA
and slow-SMA of the live micro-mid per symbol.

  fast > slow   -> uptrend   -> target = +POS_LIMIT   (fully long)
  fast < slow   -> downtrend -> target = -POS_LIMIT   (fully short)
  equal         -> target = 0

Execution reuses the INV_SKEW mechanic from round_3.py: fv_eff is
shifted by INV_SKEW * (pos - target)/POS_LIMIT so when pos is far
from target, fv_eff crosses the book and drives position toward
target via aggressive takes.

Before slow-MA has enough samples (|history| < SLOW_MA), target = 0
so we don't trade on unseeded averages.

VEV_4000 / VEV_4500 stay pure MM around fair = S_mid - K, no MA
overlay (they already inherit the underlying's direction per-tick).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    FAST_MA = 100
    SLOW_MA = 500


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    FAST_MA = 100
    SLOW_MA = 500


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 0
    FAST_MA = 0
    SLOW_MA = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 0
    FAST_MA = 0
    SLOW_MA = 0


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

        hist: Dict[str, List[float]] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                hist = td.get("hist", {})
            except Exception:
                pass

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                result[HydrogelParams.SYMBOL] = self._trade(
                    state, HydrogelParams, fv, hist)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                result[VelvetParams.SYMBOL] = self._trade(
                    state, VelvetParams, velvet_mid, hist)

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    result[P.SYMBOL] = self._trade(
                        state, P, fv, hist)

        return result, 0, json.dumps({"hist": hist})

    def _trade(self, state: TradingState, P, fv: float,
               hist: Dict[str, List[float]]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders

        target_pos = 0
        if P.SLOW_MA > 0 and P.FAST_MA > 0:
            h = hist.get(P.SYMBOL, [])
            h.append(fv)
            if len(h) > P.SLOW_MA:
                h = h[-P.SLOW_MA:]
            hist[P.SYMBOL] = h

            if len(h) >= P.SLOW_MA:
                fast_ma = sum(h[-P.FAST_MA:]) / P.FAST_MA
                slow_ma = sum(h) / len(h)
                if fast_ma > slow_ma:
                    target_pos = P.POS_LIMIT
                elif fast_ma < slow_ma:
                    target_pos = -P.POS_LIMIT

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
