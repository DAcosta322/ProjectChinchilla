"""Round 3 EFFICIENCY v2 - adaptive INV_SKEW on the underlyings only.

After testing options 1-3 on top of round_3_eff.py:
  - Cross-voucher arb: ZERO opportunities (book always priced 507/492
    against the parity gap, no free spread).
  - VEV_5000 / VEV_5100 MM: TV drifts 3-4 ticks day-to-day, fixed
    TV_OFFSET produces big losses (-$50K+ per day on VEV_5100).
  - Adaptive INV_SKEW = spread on VEV_4000/4500: spread is wide (~21)
    so INV_SKEW=21 is way too aggressive vs eff.py's tuned INV_SKEW=3,
    blew up VEV_4000 PnL.

What survives: adaptive INV_SKEW on HYDROGEL and VELVET only, where
spread is moderate and the underlyings actually need to cross the
book to express MR signals. INV_SKEW = max(MIN, observed_spread).
VEV_4000 / VEV_4500 keep eff.py's tuned INV_SKEW=3.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW_MIN = 12       # adaptive: max(MIN, spread)
    ANCHOR_SPAN = 500
    MR_STRENGTH = 5
    ADAPTIVE = True


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW_MIN = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10
    ADAPTIVE = True


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW_MIN = 3
    ADAPTIVE = False        # eff.py value works - wide spread mismatches


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW_MIN = 3
    ADAPTIVE = False


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


def _skew_unit(P, od: OrderDepth) -> float:
    if not P.ADAPTIVE:
        return float(P.INV_SKEW_MIN)
    spread = min(od.sell_orders.keys()) - max(od.buy_orders.keys())
    return max(float(P.INV_SKEW_MIN), float(spread))


class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

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
                _, orders = self._mr(state, HydrogelParams, fv, ema)
                result[HydrogelParams.SYMBOL] = orders

        velvet_mid: Optional[float] = None
        velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                tp, orders = self._mr(state, VelvetParams, velvet_mid, ema)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    target = int(round(velvet_target_rel * P.POS_LIMIT))
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))
                    result[P.SYMBOL] = self._exec(state, P, fv, target)

        return result, 0, json.dumps({"ema": ema})

    def _mr(self, state: TradingState, P, fv: float,
            ema: Dict[str, float]):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return 0, []
        target = 0
        if P.ANCHOR_SPAN > 0:
            prev = ema.get(P.SYMBOL)
            if prev is None:
                anchor = fv
            else:
                alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
                anchor = alpha * fv + (1 - alpha) * prev
            ema[P.SYMBOL] = anchor
            raw = -P.MR_STRENGTH * (fv - anchor)
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))
        return target, self._do_orders(P, fv, pos, target, od)

    def _exec(self, state: TradingState, P, fv: float, target: int) -> List[Order]:
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return []
        return self._do_orders(P, fv, pos, target, od)

    def _do_orders(self, P, fv: float, pos: int, target_pos: int,
                   od: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        skew_unit = _skew_unit(P, od)
        skew = skew_unit * (pos - target_pos) / P.POS_LIMIT
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
