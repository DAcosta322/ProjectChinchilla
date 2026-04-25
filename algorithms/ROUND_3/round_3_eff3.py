"""Round 3 EFFICIENCY v3 - eff.py + VEV_5000 with adaptive TV.

The blocker for trading VEV_5000 was day-to-day TV drift: mean
residual VEV_5000 - (VE - K) shifted from +6.75 -> +4.87 -> +3.15 ->
+3.22 across the four days. A fixed TV_OFFSET overshoots on later days
and we bleed via adverse fills.

This version tracks TV with an EWMA per tick:
  tv_t = alpha * (vev_mid - (ve_mid - K)) + (1 - alpha) * tv_{t-1}
  fair = ve_mid - STRIKE + tv_t

Within-day std of the residual is ~1, so a long EWMA span (500)
smooths to within ~0.5 of the true mean - enough that our quotes
sit on the right side of the book.

We refuse to trade VEV_5000 until MIN_SAMPLES updates have happened,
so the EWMA has stabilized. We use a higher INV_SKEW so inventory
exits faster than baseline (we don't trust the TV estimate enough to
sit on big positions). VEV_5000 has no spillover - it has moneyness
sensitivity that doesn't move tick-for-tick with VELVET.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Any
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    ANCHOR_SPAN = 500
    MR_STRENGTH = 5


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True


class VEV5000Params:
    SYMBOL = "VEV_5000"
    STRIKE = 5000
    POS_LIMIT = 300
    INV_SKEW = 2            # ~1/3 of spread
    TV_EWMA_SPAN = 100      # short - TV drifts intra-day, need fast tracking
    MIN_SAMPLES = 100


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

        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}
        ema: Dict[str, float] = td.get("ema", {})
        tv: Dict[str, float] = td.get("tv", {})
        tv_n: Dict[str, int] = td.get("tvn", {})

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

            # VEV_5000 with adaptive TV
            if VEV5000Params.SYMBOL in state.order_depths:
                od5 = state.order_depths[VEV5000Params.SYMBOL]
                vev5_mid = _micro_mid(od5)
                if vev5_mid is not None:
                    intrinsic = velvet_mid - VEV5000Params.STRIKE
                    observed_resid = vev5_mid - intrinsic
                    prev_tv = tv.get(VEV5000Params.SYMBOL)
                    if prev_tv is None:
                        cur_tv = observed_resid
                    else:
                        alpha = 2.0 / (VEV5000Params.TV_EWMA_SPAN + 1.0)
                        cur_tv = (alpha * observed_resid
                                  + (1 - alpha) * prev_tv)
                    tv[VEV5000Params.SYMBOL] = cur_tv
                    n = tv_n.get(VEV5000Params.SYMBOL, 0) + 1
                    tv_n[VEV5000Params.SYMBOL] = n

                    if n >= VEV5000Params.MIN_SAMPLES:
                        fair = intrinsic + cur_tv
                        result[VEV5000Params.SYMBOL] = self._exec(
                            state, VEV5000Params, fair, 0)

        return result, 0, json.dumps({"ema": ema, "tv": tv, "tvn": tv_n})

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
