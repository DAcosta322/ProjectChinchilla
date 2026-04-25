"""Round 3 MR with VEV capacity spillover.

Observation: in the baseline round_3.py, VEV_4000/VEV_4500 are pure
MM (MR_STRENGTH=0) and peak at ~130/300 = ~43% of their position
limit. VELVET on the other hand hits its 200 ceiling 12-19% of
historical days and has strong MR signal. VEVs are delta-1 forwards
on VELVET (fair = S_mid - K), so directional edge on VELVET is also
directional edge on the vouchers.

Change: VEVs inherit VELVET's MR target scaled by their own limit.
If VELVET wants to be full long (+200), VEVs also target full long
(+300 each). This turns 300+300=600 of previously idle VEV capacity
into additional directional exposure aligned with the VELVET signal.

Risk: same signal on three instruments means correlated losses if
the VELVET MR signal is wrong. Backtest will reveal if VELVET's MR
edge is strong enough to carry the extra VEV exposure.

HYDROGEL unchanged from baseline.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
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
    INV_SKEW = 3            # small skew to keep inventory managed
    FOLLOW_VELVET = True


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 3
    FOLLOW_VELVET = True


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
                result[HydrogelParams.SYMBOL] = self._mr_trade(
                    state, HydrogelParams, fv, ema)

        velvet_mid: Optional[float] = None
        velvet_target_rel = 0.0   # VELVET target as fraction of its POS_LIMIT
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                orders, velvet_target_rel = self._mr_trade(
                    state, VelvetParams, velvet_mid, ema,
                    return_target_rel=True)
                result[VelvetParams.SYMBOL] = orders

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    # Scale VELVET's target (as fraction of its limit) to
                    # this voucher's own POS_LIMIT. They move tick-for-tick
                    # with VELVET so the sign and proportional magnitude
                    # of VELVET's signal are the right directional bet.
                    vev_target = int(round(velvet_target_rel * P.POS_LIMIT))
                    vev_target = max(-P.POS_LIMIT,
                                     min(P.POS_LIMIT, vev_target))
                    result[P.SYMBOL] = self._mm_fixed_target(
                        state, P, fv, vev_target)

        return result, 0, json.dumps({"ema": ema})

    def _mr_trade(self, state: TradingState, P, fv: float,
                  ema: Dict[str, float],
                  return_target_rel: bool = False):
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if not od.buy_orders or not od.sell_orders:
            return (orders, 0.0) if return_target_rel else orders

        target_pos = 0
        if P.ANCHOR_SPAN > 0:
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

        self._execute(orders, P, fv, pos, target_pos, od)

        if return_target_rel:
            return orders, target_pos / P.POS_LIMIT
        return orders

    def _mm_fixed_target(self, state: TradingState, P, fv: float,
                         target_pos: int) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return orders
        self._execute(orders, P, fv, pos, target_pos, od)
        return orders

    def _execute(self, orders: List[Order], P, fv: float,
                 pos: int, target_pos: int, od: OrderDepth) -> None:
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
