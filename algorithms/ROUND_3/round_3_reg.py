"""Round 3 MR + REGRESSION-BOOST at troughs.

Base is round_3_eff.py: baseline MR target = -MR_STRENGTH*(fv-anchor),
plus VEVs inherit VELVET's relative target.

Overlay: fit a linear regression to the last WINDOW ticks. The slope
indicates whether price has been rising or falling recently. At a
forming trough the recent slope is steeply negative, so we scale our
LONG target UP by -BOOST_GAIN * slope (adding on top of MR_STRENGTH's
contribution). Similarly a positive slope (peak forming) adds to the
short target.

  raw_target = -MR_STRENGTH * (fv - anchor)  -  BOOST_GAIN * slope

Both terms have the same sign for a classic MR dip (fv < anchor AND
slope < 0), so they REINFORCE - we load more at troughs. Both go
zero in flat conditions so historicals preserve baseline edge.
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
    REG_WINDOW = 30
    BOOST_GAIN = 40
    BOOST_THRESHOLD = 0.5    # only boost when |slope| exceeds this


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10
    REG_WINDOW = 30
    BOOST_GAIN = 60
    BOOST_THRESHOLD = 0.2


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 3


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 3


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


def _ols_slope(ys: List[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    num = 0.0
    den = 0.0
    my = sum(ys) / n
    for i, y in enumerate(ys):
        dx = i - mx
        num += dx * (y - my)
        den += dx * dx
    return num / den if den > 0 else 0.0


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
        hist: Dict[str, List[float]] = td.get("hist", {})
        for P in (HydrogelParams, VelvetParams):
            hist.setdefault(P.SYMBOL, [])

        velvet_target_rel = 0.0

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                tp = self._mr_target(HydrogelParams, fv, ema,
                                     hist[HydrogelParams.SYMBOL])
                result[HydrogelParams.SYMBOL] = self._orders(
                    state, HydrogelParams, fv, tp)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                tp = self._mr_target(VelvetParams, velvet_mid, ema,
                                     hist[VelvetParams.SYMBOL])
                result[VelvetParams.SYMBOL] = self._orders(
                    state, VelvetParams, velvet_mid, tp)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    vev_target = int(round(velvet_target_rel * P.POS_LIMIT))
                    vev_target = max(-P.POS_LIMIT,
                                     min(P.POS_LIMIT, vev_target))
                    result[P.SYMBOL] = self._orders(
                        state, P, fv, vev_target)

        return result, 0, json.dumps({"ema": ema, "hist": hist})

    def _mr_target(self, P, fv: float, ema: Dict[str, float],
                   hist: List[float]) -> int:
        if P.ANCHOR_SPAN <= 0:
            return 0
        prev = ema.get(P.SYMBOL)
        if prev is None:
            anchor = fv
        else:
            alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
            anchor = alpha * fv + (1 - alpha) * prev
        ema[P.SYMBOL] = anchor

        mr_term = -P.MR_STRENGTH * (fv - anchor)

        # Regression-based trough/peak boost
        hist.append(fv)
        if len(hist) > P.REG_WINDOW:
            del hist[:len(hist) - P.REG_WINDOW]

        # Boost only fires on "unusual" slopes (|slope| > threshold);
        # in normal chop the slope is noise and contributes nothing.
        boost_term = 0.0
        if len(hist) >= P.REG_WINDOW:
            slope = _ols_slope(hist)
            if abs(slope) > P.BOOST_THRESHOLD:
                adj = slope - P.BOOST_THRESHOLD if slope > 0 else slope + P.BOOST_THRESHOLD
                boost_term = -P.BOOST_GAIN * adj

        raw = mr_term + boost_term
        return max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))

    def _orders(self, state: TradingState, P, fv: float,
                target_pos: int) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return orders

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
