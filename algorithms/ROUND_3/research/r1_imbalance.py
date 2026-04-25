"""R1: Order book imbalance signal on top of eff4.

target += IMBALANCE_GAIN * (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)

Signal is bounded in [-1, +1]. When bids stacked thicker -> bias long.
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
    PROFIT_DIST = 40
    PROFIT_RANGE = 40
    IMBALANCE_GAIN = 30


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10
    PROFIT_DIST = 15
    PROFIT_RANGE = 15
    IMBALANCE_GAIN = 30


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True
    PROFIT_DIST = 0
    PROFIT_RANGE = 0
    IMBALANCE_GAIN = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True
    PROFIT_DIST = 0
    PROFIT_RANGE = 0
    IMBALANCE_GAIN = 0


class VEV5000Params:
    SYMBOL = "VEV_5000"
    STRIKE = 5000
    POS_LIMIT = 300
    INV_SKEW = 2
    TV_EWMA_SPAN = 100
    MIN_SAMPLES = 100
    PROFIT_DIST = 0
    PROFIT_RANGE = 0
    IMBALANCE_GAIN = 0


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


def _imbalance(od: OrderDepth) -> float:
    if not od.buy_orders or not od.sell_orders:
        return 0.0
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    bv = od.buy_orders[bb]
    av = -od.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.0
    return (bv - av) / tot


def _update_avg_entry(prev_avg: float, prev_pos: int, own_trades: list):
    cur_pos = prev_pos
    avg = prev_avg if prev_pos != 0 else 0.0
    for t in own_trades:
        if getattr(t, "buyer", None) == "SUBMISSION":
            qty = t.quantity
        elif getattr(t, "seller", None) == "SUBMISSION":
            qty = -t.quantity
        else:
            continue
        new_pos = cur_pos + qty
        if new_pos == 0:
            avg = 0.0
        elif cur_pos == 0 or (cur_pos > 0) != (new_pos > 0):
            avg = float(t.price)
        elif (cur_pos > 0) == (qty > 0):
            avg = (avg * cur_pos + t.price * qty) / new_pos
        cur_pos = new_pos
    return avg


def _profit_decay(P, pos: int, fv: float, avg_entry: float) -> float:
    if avg_entry <= 0 or pos == 0 or P.PROFIT_DIST <= 0:
        return 1.0
    sign = 1.0 if pos > 0 else -1.0
    profit_per_unit = (fv - avg_entry) * sign
    if profit_per_unit <= P.PROFIT_DIST:
        return 1.0
    excess = profit_per_unit - P.PROFIT_DIST
    return max(0.0, 1.0 - excess / max(1.0, P.PROFIT_RANGE))


class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        td: Dict[str, Any] = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except Exception: td = {}
        ema = td.get("ema", {})
        avg_entry = td.get("avg", {})
        tv = td.get("tv", {})
        tv_n = td.get("tvn", {})

        for P in (HydrogelParams, VelvetParams):
            if P.PROFIT_DIST <= 0: continue
            sym = P.SYMBOL
            own = state.own_trades.get(sym, []) if state.own_trades else []
            cur_pos = state.position.get(sym, 0)
            delta = sum(t.quantity if getattr(t,"buyer",None)=="SUBMISSION"
                       else -t.quantity if getattr(t,"seller",None)=="SUBMISSION" else 0
                       for t in own)
            pre_pos = cur_pos - delta
            avg_entry[sym] = _update_avg_entry(avg_entry.get(sym, 0.0), pre_pos, own)

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                _, orders = self._mr(state, HydrogelParams, fv, ema, avg_entry)
                result[HydrogelParams.SYMBOL] = orders

        velvet_mid = None
        velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                tp, orders = self._mr(state, VelvetParams, velvet_mid, ema, avg_entry)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    target = int(round(velvet_target_rel * P.POS_LIMIT))
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))
                    result[P.SYMBOL] = self._exec(state, P, fv, target)

            if VEV5000Params.SYMBOL in state.order_depths:
                od5 = state.order_depths[VEV5000Params.SYMBOL]
                vev5_mid = _micro_mid(od5)
                if vev5_mid is not None:
                    intrinsic = velvet_mid - VEV5000Params.STRIKE
                    observed = vev5_mid - intrinsic
                    prev = tv.get(VEV5000Params.SYMBOL)
                    cur = observed if prev is None else (
                        2.0/(VEV5000Params.TV_EWMA_SPAN+1.0) * observed
                        + (1 - 2.0/(VEV5000Params.TV_EWMA_SPAN+1.0)) * prev)
                    tv[VEV5000Params.SYMBOL] = cur
                    n = tv_n.get(VEV5000Params.SYMBOL, 0) + 1
                    tv_n[VEV5000Params.SYMBOL] = n
                    if n >= VEV5000Params.MIN_SAMPLES:
                        result[VEV5000Params.SYMBOL] = self._exec(
                            state, VEV5000Params, intrinsic + cur, 0)

        return result, 0, json.dumps({"ema": ema, "avg": avg_entry, "tv": tv, "tvn": tv_n})

    def _mr(self, state, P, fv, ema, avg_entry):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return 0, []
        target = 0
        if P.ANCHOR_SPAN > 0:
            prev = ema.get(P.SYMBOL)
            anchor = fv if prev is None else (
                2.0/(P.ANCHOR_SPAN+1.0) * fv
                + (1 - 2.0/(P.ANCHOR_SPAN+1.0)) * prev)
            ema[P.SYMBOL] = anchor
            raw = -P.MR_STRENGTH * (fv - anchor)
            # NEW: imbalance overlay
            if P.IMBALANCE_GAIN > 0:
                raw += P.IMBALANCE_GAIN * _imbalance(od)
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))
        decay = _profit_decay(P, pos, fv, avg_entry.get(P.SYMBOL, 0.0))
        target = int(round(target * decay))
        return target, self._do_orders(P, fv, pos, target, od)

    def _exec(self, state, P, fv, target):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders: return []
        return self._do_orders(P, fv, pos, target, od)

    def _do_orders(self, P, fv, pos, target_pos, od):
        orders = []
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
        our_bid = min(base_bid + shift, int(fv_eff))
        our_ask = max(base_ask + shift, int(fv_eff) + 1)
        if our_ask <= our_bid: our_ask = our_bid + 1
        if buy_cap > 0: orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0: orders.append(Order(P.SYMBOL, our_ask, -sell_cap))
        return orders
