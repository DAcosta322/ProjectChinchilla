"""T1 #1: Avellaneda-Stoikov optimal MM.

Reservation price r = mid - q * gamma * sigma^2 * (T-t)
Half-spread     d* = gamma * sigma^2 * (T-t) / 2 + (1/gamma) * ln(1 + gamma/kappa)
  bid = r - d*, ask = r + d*

q = inventory, gamma = risk aversion, sigma^2 = realized variance,
T-t = time remaining (normalized to [0,1]), kappa = order arrival rate.

Replaces our heuristic INV_SKEW + penny-best posting with the closed-form
HJB-derived quotes. Keep aggressive-take loop and MR target overlay
(otherwise we lose fill flow on day 3).

Day = 100k timestamps. (T-t) decreases linearly from 1.0 -> 0.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Any
import json
import math


DAY_TS = 100000.0


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    ANCHOR_SPAN = 500; MR_STRENGTH = 5
    PROFIT_DIST = 40; PROFIT_RANGE = 40
    GAMMA = 0.05            # risk aversion
    KAPPA = 0.5             # arrival intensity
    VOL_SPAN = 200; VOL_REF = 2.0


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    ANCHOR_SPAN = 5000; MR_STRENGTH = 10
    PROFIT_DIST = 15; PROFIT_RANGE = 15
    GAMMA = 0.1
    KAPPA = 1.0
    VOL_SPAN = 0; VOL_REF = 1.0


class VEV4000Params:
    SYMBOL = "VEV_4000"; STRIKE = 4000; POS_LIMIT = 300
    INV_SKEW = 3; SPILLOVER = True
    PROFIT_DIST = 0; PROFIT_RANGE = 0
    OWN_ANCHOR = 1000; OWN_MR = 5


class VEV4500Params:
    SYMBOL = "VEV_4500"; STRIKE = 4500; POS_LIMIT = 300
    INV_SKEW = 3; SPILLOVER = True
    PROFIT_DIST = 0; PROFIT_RANGE = 0
    OWN_ANCHOR = 1000; OWN_MR = 5


class VEV5000Params:
    SYMBOL = "VEV_5000"; STRIKE = 5000; POS_LIMIT = 300
    INV_SKEW = 2; TV_EWMA_SPAN = 50; MIN_SAMPLES = 100
    PROFIT_DIST = 0; PROFIT_RANGE = 0


def _micro_mid(od):
    if not od.buy_orders or not od.sell_orders: return None
    bb = max(od.buy_orders.keys()); ba = min(od.sell_orders.keys())
    bv = od.buy_orders[bb]; av = -od.sell_orders[ba]
    if bv + av <= 0: return (bb + ba) / 2.0
    return (bb * av + ba * bv) / (bv + av)


def _update_avg_entry(prev_avg, prev_pos, own_trades):
    cur_pos = prev_pos
    avg = prev_avg if prev_pos != 0 else 0.0
    for t in own_trades:
        if getattr(t, "buyer", None) == "SUBMISSION": qty = t.quantity
        elif getattr(t, "seller", None) == "SUBMISSION": qty = -t.quantity
        else: continue
        new_pos = cur_pos + qty
        if new_pos == 0: avg = 0.0
        elif cur_pos == 0 or (cur_pos > 0) != (new_pos > 0): avg = float(t.price)
        elif (cur_pos > 0) == (qty > 0): avg = (avg*cur_pos + t.price*qty)/new_pos
        cur_pos = new_pos
    return avg


def _profit_decay(P, pos, fv, avg_entry):
    if avg_entry <= 0 or pos == 0 or P.PROFIT_DIST <= 0: return 1.0
    sign = 1.0 if pos > 0 else -1.0
    profit = (fv - avg_entry) * sign
    if profit <= P.PROFIT_DIST: return 1.0
    return max(0.0, 1.0 - (profit - P.PROFIT_DIST) / max(1.0, P.PROFIT_RANGE))


class Trader:
    def bid(self): return 3337

    def run(self, state):
        result = {}
        td = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: pass
        ema = td.get("ema", {}); avg_entry = td.get("avg", {})
        tv = td.get("tv", {}); tv_n = td.get("tvn", {})
        prev_fv = td.get("pfv", {}); var_ewma = td.get("var", {})

        T_remaining = max(0.05, 1.0 - state.timestamp / DAY_TS)

        for P in (HydrogelParams, VelvetParams):
            if P.PROFIT_DIST <= 0: continue
            sym = P.SYMBOL
            own = state.own_trades.get(sym, []) if state.own_trades else []
            cur_pos = state.position.get(sym, 0)
            delta = sum(t.quantity if getattr(t,"buyer",None)=="SUBMISSION"
                       else -t.quantity if getattr(t,"seller",None)=="SUBMISSION" else 0
                       for t in own)
            avg_entry[sym] = _update_avg_entry(avg_entry.get(sym, 0.0), cur_pos - delta, own)

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                _, orders = self._as_mr(state, HydrogelParams, fv, ema,
                                        avg_entry, prev_fv, var_ewma, T_remaining)
                result[HydrogelParams.SYMBOL] = orders

        velvet_mid = None; velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                tp, orders = self._as_mr(state, VelvetParams, velvet_mid, ema,
                                          avg_entry, prev_fv, var_ewma, T_remaining)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    target = int(round(velvet_target_rel * P.POS_LIMIT))
                    if getattr(P, 'OWN_MR', 0) > 0:
                        prev = ema.get(P.SYMBOL)
                        if prev is None: anchor = fv
                        else:
                            a = 2.0 / (P.OWN_ANCHOR + 1.0)
                            anchor = a * fv + (1 - a) * prev
                        ema[P.SYMBOL] = anchor
                        target += int(round(-P.OWN_MR * (fv - anchor)))
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))
                    result[P.SYMBOL] = self._exec_simple(state, P, fv, target)
            if VEV5000Params.SYMBOL in state.order_depths:
                od5 = state.order_depths[VEV5000Params.SYMBOL]
                vev5_mid = _micro_mid(od5)
                if vev5_mid is not None:
                    intrinsic = velvet_mid - VEV5000Params.STRIKE
                    observed = vev5_mid - intrinsic
                    prev = tv.get(VEV5000Params.SYMBOL)
                    a = 2.0/(VEV5000Params.TV_EWMA_SPAN+1.0)
                    cur = observed if prev is None else a*observed + (1-a)*prev
                    cur = max(-50.0, min(50.0, cur))
                    tv[VEV5000Params.SYMBOL] = cur
                    n = tv_n.get(VEV5000Params.SYMBOL, 0) + 1
                    tv_n[VEV5000Params.SYMBOL] = n
                    if n >= VEV5000Params.MIN_SAMPLES:
                        result[VEV5000Params.SYMBOL] = self._exec_simple(
                            state, VEV5000Params, intrinsic + cur, 0)

        return result, 0, json.dumps({"ema": ema, "avg": avg_entry, "tv": tv, "tvn": tv_n,
                                      "pfv": prev_fv, "var": var_ewma})

    def _as_mr(self, state, P, fv, ema, avg_entry, prev_fv, var_ewma, T):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders: return 0, []
        # Update vol estimate
        if P.VOL_SPAN > 0:
            if P.SYMBOL in prev_fv:
                ret = max(-50.0, min(50.0, fv - prev_fv[P.SYMBOL]))
                prev_var = var_ewma.get(P.SYMBOL, P.VOL_REF * P.VOL_REF)
                a = 2.0 / (P.VOL_SPAN + 1.0)
                var_ewma[P.SYMBOL] = a * (ret * ret) + (1 - a) * prev_var
            prev_fv[P.SYMBOL] = fv
            sigma_sq = max(0.5, var_ewma.get(P.SYMBOL, P.VOL_REF * P.VOL_REF))
        else:
            sigma_sq = P.VOL_REF * P.VOL_REF

        # MR target (combine with A-S reservation price)
        target = 0
        if P.ANCHOR_SPAN > 0:
            prev = ema.get(P.SYMBOL)
            anchor = fv if prev is None else (
                2.0/(P.ANCHOR_SPAN+1.0) * fv
                + (1 - 2.0/(P.ANCHOR_SPAN+1.0)) * prev)
            ema[P.SYMBOL] = anchor
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT,
                int(round(-P.MR_STRENGTH * (fv - anchor)))))

        decay = _profit_decay(P, pos, fv, avg_entry.get(P.SYMBOL, 0.0))
        target = int(round(target * decay))

        # Avellaneda-Stoikov: reservation price + optimal spread
        # q = pos relative to target (we're "long against target" if pos > target)
        q_norm = (pos - target) / P.POS_LIMIT
        gamma = P.GAMMA
        # Reservation: shifts mid against our inventory
        reservation = fv - q_norm * gamma * sigma_sq * T * P.POS_LIMIT
        # Half-spread: time-decaying inventory penalty + arrival-rate base
        kappa = max(0.01, P.KAPPA)
        half_spread = gamma * sigma_sq * T / 2.0 + (1.0 / gamma) * math.log(1.0 + gamma / kappa)
        half_spread = max(1.0, half_spread)
        as_bid = reservation - half_spread
        as_ask = reservation + half_spread

        return target, self._do_orders_as(P, fv, pos, target, od, as_bid, as_ask, reservation)

    def _do_orders_as(self, P, fv, pos, target_pos, od, as_bid, as_ask, reservation):
        orders = []
        best_bid = max(od.buy_orders.keys()); best_ask = min(od.sell_orders.keys())
        buy_cap = P.POS_LIMIT - pos; sell_cap = P.POS_LIMIT + pos

        # Aggressive takes: take any ask BELOW reservation, any bid ABOVE
        for price in sorted(od.sell_orders.keys()):
            if price < reservation and buy_cap > 0:
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty)); buy_cap -= qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > reservation and sell_cap > 0:
                qty = min(od.buy_orders[price], sell_cap)
                orders.append(Order(P.SYMBOL, price, -qty)); sell_cap -= qty

        # Posted quotes at A-S optimal levels, but constrained by book
        our_bid = int(round(as_bid))
        our_ask = int(round(as_ask))
        # Don't post worse than penny-best, don't post inverted
        our_bid = min(our_bid, best_bid + 1)
        our_ask = max(our_ask, best_ask - 1)
        if our_ask <= our_bid: our_ask = our_bid + 1

        if buy_cap > 0: orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0: orders.append(Order(P.SYMBOL, our_ask, -sell_cap))
        return orders

    def _exec_simple(self, state, P, fv, target):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders: return []
        orders = []
        skew = P.INV_SKEW * (pos - target) / P.POS_LIMIT
        fv_eff = fv - skew
        best_bid = max(od.buy_orders.keys()); best_ask = min(od.sell_orders.keys())
        buy_cap = P.POS_LIMIT - pos; sell_cap = P.POS_LIMIT + pos
        for price in sorted(od.sell_orders.keys()):
            if price < fv_eff and buy_cap > 0:
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty)); buy_cap -= qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > fv_eff and sell_cap > 0:
                qty = min(od.buy_orders[price], sell_cap)
                orders.append(Order(P.SYMBOL, price, -qty)); sell_cap -= qty
        base_bid = best_bid + 1; base_ask = best_ask - 1
        if base_bid >= base_ask: base_bid = best_bid; base_ask = best_ask
        shift = int(round(-skew))
        our_bid = min(base_bid + shift, int(fv_eff))
        our_ask = max(base_ask + shift, int(fv_eff) + 1)
        if our_ask <= our_bid: our_ask = our_bid + 1
        if buy_cap > 0: orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0: orders.append(Order(P.SYMBOL, our_ask, -sell_cap))
        return orders
