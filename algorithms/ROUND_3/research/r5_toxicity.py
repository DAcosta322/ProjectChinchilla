"""R5: Adverse-selection / toxicity detection on top of eff4.

After each fill, watch the price move against us. Track an EWMA of
adverse_move per recent fill. When this 'toxicity' is high, scale
down our willingness to trade by tightening fv_eff so passive quotes
sit further from the book and aggressive takes only fire on bigger
mispricings.

  toxicity_t = EWMA( max(0, adverse_move_per_unit) )
  size_factor = max(0.2, 1 - toxicity / TOX_FULL_OFF)
  effective INV_SKEW *= size_factor   (less crossing when toxic)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Any
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200; INV_SKEW = 15
    ANCHOR_SPAN = 500; MR_STRENGTH = 5
    PROFIT_DIST = 40; PROFIT_RANGE = 40
    TOX_SPAN = 50
    TOX_FULL_OFF = 8        # adverse-per-unit at which we fully retract


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200; INV_SKEW = 3
    ANCHOR_SPAN = 5000; MR_STRENGTH = 10
    PROFIT_DIST = 15; PROFIT_RANGE = 15
    TOX_SPAN = 50
    TOX_FULL_OFF = 4


class VEV4000Params:
    SYMBOL = "VEV_4000"; STRIKE = 4000; POS_LIMIT = 300
    INV_SKEW = 3; SPILLOVER = True
    PROFIT_DIST = 0; PROFIT_RANGE = 0
    TOX_SPAN = 0; TOX_FULL_OFF = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"; STRIKE = 4500; POS_LIMIT = 300
    INV_SKEW = 3; SPILLOVER = True
    PROFIT_DIST = 0; PROFIT_RANGE = 0
    TOX_SPAN = 0; TOX_FULL_OFF = 0


class VEV5000Params:
    SYMBOL = "VEV_5000"; STRIKE = 5000; POS_LIMIT = 300
    INV_SKEW = 2; TV_EWMA_SPAN = 100; MIN_SAMPLES = 100
    PROFIT_DIST = 0; PROFIT_RANGE = 0
    TOX_SPAN = 0; TOX_FULL_OFF = 0


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


def _update_toxicity(P, prev_tox: float, own_trades: list, fv_now: float) -> float:
    """Adverse move (per unit) on each just-filled trade vs current fv."""
    if not own_trades or P.TOX_SPAN <= 0:
        return prev_tox * 0.95   # gentle decay each tick if no fills
    bad = 0.0
    n = 0
    for t in own_trades:
        if getattr(t, "buyer", None) == "SUBMISSION":
            adv = max(0.0, t.price - fv_now)   # we bought; price now lower = adverse
        elif getattr(t, "seller", None) == "SUBMISSION":
            adv = max(0.0, fv_now - t.price)   # we sold; price now higher = adverse
        else:
            continue
        bad += adv
        n += 1
    if n == 0:
        return prev_tox * 0.95
    avg_adverse = bad / n
    a = 2.0 / (P.TOX_SPAN + 1.0)
    return a * avg_adverse + (1 - a) * prev_tox


class Trader:
    def bid(self): return 3337

    def run(self, state):
        result = {}
        td = {}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: pass
        ema = td.get("ema", {}); avg_entry = td.get("avg", {})
        tox = td.get("tox", {}); tv = td.get("tv", {}); tv_n = td.get("tvn", {})

        for P in (HydrogelParams, VelvetParams):
            sym = P.SYMBOL
            if P.PROFIT_DIST <= 0 and P.TOX_FULL_OFF <= 0: continue
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
                own = state.own_trades.get(HydrogelParams.SYMBOL, []) if state.own_trades else []
                tox[HydrogelParams.SYMBOL] = _update_toxicity(
                    HydrogelParams, tox.get(HydrogelParams.SYMBOL, 0.0), own, fv)
                _, orders = self._mr(state, HydrogelParams, fv, ema, avg_entry, tox)
                result[HydrogelParams.SYMBOL] = orders

        velvet_mid = None; velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                own = state.own_trades.get(VelvetParams.SYMBOL, []) if state.own_trades else []
                tox[VelvetParams.SYMBOL] = _update_toxicity(
                    VelvetParams, tox.get(VelvetParams.SYMBOL, 0.0), own, velvet_mid)
                tp, orders = self._mr(state, VelvetParams, velvet_mid, ema, avg_entry, tox)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT,
                                int(round(velvet_target_rel * P.POS_LIMIT))))
                    result[P.SYMBOL] = self._exec(state, P, fv, target)
            if VEV5000Params.SYMBOL in state.order_depths:
                od5 = state.order_depths[VEV5000Params.SYMBOL]
                vev5_mid = _micro_mid(od5)
                if vev5_mid is not None:
                    intrinsic = velvet_mid - VEV5000Params.STRIKE
                    observed = vev5_mid - intrinsic
                    prev = tv.get(VEV5000Params.SYMBOL)
                    a = 2.0/(VEV5000Params.TV_EWMA_SPAN+1.0)
                    cur = observed if prev is None else a*observed + (1-a)*prev
                    tv[VEV5000Params.SYMBOL] = cur
                    n = tv_n.get(VEV5000Params.SYMBOL, 0) + 1
                    tv_n[VEV5000Params.SYMBOL] = n
                    if n >= VEV5000Params.MIN_SAMPLES:
                        result[VEV5000Params.SYMBOL] = self._exec(
                            state, VEV5000Params, intrinsic + cur, 0)

        return result, 0, json.dumps({"ema": ema, "avg": avg_entry, "tox": tox,
                                      "tv": tv, "tvn": tv_n})

    def _mr(self, state, P, fv, ema, avg_entry, tox):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders: return 0, []
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
        # Toxicity scaling: when toxic, reduce effective skew (less crossing)
        tox_factor = 1.0
        if P.TOX_FULL_OFF > 0:
            tox_factor = max(0.2, 1.0 - tox.get(P.SYMBOL, 0.0) / P.TOX_FULL_OFF)
        return target, self._do_orders(P, fv, pos, target, od, tox_factor)

    def _exec(self, state, P, fv, target):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders: return []
        return self._do_orders(P, fv, pos, target, od, 1.0)

    def _do_orders(self, P, fv, pos, target_pos, od, skew_factor=1.0):
        orders = []
        skew = P.INV_SKEW * skew_factor * (pos - target_pos) / P.POS_LIMIT
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
