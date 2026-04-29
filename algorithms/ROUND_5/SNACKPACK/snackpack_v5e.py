"""SNACKPACK v5e: v4a (PISTA pass-through MM) + imbalance lean (v5b idea).

v4a posts symmetric passive MM on PISTA targeting pos=0. v5b uses
PISTA imbalance as a directional signal but loses spread-capture on
the silent side. v5e combines: still post both sides (MM), but bias
sizes by smoothed imbalance — bigger bid when imb negative
(expecting drop, lean against), bigger ask when imb positive.

Wait, the +0.13 correlation means imb > 0 -> price rises.
If we expect a rise, we want to BUY (be long) before it. So bias
inventory toward LONG when imb > 0:
    target_pista = round(POS_LIMIT * imb_smoothed / IMB_FULL), capped
    quote both sides at bb+1/ba-1
    bid_qty larger when target > pos (need to buy more)
    ask_qty larger when target < pos
But also keep some quoting on the other side for spread harvest.

Implementation: target = imbalance-driven, then post asymmetric MM
ladder: 5 contracts toward target, 2 contracts back-side for spread.
"""
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json

CHOC  = "SNACKPACK_CHOCOLATE"
VAN   = "SNACKPACK_VANILLA"
PISTA = "SNACKPACK_PISTACHIO"
STRAW = "SNACKPACK_STRAWBERRY"
RASP  = "SNACKPACK_RASPBERRY"
PRODUCTS = [CHOC, VAN, PISTA, STRAW, RASP]
PAIRS = [CHOC, VAN, STRAW, RASP]


class P:
    POS_LIMIT = 10
    HL_MEAN = 4000
    HL_VAR  = 8000
    Z_FULL = 0.2
    Z_AGG  = 2.5
    MIN_STD = {"A": 150.0, "B": 200.0}
    WARMUP_TICKS = 200
    MIN_GAP = 1
    # PISTA MM with imbalance lean
    HL_IMB = 200            # imbalance signal contributes ~zero value;
    IMB_FULL = 1.0          # the real edge is the asymmetric MM ladder below
    PISTA_MAIN_QTY = 8      # size on the inventory-restoring side
    PISTA_OPP_QTY  = 5      # size on the opposite side (spread harvest)


ALPHA_M = 1.0 - 0.5 ** (1.0 / P.HL_MEAN)
ALPHA_V = 1.0 - 0.5 ** (1.0 / P.HL_VAR)
ALPHA_IMB = 1.0 - 0.5 ** (1.0 / P.HL_IMB)


def _book(od):
    if not od or not od.buy_orders or not od.sell_orders: return None
    bb = max(od.buy_orders.keys()); ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]

def _clip(x, lim): return max(-lim, min(lim, x))


class Trader:
    def run(self, state):
        try: data = json.loads(state.traderData) if state.traderData else {}
        except: data = {}
        ema = data.get("ema", {}); var = data.get("var", {}); n = data.get("n", 0)
        imb_pista = data.get("imb_pista", 0.0)

        mids = {}; books = {}
        for p in PRODUCTS:
            r = _book(state.order_depths.get(p))
            if r is None: continue
            bb, ba, bv, av = r
            mids[p] = (bb+ba)/2.0; books[p] = (bb,ba,bv,av)
        if not all(p in mids for p in PAIRS):
            return {}, 0, json.dumps({"ema":ema,"var":var,"n":n,"imb_pista":imb_pista})

        f_A = mids[CHOC] - mids[VAN]; f_B = mids[STRAW] - mids[RASP]
        factors = {"A": f_A, "B": f_B}
        for k, v in factors.items():
            if k not in ema:
                ema[k] = v; var[k] = P.MIN_STD[k]**2
            else:
                ema[k] = ema[k] + ALPHA_M*(v - ema[k])
                var[k] = (1-ALPHA_V)*var[k] + ALPHA_V*(v - ema[k])**2

        if PISTA in books:
            bb_p, ba_p, bv_p, av_p = books[PISTA]
            raw_imb = (bv_p - av_p) / max(bv_p + av_p, 1)
            imb_pista = imb_pista + ALPHA_IMB * (raw_imb - imb_pista)

        n += 1
        out_state = json.dumps({"ema":ema,"var":var,"n":n,"imb_pista":imb_pista})
        if n < P.WARMUP_TICKS: return {}, 0, out_state

        def z(k):
            sd = max(P.MIN_STD[k], var[k]**0.5)
            return (factors[k] - ema[k]) / sd
        z_A = z("A"); z_B = z("B")
        scale = P.POS_LIMIT / P.Z_FULL
        target = {
            CHOC:  _clip(int(round(-scale*z_A)), P.POS_LIMIT),
            VAN:   _clip(int(round(+scale*z_A)), P.POS_LIMIT),
            STRAW: _clip(int(round(-scale*z_B)), P.POS_LIMIT),
            RASP:  _clip(int(round(+scale*z_B)), P.POS_LIMIT),
        }
        product_z = {CHOC:abs(z_A), VAN:abs(z_A), STRAW:abs(z_B), RASP:abs(z_B)}

        # PISTA target from imbalance
        pista_scale = P.POS_LIMIT / P.IMB_FULL
        target_pista = _clip(int(round(pista_scale * imb_pista)), P.POS_LIMIT)

        all_p = PAIRS + [PISTA]
        orders = {p: [] for p in all_p}
        buy_used = {p: 0 for p in all_p}; sell_used = {p: 0 for p in all_p}
        def buy_cap(p): return P.POS_LIMIT - state.position.get(p,0) - buy_used[p]
        def sell_cap(p): return P.POS_LIMIT + state.position.get(p,0) - sell_used[p]

        # ---- Pair MR (unchanged from v4) ----
        for p in PAIRS:
            bb, ba, bv, av = books[p]
            pos = state.position.get(p,0); tgt = target[p]; gap = tgt - pos
            if abs(gap) < P.MIN_GAP: continue
            aggressive = product_z[p] >= P.Z_AGG
            if gap > 0:
                if aggressive:
                    fill = min(gap, av, buy_cap(p))
                    if fill > 0:
                        orders[p].append(Order(p, ba, fill)); buy_used[p] += fill
                rem = tgt - (pos + buy_used[p])
                if rem > 0:
                    px = bb+1 if bb+1 < ba else bb
                    cap = min(rem, buy_cap(p))
                    if cap > 0:
                        orders[p].append(Order(p, px, cap)); buy_used[p] += cap
            else:
                need = -gap
                if aggressive:
                    fill = min(need, bv, sell_cap(p))
                    if fill > 0:
                        orders[p].append(Order(p, bb, -fill)); sell_used[p] += fill
                rem = (pos - sell_used[p]) - tgt
                if rem > 0:
                    px = ba-1 if ba-1 > bb else ba
                    cap = min(rem, sell_cap(p))
                    if cap > 0:
                        orders[p].append(Order(p, px, -cap)); sell_used[p] += cap

        # ---- PISTA: asymmetric MM with imbalance lean ----
        if PISTA in books:
            bb, ba, bv, av = books[PISTA]
            pos = state.position.get(PISTA, 0)
            gap = target_pista - pos
            # Both-sided MM, with main side larger
            if gap >= 0:  # need to buy more (or hold)
                bid_q = P.PISTA_MAIN_QTY
                ask_q = P.PISTA_OPP_QTY
            else:
                bid_q = P.PISTA_OPP_QTY
                ask_q = P.PISTA_MAIN_QTY
            if bb + 1 < ba and bid_q > 0:
                cap = min(bid_q, buy_cap(PISTA))
                if cap > 0: orders[PISTA].append(Order(PISTA, bb+1, cap))
            if ba - 1 > bb and ask_q > 0:
                cap = min(ask_q, sell_cap(PISTA))
                if cap > 0: orders[PISTA].append(Order(PISTA, ba-1, -cap))

        orders = {p: o for p, o in orders.items() if o}
        return orders, 0, out_state
