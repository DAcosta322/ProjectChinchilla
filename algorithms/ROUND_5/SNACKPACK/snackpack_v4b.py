"""SNACKPACK v4b: v4 + PISTA factor MR (option B).

Adds f_P = PISTA - 0.59*STRAW back as a factor with realistic MIN_STD=80
(empirical f_P ema-resid std: p10 across days 66-199, p50 75-221).

Position derivation (combines factor A, B, P contributions):
    target[CHOC]  = -k * z_A
    target[VAN]   = +k * z_A
    target[STRAW] = -k * z_B + BETA * k * z_P  (B + P hedge)
    target[RASP]  = +k * z_B
    target[PISTA] = -k * z_P

This is what v1 did but with realistic stds and tuned Z_FULL/Z_AGG.
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
TRADED   = PRODUCTS


class P:
    POS_LIMIT = 10
    BETA_PISTA = 0.59
    HL_MEAN = 4000
    HL_VAR  = 8000
    Z_FULL = 0.2
    Z_AGG  = 2.5
    MIN_STD = {"A": 150.0, "B": 200.0, "P": 80.0}
    WARMUP_TICKS = 200
    MIN_GAP = 1


ALPHA_M = 1.0 - 0.5 ** (1.0 / P.HL_MEAN)
ALPHA_V = 1.0 - 0.5 ** (1.0 / P.HL_VAR)


def _book(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


def _clip(x: int, lim: int) -> int:
    return max(-lim, min(lim, x))


class Trader:
    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        ema: Dict[str, float] = data.get("ema", {})
        var: Dict[str, float] = data.get("var", {})
        n: int = data.get("n", 0)

        mids: Dict[str, float] = {}
        books: Dict[str, tuple] = {}
        for p in PRODUCTS:
            r = _book(state.order_depths.get(p))
            if r is None:
                continue
            bb, ba, bv, av = r
            mids[p] = (bb + ba) / 2.0
            books[p] = (bb, ba, bv, av)

        if not all(p in mids for p in TRADED):
            return {}, 0, json.dumps({"ema": ema, "var": var, "n": n})

        f_A = mids[CHOC] - mids[VAN]
        f_B = mids[STRAW] - mids[RASP]
        f_P = mids[PISTA] - P.BETA_PISTA * mids[STRAW]
        factors = {"A": f_A, "B": f_B, "P": f_P}

        for k, v in factors.items():
            if k not in ema:
                ema[k] = v
                var[k] = P.MIN_STD[k] ** 2
            else:
                ema[k] = ema[k] + ALPHA_M * (v - ema[k])
                var[k] = (1.0 - ALPHA_V) * var[k] + ALPHA_V * (v - ema[k]) ** 2
        n += 1
        out_state = json.dumps({"ema": ema, "var": var, "n": n})

        if n < P.WARMUP_TICKS:
            return {}, 0, out_state

        def z(k: str) -> float:
            sd = max(P.MIN_STD[k], var[k] ** 0.5)
            return (factors[k] - ema[k]) / sd

        z_A = z("A")
        z_B = z("B")
        z_P = z("P")

        scale = P.POS_LIMIT / P.Z_FULL
        target = {
            CHOC:  _clip(int(round(-scale * z_A)), P.POS_LIMIT),
            VAN:   _clip(int(round(+scale * z_A)), P.POS_LIMIT),
            STRAW: _clip(int(round(-scale * z_B + P.BETA_PISTA * scale * z_P)), P.POS_LIMIT),
            RASP:  _clip(int(round(+scale * z_B)), P.POS_LIMIT),
            PISTA: _clip(int(round(-scale * z_P)), P.POS_LIMIT),
        }

        product_z = {
            CHOC:  abs(z_A),
            VAN:   abs(z_A),
            STRAW: max(abs(z_B), abs(z_P)),
            RASP:  abs(z_B),
            PISTA: abs(z_P),
        }

        orders: Dict[str, List[Order]] = {p: [] for p in TRADED}
        buy_used  = {p: 0 for p in TRADED}
        sell_used = {p: 0 for p in TRADED}

        def buy_cap(p: str) -> int:
            return P.POS_LIMIT - state.position.get(p, 0) - buy_used[p]

        def sell_cap(p: str) -> int:
            return P.POS_LIMIT + state.position.get(p, 0) - sell_used[p]

        for p in TRADED:
            bb, ba, bv, av = books[p]
            pos = state.position.get(p, 0)
            tgt = target[p]
            gap = tgt - pos
            if abs(gap) < P.MIN_GAP:
                continue

            aggressive = product_z[p] >= P.Z_AGG

            if gap > 0:
                if aggressive:
                    fill = min(gap, av, buy_cap(p))
                    if fill > 0:
                        orders[p].append(Order(p, ba, fill))
                        buy_used[p] += fill
                rem = tgt - (pos + buy_used[p])
                if rem > 0:
                    px = bb + 1 if bb + 1 < ba else bb
                    cap = min(rem, buy_cap(p))
                    if cap > 0:
                        orders[p].append(Order(p, px, cap))
                        buy_used[p] += cap
            else:
                need = -gap
                if aggressive:
                    fill = min(need, bv, sell_cap(p))
                    if fill > 0:
                        orders[p].append(Order(p, bb, -fill))
                        sell_used[p] += fill
                rem = (pos - sell_used[p]) - tgt
                if rem > 0:
                    px = ba - 1 if ba - 1 > bb else ba
                    cap = min(rem, sell_cap(p))
                    if cap > 0:
                        orders[p].append(Order(p, px, -cap))
                        sell_used[p] += cap

        orders = {p: o for p, o in orders.items() if o}
        return orders, 0, out_state
