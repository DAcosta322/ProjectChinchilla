"""SNACKPACK v3: 2-pair spread MR, passive-only quoting (sweep winner).

Sweep result on days 2/3/4: HL_MEAN=4000, Z_FULL=0.2, Z_AGG=inf
    d2=50,688  d3=32,150  d4=26,202   total = $109,040

Why passive-only beats aggressive: bid-ask spread is ~17 ticks. Crossing
costs 17 ticks per leg. The factor MR signal (z-score on CHOC-VAN and
STRAW-RASP spreads, both ~300-380 sigma) is robust enough that even
patient passive fills capture the reversion. Eating 17-tick spread on
entry burns the edge.

Z_FULL=0.2 is essentially binary: when |z|>0.2, target = +/- POS_LIMIT;
otherwise target = 0. Combined with passive-only execution, this is an
inventory-target market-making strategy where the target flips with
small spread deviations.

Drops PISTA (its residual after STRAW is mostly random walk; v1 with
PISTA factor lost $5K).
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
TRADED = [CHOC, VAN, STRAW, RASP]


class P:
    POS_LIMIT = 10
    HL_MEAN = 4000
    HL_VAR  = 8000
    Z_FULL = 0.2
    Z_AGG  = 99.0     # passive-only
    MIN_STD = 20.0
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
        factors = {"A": f_A, "B": f_B}

        for k, v in factors.items():
            if k not in ema:
                ema[k] = v
                var[k] = 0.0
            else:
                ema[k] = ema[k] + ALPHA_M * (v - ema[k])
                var[k] = (1.0 - ALPHA_V) * var[k] + ALPHA_V * (v - ema[k]) ** 2
        n += 1
        out_state = json.dumps({"ema": ema, "var": var, "n": n})

        if n < P.WARMUP_TICKS:
            return {}, 0, out_state

        def z(k: str) -> float:
            sd = max(P.MIN_STD, var[k] ** 0.5)
            return (factors[k] - ema[k]) / sd

        z_A = z("A")
        z_B = z("B")

        scale = P.POS_LIMIT / P.Z_FULL
        target = {
            CHOC:  _clip(int(round(-scale * z_A)), P.POS_LIMIT),
            VAN:   _clip(int(round(+scale * z_A)), P.POS_LIMIT),
            STRAW: _clip(int(round(-scale * z_B)), P.POS_LIMIT),
            RASP:  _clip(int(round(+scale * z_B)), P.POS_LIMIT),
        }

        product_z = {
            CHOC:  abs(z_A),
            VAN:   abs(z_A),
            STRAW: abs(z_B),
            RASP:  abs(z_B),
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
