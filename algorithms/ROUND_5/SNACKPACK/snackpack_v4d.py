"""SNACKPACK v4d: PISTA prediction centrally limited by STRAW + RASP.

Key insight from per-day return-OLS (PISTA ~ a*STRAW + b*RASP):
    a = +0.671 / +0.632 / +0.620   (very stable across 3 days)
    b = +0.084 / +0.050 / +0.039   (essentially 0; collinear with STRAW)
    R^2 = 0.835 every day

In returns, PISTA loads ~0.65 on STRAW and ~0 on RASP. But because
STRAW and RASP are themselves anti-correlated (rho=-0.92), we can
re-express PISTA's loading via the factor-B spread (STRAW - RASP):

    PISTA_ret = 0.29 * (STRAW_ret - RASP_ret) + idio
    => PISTA_level = 0.29 * (STRAW - RASP) + drift_term + noise

The 'centrally limited' prediction uses BOTH anti-pair members.
Each contributes equal-magnitude opposite-sign moves to the (STRAW-RASP)
factor; they bracket PISTA's expected position. When STRAW & RASP
diverge in opposite directions (factor B active), PISTA should follow
with weight 0.29. When they don't diverge, prediction is anchored to
the slow drift.

Implementation:
    pred[t] = LOADING * (STRAW[t] - RASP[t]) + drift_ema[t]
    drift_ema[t] = EMA of (PISTA[t] - LOADING*(STRAW[t]-RASP[t]))
    residual = PISTA - pred
    z = residual / max(MIN_STD, sqrt(var_ema(residual)))
    target_PISTA = -k * z   (MR around the prediction)

Compared to v4b's single-regressor f_P = PISTA - 0.59*STRAW: this uses
the cleaner anti-pair difference, which removes the factor B common
drift and isolates true PISTA-specific residual.

Empirical level-OLS resid std (2-regressor): 96/63/54 across days.
Use MIN_STD_P = 60.
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
    HL_MEAN = 4000
    HL_VAR  = 8000
    Z_FULL = 0.2
    Z_AGG  = 2.5
    MIN_STD = {"A": 150.0, "B": 200.0, "P": 60.0}
    PISTA_LOADING = 0.29       # PISTA ret loading on (STRAW-RASP) ret
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
        # PISTA residual after STRAW-RASP factor (level form): isolates idio + drift
        f_P = mids[PISTA] - P.PISTA_LOADING * f_B
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
        # PISTA pred = LOADING*(STRAW-RASP) + drift_EMA. Residual = mid - pred
        # = f_P - ema_P. z_P normalizes that residual. Trade -k*z_P (MR).
        target = {
            CHOC:  _clip(int(round(-scale * z_A)), P.POS_LIMIT),
            VAN:   _clip(int(round(+scale * z_A)), P.POS_LIMIT),
            STRAW: _clip(int(round(-scale * z_B)), P.POS_LIMIT),
            RASP:  _clip(int(round(+scale * z_B)), P.POS_LIMIT),
            PISTA: _clip(int(round(-scale * z_P)), P.POS_LIMIT),
        }

        product_z = {
            CHOC:  abs(z_A),
            VAN:   abs(z_A),
            STRAW: abs(z_B),
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
