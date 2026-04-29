"""Round 5 — PEBBLES static directional bet (sanity check).

Hypothesis: every backtested day shows XL up, XS down. Pure static
bet: target_pos[XL] = +10, target_pos[XS] = -10, others = 0. Build
positions via aggressive crossing on first tick, hold to EOD.

If this beats the dynamic strategies, the trend signal direction is
correct but the v1-v5 dynamics (MM bleed, vel whipsaw) are noise. If
this loses too, the directional pattern doesn't hold once you pay
spread.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class P:
    POS_LIMIT = 10
    TARGETS = {
        "PEBBLES_XS": -10,
        "PEBBLES_S":   0,
        "PEBBLES_M":   0,
        "PEBBLES_L":   0,
        "PEBBLES_XL": +10,
    }


def best_bid_ask(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


class Trader:
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        for p, target in P.TARGETS.items():
            r = best_bid_ask(state.order_depths.get(p))
            if r is None:
                continue
            bb, ba, bv, av = r
            pos = state.position.get(p, 0)
            gap = target - pos
            if gap > 0:
                qty = min(int(gap), av, P.POS_LIMIT - pos)
                if qty > 0:
                    orders[p] = [Order(p, ba, qty)]
            elif gap < 0:
                qty = min(int(-gap), bv, P.POS_LIMIT + pos)
                if qty > 0:
                    orders[p] = [Order(p, bb, -qty)]
        return orders, 0, ""
