"""Round 5 PEBBLES — Baseline D: ETF arb only, no per-product MM.

Behavior:
  - Compute sum_bid and sum_ask across all 5 PEBBLES.
  - sum_bid > 50000 → SELL all 5 at their bids (lock instant profit).
  - sum_ask < 50000 → BUY all 5 at their asks (lock instant profit).
  - That's it. No per-product trading, no MM, no anchor.

The pure "ETF holding" strategy from v3. Useful as a sanity-check baseline
that locks zero directional risk per fragment. Expected to be small but
strictly positive on platform.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10
    BASKET_QTY = 10


def best_bid_ask(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


class Trader:
    def run(self, state: TradingState):
        books = {}
        for p in P.PRODUCTS:
            r = best_bid_ask(state.order_depths.get(p))
            if r is None:
                return {}, 0, ""
            books[p] = r

        sum_bb = sum(books[p][0] for p in P.PRODUCTS)
        sum_ba = sum(books[p][1] for p in P.PRODUCTS)

        orders: Dict[str, List[Order]] = {p: [] for p in P.PRODUCTS}

        if sum_bb > P.BASKET_SUM:
            qty = P.BASKET_QTY
            for p in P.PRODUCTS:
                pos = state.position.get(p, 0)
                qty = min(qty, books[p][2], P.POS_LIMIT + pos)
            if qty > 0:
                for p in P.PRODUCTS:
                    orders[p].append(Order(p, books[p][0], -qty))

        if sum_ba < P.BASKET_SUM:
            qty = P.BASKET_QTY
            for p in P.PRODUCTS:
                pos = state.position.get(p, 0)
                qty = min(qty, books[p][3], P.POS_LIMIT - pos)
            if qty > 0:
                for p in P.PRODUCTS:
                    orders[p].append(Order(p, books[p][1], qty))

        orders = {k: v for k, v in orders.items() if v}
        return orders, 0, ""
