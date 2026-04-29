"""Round 5 PEBBLES — Baseline C: spread-aware MM + ETF arb.

Behavior:
  - Only post passive quotes when spread >= MIN_SPREAD (8 ticks).
  - Quote at best_bid+1 / best_ask-1 with size scaling toward zero-pos:
    * If pos == 0: BASE_QTY both sides
    * If pos > 0: 0 on bid, BASE_QTY on ask (force unload)
    * If pos < 0: BASE_QTY on bid, 0 on ask (force cover)
    * Linear interpolation between (no hard threshold).
  - ETF arb on top.

Differs from A by: scales quote size with inventory (continuous) rather than
binary one-sided gating. And adds spread filter so we don't quote when spread
is too tight to capture.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10
    MIN_SPREAD = 8
    BASE_QTY = 4
    BASKET_QTY = 10


def best_bid_ask(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


class Trader:
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {p: [] for p in P.PRODUCTS}
        books = {}
        for p in P.PRODUCTS:
            r = best_bid_ask(state.order_depths.get(p))
            if r is None:
                continue
            books[p] = r

        if len(books) != 5:
            return {k: v for k, v in orders.items() if v}, 0, ""

        buy_used = {p: 0 for p in P.PRODUCTS}
        sell_used = {p: 0 for p in P.PRODUCTS}

        def buy_cap(p):
            return P.POS_LIMIT - state.position.get(p, 0) - buy_used[p]

        def sell_cap(p):
            return P.POS_LIMIT + state.position.get(p, 0) - sell_used[p]

        for p in P.PRODUCTS:
            bb, ba, bv, av = books[p]
            pos = state.position.get(p, 0)
            spread = ba - bb
            if spread < P.MIN_SPREAD:
                continue

            # Linear inventory taper:
            # bid_size = BASE * max(0, 1 - pos/POS_LIMIT)
            # ask_size = BASE * max(0, 1 + pos/POS_LIMIT)
            bid_size = int(P.BASE_QTY * max(0.0, 1.0 - pos / P.POS_LIMIT))
            ask_size = int(P.BASE_QTY * max(0.0, 1.0 + pos / P.POS_LIMIT))

            if bid_size > 0:
                cap = min(bid_size, buy_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, bb + 1, cap))
                    buy_used[p] += cap
            if ask_size > 0:
                cap = min(ask_size, sell_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, ba - 1, -cap))
                    sell_used[p] += cap

        # ETF arb
        sum_bb = sum(books[p][0] for p in P.PRODUCTS)
        sum_ba = sum(books[p][1] for p in P.PRODUCTS)
        if sum_bb > P.BASKET_SUM:
            qty = P.BASKET_QTY
            for p in P.PRODUCTS:
                qty = min(qty, books[p][2], max(0, sell_cap(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    orders[p].append(Order(p, books[p][0], -qty))
                    sell_used[p] += qty
        if sum_ba < P.BASKET_SUM:
            qty = P.BASKET_QTY
            for p in P.PRODUCTS:
                qty = min(qty, books[p][3], max(0, buy_cap(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    orders[p].append(Order(p, books[p][1], qty))
                    buy_used[p] += qty

        orders = {k: v for k, v in orders.items() if v}
        return orders, 0, ""
