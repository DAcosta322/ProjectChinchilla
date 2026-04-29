"""Round 5 PEBBLES — Baseline J: A + ETF arb edge threshold.

Currently ETF arb fires whenever sum_bid > 50000 or sum_ask < 50000 (zero
edge). Maybe firing only on bigger deviations reduces noise/churn.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10
    MM_QTY = 10
    BASKET_QTY = 10
    BASKET_EDGE = 5         # require sum to be >5 ticks past 50000 to fire


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
            if ba <= bb:
                continue

            if pos <= 0 and bb + 1 < ba:
                cap = min(P.MM_QTY, buy_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, bb + 1, cap))
                    buy_used[p] += cap
            if pos >= 0 and ba - 1 > bb:
                cap = min(P.MM_QTY, sell_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, ba - 1, -cap))
                    sell_used[p] += cap

        sum_bb = sum(books[p][0] for p in P.PRODUCTS)
        sum_ba = sum(books[p][1] for p in P.PRODUCTS)
        if sum_bb > P.BASKET_SUM + P.BASKET_EDGE:
            qty = P.BASKET_QTY
            for p in P.PRODUCTS:
                qty = min(qty, books[p][2], max(0, sell_cap(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    orders[p].append(Order(p, books[p][0], -qty))
                    sell_used[p] += qty
        if sum_ba < P.BASKET_SUM - P.BASKET_EDGE:
            qty = P.BASKET_QTY
            for p in P.PRODUCTS:
                qty = min(qty, books[p][3], max(0, buy_cap(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    orders[p].append(Order(p, books[p][1], qty))
                    buy_used[p] += qty

        orders = {k: v for k, v in orders.items() if v}
        return orders, 0, ""
