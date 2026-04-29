"""Round 5 PEBBLES — Baseline I: A + per-product MM_QTY based on platform diagnostics.

Sub 551400 spread/RT was: XL +70, XS +25, L -12, M -19, S -34.
I scales MM size per product so heavier on the spread-positive ones:
  XL: 10 (positive edge)
  XS: 10 (positive edge)
  L:  6  (small loss)
  M:  4  (small loss)
  S:  3  (biggest loss)

Pull-to-zero one-sided, ETF arb same as A.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
BASKET_SUM = 50000
POS_LIMIT = 10
BASKET_QTY = 10
MM_QTY = {
    "PEBBLES_XS": 10,
    "PEBBLES_S":  3,
    "PEBBLES_M":  4,
    "PEBBLES_L":  6,
    "PEBBLES_XL": 10,
}


def best_bid_ask(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


class Trader:
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {p: [] for p in PRODUCTS}
        books = {}
        for p in PRODUCTS:
            r = best_bid_ask(state.order_depths.get(p))
            if r is None:
                continue
            books[p] = r

        if len(books) != 5:
            return {k: v for k, v in orders.items() if v}, 0, ""

        buy_used = {p: 0 for p in PRODUCTS}
        sell_used = {p: 0 for p in PRODUCTS}

        def buy_cap(p):
            return POS_LIMIT - state.position.get(p, 0) - buy_used[p]

        def sell_cap(p):
            return POS_LIMIT + state.position.get(p, 0) - sell_used[p]

        for p in PRODUCTS:
            bb, ba, bv, av = books[p]
            pos = state.position.get(p, 0)
            sz = MM_QTY[p]

            if ba <= bb:
                continue

            if pos <= 0 and bb + 1 < ba:
                cap = min(sz, buy_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, bb + 1, cap))
                    buy_used[p] += cap
            if pos >= 0 and ba - 1 > bb:
                cap = min(sz, sell_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, ba - 1, -cap))
                    sell_used[p] += cap

        # ETF arb
        sum_bb = sum(books[p][0] for p in PRODUCTS)
        sum_ba = sum(books[p][1] for p in PRODUCTS)
        if sum_bb > BASKET_SUM:
            qty = BASKET_QTY
            for p in PRODUCTS:
                qty = min(qty, books[p][2], max(0, sell_cap(p)))
            if qty > 0:
                for p in PRODUCTS:
                    orders[p].append(Order(p, books[p][0], -qty))
                    sell_used[p] += qty
        if sum_ba < BASKET_SUM:
            qty = BASKET_QTY
            for p in PRODUCTS:
                qty = min(qty, books[p][3], max(0, buy_cap(p)))
            if qty > 0:
                for p in PRODUCTS:
                    orders[p].append(Order(p, books[p][1], qty))
                    buy_used[p] += qty

        orders = {k: v for k, v in orders.items() if v}
        return orders, 0, ""
