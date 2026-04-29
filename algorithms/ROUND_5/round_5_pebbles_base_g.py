"""Round 5 PEBBLES — Baseline G: Baseline A + spread filter.

Only post passive quotes when spread >= MIN_SPREAD. PEBBLES is typically
12 ticks; tight spreads (<8) usually mean someone moved to the touch and
quoting inside is a losing trade. Skipping those keeps capture rate high.

Otherwise identical to A: pull-to-zero one-sided MM + ETF arb.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
BASKET_SUM = 50000
POS_LIMIT = 10
MM_QTY = 3
MIN_SPREAD = 6
BASKET_QTY = 10


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
            spread = ba - bb
            if spread < MIN_SPREAD:
                continue

            mm_bid = bb + 1
            mm_ask = ba - 1

            if pos <= 0:
                cap = min(MM_QTY, buy_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, mm_bid, cap))
                    buy_used[p] += cap
            if pos >= 0:
                cap = min(MM_QTY, sell_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, mm_ask, -cap))
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
