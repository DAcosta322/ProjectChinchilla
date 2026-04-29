"""Round 5 PEBBLES — Baseline H: A + multi-level passive quotes (ladder).

After Baseline A v2 (+$1,276 platform), try posting at TWO price levels per
side: best_bid+1 (primary) AND best_bid+2 (deeper improvement). The +2 quote
fills rarely but when it does the spread capture is bigger.

Behavior identical to A otherwise: pull-to-zero, ETF arb.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10
    MM_QTY_L1 = 6           # primary quote at best_bid+1
    MM_QTY_L2 = 4           # deeper at best_bid+2 (sum capped by POS_LIMIT)
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
            if spread < 2:
                continue

            if pos <= 0:
                # Primary at bb+1
                cap = min(P.MM_QTY_L1, buy_cap(p))
                if cap > 0 and bb + 1 < ba:
                    orders[p].append(Order(p, bb + 1, cap))
                    buy_used[p] += cap
                # Deeper at bb+2 if spread permits
                if spread >= 4:
                    cap2 = min(P.MM_QTY_L2, buy_cap(p))
                    if cap2 > 0 and bb + 2 < ba:
                        orders[p].append(Order(p, bb + 2, cap2))
                        buy_used[p] += cap2

            if pos >= 0:
                cap = min(P.MM_QTY_L1, sell_cap(p))
                if cap > 0 and ba - 1 > bb:
                    orders[p].append(Order(p, ba - 1, -cap))
                    sell_used[p] += cap
                if spread >= 4:
                    cap2 = min(P.MM_QTY_L2, sell_cap(p))
                    if cap2 > 0 and ba - 2 > bb:
                        orders[p].append(Order(p, ba - 2, -cap2))
                        sell_used[p] += cap2

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
