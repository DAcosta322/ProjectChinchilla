"""Round 5 PEBBLES — Baseline A: pull-to-zero one-sided MM + ETF arb.

Behavior:
  - Always pull position toward 0:
    * pos > 0: post passive ASK only (no bids, can't add long)
    * pos < 0: post passive BID only (no asks, can't add short)
    * pos == 0: post both sides (neutral MM)
  - Quote price: best_bid+1 / best_ask-1, capped at small size
  - ETF basket arb when sum_bid > 50000 or sum_ask < 50000 (lock-in)
  - No anchor, no MR, no velocity, no trend.

Goal: small, ALWAYS positive spread capture on oscillating ticks; never adds
to a losing direction. Platform-safe by construction.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10
    MM_QTY = 10               # full POS_LIMIT — sweep showed strictly improves
                              # BT total ($26K → $35K) AND worst fragment
                              # ($340 → $366), still 30/30 positive. Pos cap
                              # is enforced by buy_cap/sell_cap regardless.
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
            for p in list(orders):
                if not orders[p]:
                    del orders[p]
            return orders, 0, ""

        buy_used = {p: 0 for p in P.PRODUCTS}
        sell_used = {p: 0 for p in P.PRODUCTS}

        def buy_cap(p):
            return P.POS_LIMIT - state.position.get(p, 0) - buy_used[p]

        def sell_cap(p):
            return P.POS_LIMIT + state.position.get(p, 0) - sell_used[p]

        # Per-product one-sided MM
        for p in P.PRODUCTS:
            bb, ba, bv, av = books[p]
            pos = state.position.get(p, 0)

            # Don't quote if spread is degenerate
            if ba <= bb:
                continue

            mm_bid = bb + 1 if bb + 1 < ba else None
            mm_ask = ba - 1 if ba - 1 > bb else None

            # Quote bid only when we can/want to BUY (not already long)
            if pos <= 0 and mm_bid is not None:
                cap = min(P.MM_QTY, buy_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, mm_bid, cap))
                    buy_used[p] += cap

            # Quote ask only when we can/want to SELL (not already short)
            if pos >= 0 and mm_ask is not None:
                cap = min(P.MM_QTY, sell_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, mm_ask, -cap))
                    sell_used[p] += cap

        # ETF arb: lock in deviations from BASKET_SUM
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
