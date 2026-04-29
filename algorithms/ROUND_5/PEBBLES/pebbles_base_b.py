"""Round 5 PEBBLES — Baseline B: anchor=mid + INV_SKEW MM + ETF arb.

Behavior:
  - Anchor = current mid (no EMA, no MR fight against drift).
  - Skew anchor by INV_SKEW * (pos / POS_LIMIT) — long pos pushes anchor down
    so ask becomes more attractive (sell side cheaper); short pos pushes up.
  - Quote at int(anchor - 1) / int(anchor + 1) with BASE_QTY size.
  - Cross spread aggressively only if best ask < anchor - AGG_EDGE (genuinely
    cheap relative to inventory-skewed anchor).
  - ETF basket arb on top.

Differs from A: still posts both sides always, but quotes shift with inventory.
The skew alone keeps inventory bounded without explicit one-side gating.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10
    INV_SKEW = 6.0          # at full pos, anchor shifts by 6 ticks
    AGG_EDGE = 5            # cross only if ba < anchor - AGG_EDGE (rare)
    BASE_QTY = 3
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
        mids = {}
        for p in P.PRODUCTS:
            r = best_bid_ask(state.order_depths.get(p))
            if r is None:
                continue
            books[p] = r
            mids[p] = (r[0] + r[1]) / 2.0

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
            mid = mids[p]

            # Inventory-skewed anchor
            anchor = mid - P.INV_SKEW * (pos / P.POS_LIMIT)

            # Aggressive cross only if very mispriced vs anchor
            if ba < anchor - P.AGG_EDGE:
                cap = min(av, buy_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, ba, cap))
                    buy_used[p] += cap
            if bb > anchor + P.AGG_EDGE:
                cap = min(bv, sell_cap(p))
                if cap > 0:
                    orders[p].append(Order(p, bb, -cap))
                    sell_used[p] += cap

            # Passive both-sided MM at int(anchor) ± 1, never inside book
            our_bid = min(int(anchor) - 1, ba - 1)
            our_ask = max(int(anchor) + 1, bb + 1)
            if our_bid >= ba:
                our_bid = ba - 1
            if our_ask <= bb:
                our_ask = bb + 1

            cap = min(P.BASE_QTY, buy_cap(p))
            if cap > 0 and our_bid > 0:
                orders[p].append(Order(p, our_bid, cap))
                buy_used[p] += cap
            cap = min(P.BASE_QTY, sell_cap(p))
            if cap > 0 and our_ask > 0:
                orders[p].append(Order(p, our_ask, -cap))
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
