"""Round 5 — PEBBLES v9: high-threshold anchor commit + ETF arb.

v5/v8 issue: with DEV_THRESHOLD=30-50, the algo flipped target every
time dev crossed zero. Day 3 all 4 small products dipped above AND below
their first-mid anchor, causing whipsaws.

v9 raises the bar:
  - DEV_COMMIT = 500: don't commit until price has moved 500 ticks from
    anchor in one direction.
  - DEV_FLIP = 1500: once committed, only flip if price moves 1500 ticks
    in the OPPOSITE direction. Strong hysteresis.
  - This means we may miss small drifts, but we DON'T whipsaw.

Once committed, take full POS_LIMIT in that direction.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10

    DEV_COMMIT = 500          # commit when |drift from anchor| > this
    DEV_FLIP = 1500           # once committed, flip requires this much opposite drift

    AGG_BUDGET = 4
    PASSIVE_QTY = 5
    BASKET_ARB_QTY = 10


def best_bid_ask(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


class Trader:
    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        anchor: Dict[str, float] = data.get("anchor", {})
        commit: Dict[str, int] = data.get("commit", {})  # -1 short, +1 long, 0 neutral

        mids: Dict[str, float] = {}
        books: Dict[str, tuple] = {}
        for p in P.PRODUCTS:
            r = best_bid_ask(state.order_depths.get(p))
            if r is None:
                continue
            bb, ba, bv, av = r
            mids[p] = (bb + ba) / 2.0
            books[p] = (bb, ba, bv, av)

        if len(mids) != 5:
            return {}, 0, json.dumps({"anchor": anchor, "commit": commit})

        for p in P.PRODUCTS:
            anchor.setdefault(p, mids[p])
            commit.setdefault(p, 0)

        # Update commit state
        for p in P.PRODUCTS:
            dev = mids[p] - anchor[p]
            c = commit[p]
            if c == 0:
                if dev > P.DEV_COMMIT:
                    commit[p] = +1
                elif dev < -P.DEV_COMMIT:
                    commit[p] = -1
            elif c == +1 and dev < -P.DEV_FLIP:
                commit[p] = -1
            elif c == -1 and dev > P.DEV_FLIP:
                commit[p] = +1

        orders: Dict[str, List[Order]] = {p: [] for p in P.PRODUCTS}
        buy_used = {p: 0 for p in P.PRODUCTS}
        sell_used = {p: 0 for p in P.PRODUCTS}

        def buy_cap(p):
            return P.POS_LIMIT - state.position.get(p, 0) - buy_used[p]

        def sell_cap(p):
            return P.POS_LIMIT + state.position.get(p, 0) - sell_used[p]

        # ---- Per-product target from commit ----
        for p in P.PRODUCTS:
            bb, ba, bv, av = books[p]
            pos = state.position.get(p, 0)
            mid = mids[p]
            target = commit[p] * P.POS_LIMIT

            gap = target - pos

            if gap > 0.5 and ba <= mid + P.AGG_BUDGET:
                qty = min(int(gap + 0.5), av, buy_cap(p))
                if qty > 0:
                    orders[p].append(Order(p, ba, qty))
                    buy_used[p] += qty
            elif gap < -0.5 and bb >= mid - P.AGG_BUDGET:
                qty = min(int(-gap + 0.5), bv, sell_cap(p))
                if qty > 0:
                    orders[p].append(Order(p, bb, -qty))
                    sell_used[p] += qty

            recompute_gap = target - (pos + buy_used[p] - sell_used[p])
            if recompute_gap > 0.5:
                cap = min(P.PASSIVE_QTY, buy_cap(p))
                if cap > 0:
                    px = bb + 1 if bb + 1 < ba else bb
                    orders[p].append(Order(p, px, cap))
                    buy_used[p] += cap
            elif recompute_gap < -0.5:
                cap = min(P.PASSIVE_QTY, sell_cap(p))
                if cap > 0:
                    px = ba - 1 if ba - 1 > bb else ba
                    orders[p].append(Order(p, px, -cap))
                    sell_used[p] += cap

        # ---- ETF arb ----
        sum_bb = sum(books[p][0] for p in P.PRODUCTS)
        sum_ba = sum(books[p][1] for p in P.PRODUCTS)

        if sum_bb > P.BASKET_SUM:
            qty = P.BASKET_ARB_QTY
            for p in P.PRODUCTS:
                _, _, bv, _ = books[p]
                qty = min(qty, bv, max(0, sell_cap(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    bb = books[p][0]
                    orders[p].append(Order(p, bb, -qty))
                    sell_used[p] += qty

        if sum_ba < P.BASKET_SUM:
            qty = P.BASKET_ARB_QTY
            for p in P.PRODUCTS:
                _, _, _, av = books[p]
                qty = min(qty, av, max(0, buy_cap(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    ba = books[p][1]
                    orders[p].append(Order(p, ba, qty))
                    buy_used[p] += qty

        orders = {p: o for p, o in orders.items() if o}
        return orders, 0, json.dumps({"anchor": anchor, "commit": commit})
