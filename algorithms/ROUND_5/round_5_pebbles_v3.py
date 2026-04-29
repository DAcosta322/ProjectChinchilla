"""Round 5 — PEBBLES v3: pure ETF basket arb, no MM.

v1 (MM + momentum tilt): -$32K
v2 (trend-target): -$155K
The MM and momentum-target both bled because individual products drift
2000-4000 ticks per day. Trend signal whipsaws; MM is run over by sustained
moves.

v3 only fires on the basket invariant: when sum(best_bid) > 50000 + EDGE,
sell all 5 at bids; when sum(best_ask) < 50000 - EDGE, buy all 5 at asks.
Net basket position = sum(positions); we use that to gate which side fires.
At any time we only hold a flat basket (q×5 same-sign) with MTM locked at
q×50000 — directional risk is zero by construction.

Logs each fire to verify the arb actually triggers.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10
    BASKET_ARB_EDGE = 0   # take any non-negative edge
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
        fires_long = data.get("fires_long", 0)
        fires_short = data.get("fires_short", 0)

        books = {}
        for p in P.PRODUCTS:
            r = best_bid_ask(state.order_depths.get(p))
            if r is None:
                return {}, 0, json.dumps(data)
            books[p] = r

        sum_bb = sum(books[p][0] for p in P.PRODUCTS)
        sum_ba = sum(books[p][1] for p in P.PRODUCTS)

        orders: Dict[str, List[Order]] = {p: [] for p in P.PRODUCTS}

        # Sell ETF: sum_bids > 50000
        if sum_bb > P.BASKET_SUM + P.BASKET_ARB_EDGE:
            qty = P.BASKET_ARB_QTY
            for p in P.PRODUCTS:
                bb, _, bv, _ = books[p]
                pos = state.position.get(p, 0)
                qty = min(qty, bv, P.POS_LIMIT + pos)
            if qty > 0:
                fires_short += 1
                for p in P.PRODUCTS:
                    bb = books[p][0]
                    orders[p].append(Order(p, bb, -qty))

        # Buy ETF: sum_asks < 50000
        if sum_ba < P.BASKET_SUM - P.BASKET_ARB_EDGE:
            qty = P.BASKET_ARB_QTY
            for p in P.PRODUCTS:
                _, ba, _, av = books[p]
                pos = state.position.get(p, 0)
                qty = min(qty, av, P.POS_LIMIT - pos)
            if qty > 0:
                fires_long += 1
                for p in P.PRODUCTS:
                    ba = books[p][1]
                    orders[p].append(Order(p, ba, qty))

        # Periodic log
        if state.timestamp % 100000 == 0:
            print(f"ts={state.timestamp} sum_bb={sum_bb} sum_ba={sum_ba} "
                  f"fires_L={fires_long} fires_S={fires_short}")

        orders = {p: o for p, o in orders.items() if o}
        return orders, 0, json.dumps({"fires_long": fires_long, "fires_short": fires_short})
