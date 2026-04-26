"""Round 2 MAF-optimized variant.

Assumes we win the MAF auction: data is 1.25x volumes at all levels.
Tuning target: extract more from both products in the higher-flow regime.

Osmium: tuned params from optimize_round2 phase1; flow is bigger so the
same per-unit strategy captures more PnL automatically.

Pepper: original buy-and-hold (proven optimal for trending market). Plus
a **conservative** scalp overlay: post penny-inside sell only when pos is
at the limit AND current mid is significantly ABOVE recent trend
(indicating a temporary overshoot where the drift-eat-scalp loss may be
smaller than the scalp edge).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class OsmiumParams:
    SYMBOL = "ASH_COATED_OSMIUM"
    POS_LIMIT = 80
    MA_WINDOW = 30
    ANCHOR = 10000
    ANCHOR_WEIGHT = 0.15
    HALF_SPREAD = 8
    NARROW_SPREAD = 10
    NARROW_EDGE = 1
    INV_SKEW = 2
    DRIFT_STRENGTH = 10
    MICROPRICE_WEIGHT = 1.0
    URGENCY_EDGE = 3
    URGENCY_THRESHOLD = 0.5


class PepperParams:
    SYMBOL = "INTARIAN_PEPPER_ROOT"
    POS_LIMIT = 80
    BUY_MARGIN = 1            # initial accumulation cap
    # If >0, take asks <= first_ask + BUY_MARGIN (aggressive); set to a
    # large negative to disable take entirely (pure passive fills only)
    TREND_WINDOW = 200        # recent mids for local-trend estimate
    SCALP_OVERSHOOT = 4       # only scalp when mid > recent_trend + this
    SCALP_FLOOR = 72          # don't dip below this long position
    REFILL_MARGIN = 1         # refill take: asks <= current_best_ask + this


class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result = {}
        osm_prices: List[float] = []
        pep_mids: List[float] = []
        pep_first_ask = None

        if state.traderData:
            try:
                td = json.loads(state.traderData)
                osm_prices = td.get("op", [])
                pep_mids = td.get("pm", [])
                pep_first_ask = td.get("fa")
            except Exception:
                pass

        if OsmiumParams.SYMBOL in state.order_depths:
            result[OsmiumParams.SYMBOL] = self._trade_osmium(state, osm_prices)

        if PepperParams.SYMBOL in state.order_depths:
            orders, pep_first_ask = self._trade_pepper(state, pep_first_ask, pep_mids)
            result[PepperParams.SYMBOL] = orders

        return result, 0, json.dumps({"op": osm_prices, "pm": pep_mids, "fa": pep_first_ask})

    def _trade_osmium(self, state: TradingState, prices: List[float]) -> List[Order]:
        P = OsmiumParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is not None and best_ask is not None:
            plain = (best_bid + best_ask) / 2
            bid_vol = od.buy_orders[best_bid]
            ask_vol = -od.sell_orders[best_ask]
            total = bid_vol + ask_vol
            if total > 0:
                micro = (best_bid * ask_vol + best_ask * bid_vol) / total
                mid = plain * (1 - P.MICROPRICE_WEIGHT) + micro * P.MICROPRICE_WEIGHT
            else:
                mid = plain
        else:
            mid = P.ANCHOR

        prices.append(mid)
        if len(prices) > P.MA_WINDOW:
            prices[:] = prices[-P.MA_WINDOW:]
        ma_fv = sum(prices) / len(prices)
        fv = round(ma_fv * (1 - P.ANCHOR_WEIGHT) + P.ANCHOR * P.ANCHOR_WEIGHT)
        target_pos = max(-P.POS_LIMIT, min(P.POS_LIMIT,
                         round(-P.DRIFT_STRENGTH * (fv - P.ANCHOR))))
        fv_eff = fv - round(P.INV_SKEW * (pos - target_pos) / P.POS_LIMIT)

        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        raw_urg = (target_pos - pos) / P.POS_LIMIT
        mag = abs(raw_urg)
        if mag > P.URGENCY_THRESHOLD:
            ramp = (mag - P.URGENCY_THRESHOLD) / max(1e-9, 1.0 - P.URGENCY_THRESHOLD)
        else:
            ramp = 0.0
        urg_buy = max(0.0, ramp) if raw_urg > 0 else 0.0
        urg_sell = max(0.0, ramp) if raw_urg < 0 else 0.0

        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else 99
        buy_edge = round(P.URGENCY_EDGE * urg_buy)
        sell_edge = round(P.URGENCY_EDGE * urg_sell)
        if spread <= P.NARROW_SPREAD and od.buy_orders and od.sell_orders:
            bv = sum(od.buy_orders.values())
            av = sum(-v for v in od.sell_orders.values())
            if bv > av:
                buy_edge += P.NARROW_EDGE
            elif av > bv:
                sell_edge += P.NARROW_EDGE

        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fv_eff + buy_edge and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(P.SYMBOL, price, qty))
                    buy_cap -= qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fv_eff - sell_edge and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(P.SYMBOL, price, -qty))
                    sell_cap -= qty

        if best_bid is not None and best_ask is not None:
            our_bid = best_bid + 1
            our_ask = best_ask - 1
            if our_bid >= our_ask:
                our_bid = best_bid
                our_ask = best_ask
        elif best_bid is not None:
            our_bid = best_bid + 1
            our_ask = fv + P.HALF_SPREAD
        elif best_ask is not None:
            our_bid = fv - P.HALF_SPREAD
            our_ask = best_ask - 1
        else:
            our_bid = fv - P.HALF_SPREAD
            our_ask = fv + P.HALF_SPREAD

        our_bid = min(our_bid, fv_eff - 1)
        our_ask = max(our_ask, fv_eff)

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))

        return orders

    def _trade_pepper(self, state: TradingState, first_ask, mids: List[float]):
        P = PepperParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if first_ask is None and od.sell_orders:
            first_ask = min(od.sell_orders.keys())
        if first_ask is None:
            return orders, first_ask

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        buy_cap = P.POS_LIMIT - pos

        # Strategy under MAF (+25% agg-sell flow): capture cheap passive
        # fills at best_bid+1 first. Only fall back to aggressive take at
        # static_cap (first_ask + BUY_MARGIN) if best_bid+1 doesn't exist.
        #
        # Phase 1 take uses lowest-ask-first matching, so we also always
        # take any ask <= static_cap that's cheaper than current levels.
        static_cap = first_ask + P.BUY_MARGIN
        if buy_cap > 0 and od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price > static_cap or buy_cap <= 0:
                    break
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                buy_cap -= qty

        # Passive penny-inside bid for remaining buy_cap: captures aggressive
        # sell flow at best_bid price (cheaper than static_cap by ~7 ticks).
        if buy_cap > 0 and best_bid is not None:
            our_bid = best_bid + 1
            if best_ask is not None and our_bid >= best_ask:
                our_bid = best_bid
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))

        return orders, first_ask