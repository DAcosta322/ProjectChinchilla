import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple


class Trader:

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Load persisted state
        trader_state = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except:
                trader_state = {}

        # === EMERALDS ===
        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state)

        # === TOMATOES ===
        if "TOMATOES" in state.order_depths:
            orders, trader_state = self.trade_tomatoes(state, trader_state)
            result["TOMATOES"] = orders

        traderData = json.dumps(trader_state)
        return result, conversions, traderData

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        """
        EMERALDS: stable fair value = 10000
        Bot walls: bids at 9990/9992, asks at 10008/10010

        Strategy (informed by Deep Research + Deep Think):
          1. Aggressively TAKE any mispriced resting liquidity (buy <= fair, sell >= fair)
          2. 0 EV inventory clearing: if position is heavy, flatten at fair value
             to free capacity for future positive-EV fills
          3. Single wide passive quote at 9993/10007 with full remaining capacity
        """
        FAIR = 10000
        LIMIT = 80
        orders: List[Order] = []

        order_depth = state.order_depths["EMERALDS"]
        position = state.position.get("EMERALDS", 0)

        buy_volume = 0
        sell_volume = 0

        # ------------------------------------------------------------------
        # STEP 1: Aggressively take ALL mispriced resting liquidity
        # Buy anything offered at or below fair, sell into anything bid at or above fair
        # ------------------------------------------------------------------

        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= FAIR:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    can_buy = LIMIT - position - buy_volume
                    take = min(ask_vol, can_buy)
                    if take > 0:
                        orders.append(Order("EMERALDS", ask_price, take))
                        buy_volume += take

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= FAIR:
                    bid_vol = order_depth.buy_orders[bid_price]
                    can_sell = LIMIT + position - sell_volume
                    take = min(bid_vol, can_sell)
                    if take > 0:
                        orders.append(Order("EMERALDS", bid_price, -take))
                        sell_volume += take

        # ------------------------------------------------------------------
        # STEP 2: 0 EV inventory clearing (Linear Utility technique)
        # If position is heavy, post aggressive orders at fair value to flatten.
        # These generate 0 profit but free capacity for positive-EV fills.
        # ------------------------------------------------------------------

        current_pos = position + buy_volume - sell_volume
        CLEAR_THRESHOLD = 40  # Start clearing when abs(position) > this

        if current_pos > CLEAR_THRESHOLD:
            # We're too long — post aggressive sell at fair value
            clear_qty = current_pos - CLEAR_THRESHOLD
            can_sell = LIMIT + position - sell_volume
            clear_qty = min(clear_qty, can_sell)
            if clear_qty > 0:
                orders.append(Order("EMERALDS", FAIR, -clear_qty))
                sell_volume += clear_qty

        elif current_pos < -CLEAR_THRESHOLD:
            # We're too short — post aggressive buy at fair value
            clear_qty = -current_pos - CLEAR_THRESHOLD
            can_buy = LIMIT - position - buy_volume
            clear_qty = min(clear_qty, can_buy)
            if clear_qty > 0:
                orders.append(Order("EMERALDS", FAIR, clear_qty))
                buy_volume += clear_qty

        # ------------------------------------------------------------------
        # STEP 3: Single wide passive quote — maximize edge per fill
        # Full remaining capacity at one level per side
        # ------------------------------------------------------------------

        remaining_buy = LIMIT - position - buy_volume
        remaining_sell = LIMIT + position - sell_volume

        if remaining_buy > 0:
            orders.append(Order("EMERALDS", 9993, remaining_buy))

        if remaining_sell > 0:
            orders.append(Order("EMERALDS", 10007, -remaining_sell))

        return orders

    def get_vwap_fair_value(self, order_depth: OrderDepth) -> float:
        """
        VWAP across all book levels — reverse-engineered as the engine's
        actual PnL marking price (MAE=0.23, lowest of all formulas tested).
        """
        total_bid_val = 0.0
        total_bid_vol = 0
        for price, vol in order_depth.buy_orders.items():
            total_bid_val += price * vol
            total_bid_vol += vol

        total_ask_val = 0.0
        total_ask_vol = 0
        for price, vol in order_depth.sell_orders.items():
            total_ask_val += price * (-vol)  # sell volumes are negative
            total_ask_vol += (-vol)

        vwap_bid = total_bid_val / total_bid_vol if total_bid_vol > 0 else 0
        vwap_ask = total_ask_val / total_ask_vol if total_ask_vol > 0 else 0
        return (vwap_bid + vwap_ask) / 2.0

    def get_wall_fair_value(self, order_depth: OrderDepth) -> float:
        """Thick-wall average (kept for reference)."""
        thick_bid = max(order_depth.buy_orders.keys(),
                        key=lambda p: order_depth.buy_orders[p])
        thick_ask = min(order_depth.sell_orders.keys(),
                        key=lambda p: order_depth.sell_orders[p])
        return (thick_bid + thick_ask) / 2.0

    def get_microprice(self, order_depth: OrderDepth) -> float:
        """Volume-weighted microprice from best bid/ask."""
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = -order_depth.sell_orders[best_ask]
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def trade_tomatoes(self, state: TradingState, trader_state: dict) -> Tuple[List[Order], dict]:
        """
        TOMATOES: CJP asymmetric quoting framework

        Strategy v6:
          - Microprice FV + mean-reversion adjustment
          - Composite L1+L2 imbalance signal (w=0.8 on L2)
          - Nonlinear signal transform f(I) = sgn(I)*|I|^2
          - ASYMMETRIC quote offsets: tighten on predicted side, widen on vulnerable
          - Inventory penalty interacts with signal (compound/cancel)
        """
        LIMIT = 80
        RHO = -0.42
        TAKE_EDGE = 2
        DELTA = 5              # live-validated optimal
        KAPPA = 0.3            # live-validated optimal
        ETA = 1.2              # live-validated optimal
        GAMMA = 2.0 / LIMIT   # inventory penalty
        CLEAR_THRESHOLD = 65
        L2_WEIGHT = 0.8        # composite: 80% L2, 20% L1
        SIGNAL_POWER = 2       # nonlinear dampening exponent
        orders: List[Order] = []

        order_depth = state.order_depths["TOMATOES"]
        position = state.position.get("TOMATOES", 0)

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        # ------------------------------------------------------------------
        # STEP 1: Fair value = VWAP all levels + mean-reversion
        # Reverse-engineered from PnL marking: VWAP has lowest MAE (0.23)
        # ------------------------------------------------------------------

        vwap_fv = self.get_vwap_fair_value(order_depth)
        prev_vwap_fv = trader_state.get("prev_vwap_fv", vwap_fv)
        mr_adjustment = RHO * (vwap_fv - prev_vwap_fv)
        fair_value = vwap_fv + mr_adjustment
        trader_state["prev_vwap_fv"] = vwap_fv

        # ------------------------------------------------------------------
        # STEP 2: Composite imbalance + L3 directional signal
        # ------------------------------------------------------------------

        sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(order_depth.sell_orders.keys())

        # L1 imbalance
        bid_vol_1 = order_depth.buy_orders[sorted_bids[0]]
        ask_vol_1 = -order_depth.sell_orders[sorted_asks[0]]
        imb_l1 = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1) if (bid_vol_1 + ask_vol_1) > 0 else 0.0

        # L2 imbalance
        imb_l2 = 0.0
        if len(sorted_bids) >= 2 and len(sorted_asks) >= 2:
            bid_vol_2 = order_depth.buy_orders[sorted_bids[1]]
            ask_vol_2 = -order_depth.sell_orders[sorted_asks[1]]
            if bid_vol_2 + ask_vol_2 > 0:
                imb_l2 = (bid_vol_2 - ask_vol_2) / (bid_vol_2 + ask_vol_2)

        # Composite: heavy L2 for structure, L1 for momentum
        imbalance = L2_WEIGHT * imb_l2 + (1 - L2_WEIGHT) * imb_l1

        # Nonlinear transform: dampen noise, amplify strong signals
        sign_imb = 1.0 if imbalance >= 0 else -1.0
        f_imb = sign_imb * (abs(imbalance) ** SIGNAL_POWER)

        # ------------------------------------------------------------------
        # STEP 3: Compute FV for taking decisions
        # Combines: imbalance signal + inventory penalty
        # ------------------------------------------------------------------

        signal_shift = ETA * f_imb
        inv_shift = -GAMMA * position

        adjusted_fv = fair_value + signal_shift + inv_shift
        fv_int = round(adjusted_fv)

        # ------------------------------------------------------------------
        # STEP 4: Selective taking at adjusted FV ± TAKE_EDGE
        # ------------------------------------------------------------------

        buy_volume = 0
        sell_volume = 0

        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= fv_int - TAKE_EDGE:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    can_buy = LIMIT - position - buy_volume
                    take = min(ask_vol, can_buy)
                    if take > 0:
                        orders.append(Order("TOMATOES", ask_price, take))
                        buy_volume += take

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= fv_int + TAKE_EDGE:
                    bid_vol = order_depth.buy_orders[bid_price]
                    can_sell = LIMIT + position - sell_volume
                    take = min(bid_vol, can_sell)
                    if take > 0:
                        orders.append(Order("TOMATOES", bid_price, -take))
                        sell_volume += take

        # ------------------------------------------------------------------
        # STEP 5: 0 EV clearing near position limit
        # ------------------------------------------------------------------

        current_pos = position + buy_volume - sell_volume

        if current_pos > CLEAR_THRESHOLD:
            clear_qty = current_pos - CLEAR_THRESHOLD
            can_sell = LIMIT + position - sell_volume
            clear_qty = min(clear_qty, can_sell)
            if clear_qty > 0:
                orders.append(Order("TOMATOES", fv_int, -clear_qty))
                sell_volume += clear_qty
        elif current_pos < -CLEAR_THRESHOLD:
            clear_qty = -current_pos - CLEAR_THRESHOLD
            can_buy = LIMIT - position - buy_volume
            clear_qty = min(clear_qty, can_buy)
            if clear_qty > 0:
                orders.append(Order("TOMATOES", fv_int, clear_qty))
                buy_volume += clear_qty

        # ------------------------------------------------------------------
        # STEP 6: ASYMMETRIC passive quotes (CJP framework)
        # Tighten on side of predicted move, widen on vulnerable side
        # ------------------------------------------------------------------

        # Bid offset: tighten when bullish (I > 0), widen when bearish
        bid_squeeze = KAPPA * max(0.0, imbalance)   # tightens bid when bullish
        ask_squeeze = KAPPA * max(0.0, -imbalance)   # tightens ask when bearish

        bid_offset = DELTA * (1.0 - bid_squeeze)
        ask_offset = DELTA * (1.0 + ask_squeeze) if imbalance >= 0 else DELTA
        if imbalance < 0:
            bid_offset = DELTA * (1.0 + (-imbalance) * KAPPA)  # widen bid when bearish
            ask_offset = DELTA * (1.0 - ask_squeeze)            # tighten ask when bearish

        our_bid = round(adjusted_fv - bid_offset)
        our_ask = round(adjusted_fv + ask_offset)

        # Safety: don't cross
        if our_bid >= our_ask:
            our_bid = round(adjusted_fv) - 1
            our_ask = round(adjusted_fv) + 1

        remaining_buy = LIMIT - position - buy_volume
        remaining_sell = LIMIT + position - sell_volume

        if remaining_buy > 0:
            orders.append(Order("TOMATOES", our_bid, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order("TOMATOES", our_ask, -remaining_sell))

        return orders, trader_state
