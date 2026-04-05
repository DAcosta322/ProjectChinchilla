"""
Backtester for IMC Prosperity 4 trading algorithms.

Algorithms are loaded from the 'algorithms/' folder and data from 'data/'.

Usage:
    python backtester.py <algo_name> [--round ROUND] [--day DAY] [--print-output]

Example:
    python backtester.py tutorial --round 0 --day -1
    python backtester.py tutorial --round 0           # runs all days in round 0
"""

import argparse
import csv
import json
import sys
import os
import uuid
import importlib.util
import copy
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Ensure the imcProsperity directory is on sys.path so datamodel imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from datamodel import (
    Listing, OrderDepth, Trade, TradingState, Order, Observation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}

CURRENCY = "XIRECS"


# ---------------------------------------------------------------------------
# Data models for CSV rows
# ---------------------------------------------------------------------------
@dataclass
class PriceRow:
    day: int
    timestamp: int
    product: str
    bid_prices: List[int]
    bid_volumes: List[int]
    ask_prices: List[int]
    ask_volumes: List[int]
    mid_price: float


@dataclass
class MarketTrade:
    timestamp: int
    buyer: str
    seller: str
    symbol: str
    price: float
    quantity: int


# ---------------------------------------------------------------------------
# CSV data reader
# ---------------------------------------------------------------------------

class DataReader:
    """Reads price and trade CSVs for a given round/day."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def available_days(self, round_num: int) -> List[int]:
        round_dir = self.data_dir / f"round{round_num}"
        days = []
        if round_dir.is_dir():
            for f in round_dir.iterdir():
                if f.name.startswith("prices_round_") and f.suffix == ".csv":
                    # prices_round_0_day_-1.csv -> extract day number
                    parts = f.stem.split("_day_")
                    if len(parts) == 2:
                        days.append(int(parts[1]))
        return sorted(days)

    def read_prices(self, round_num: int, day_num: int) -> Dict[int, Dict[str, PriceRow]]:
        """Returns {timestamp: {product: PriceRow}}"""
        path = self.data_dir / f"round{round_num}" / f"prices_round_{round_num}_day_{day_num}.csv"
        result: Dict[int, Dict[str, PriceRow]] = {}

        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                ts = int(row["timestamp"])
                product = row["product"]

                bid_prices, bid_volumes = [], []
                ask_prices, ask_volumes = [], []
                for i in range(1, 4):
                    bp = row.get(f"bid_price_{i}", "")
                    bv = row.get(f"bid_volume_{i}", "")
                    if bp and bv:
                        bid_prices.append(int(float(bp)))
                        bid_volumes.append(int(float(bv)))
                    ap = row.get(f"ask_price_{i}", "")
                    av = row.get(f"ask_volume_{i}", "")
                    if ap and av:
                        ask_prices.append(int(float(ap)))
                        ask_volumes.append(int(float(av)))

                mid = float(row.get("mid_price", 0))

                pr = PriceRow(
                    day=day_num, timestamp=ts, product=product,
                    bid_prices=bid_prices, bid_volumes=bid_volumes,
                    ask_prices=ask_prices, ask_volumes=ask_volumes,
                    mid_price=mid,
                )
                result.setdefault(ts, {})[product] = pr

        return result

    def read_trades(self, round_num: int, day_num: int) -> Dict[int, Dict[str, List[MarketTrade]]]:
        """Returns {timestamp: {symbol: [MarketTrade]}}"""
        path = self.data_dir / f"round{round_num}" / f"trades_round_{round_num}_day_{day_num}.csv"
        result: Dict[int, Dict[str, List[MarketTrade]]] = {}

        if not path.exists():
            return result

        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                ts = int(row["timestamp"])
                mt = MarketTrade(
                    timestamp=ts,
                    buyer=row.get("buyer", ""),
                    seller=row.get("seller", ""),
                    symbol=row["symbol"],
                    price=float(row["price"]),
                    quantity=int(float(row["quantity"])),
                )
                result.setdefault(ts, {}).setdefault(mt.symbol, []).append(mt)

        return result


# ---------------------------------------------------------------------------
# Order matching engine
# ---------------------------------------------------------------------------
class OrderMatcher:
    """Matches trader orders against the order book, updates positions and PnL."""

    def __init__(self, position: Dict[str, int], pnl: Dict[str, float]):
        self.position = position
        self.pnl = pnl

    def match(
        self,
        orders: Dict[str, List[Order]],
        order_depths: Dict[str, OrderDepth],
        timestamp: int,
    ) -> List[dict]:
        """Match orders and return list of trade dicts."""
        trades = []
        for product, product_orders in orders.items():
            od = order_depths.get(product)
            if od is None:
                continue

            # Work on a copy so we can consume liquidity
            sell_book = dict(od.sell_orders)  # price -> negative qty
            buy_book = dict(od.buy_orders)    # price -> positive qty

            for order in product_orders:
                if order.quantity > 0:
                    # BUY order — match against sell book (ascending price)
                    remaining = order.quantity
                    for price in sorted(sell_book.keys()):
                        if price > order.price or remaining <= 0:
                            break
                        available = -sell_book[price]  # make positive
                        fill = min(remaining, available)
                        if fill <= 0:
                            continue

                        # Execute trade
                        self.position[product] = self.position.get(product, 0) + fill
                        self.pnl[product] = self.pnl.get(product, 0.0) - price * fill
                        remaining -= fill
                        sell_book[price] += fill  # less negative

                        trades.append({
                            "timestamp": timestamp,
                            "buyer": "SUBMISSION",
                            "seller": "",
                            "symbol": product,
                            "currency": CURRENCY,
                            "price": float(price),
                            "quantity": fill,
                        })

                elif order.quantity < 0:
                    # SELL order — match against buy book (descending price)
                    remaining = -order.quantity  # make positive
                    for price in sorted(buy_book.keys(), reverse=True):
                        if price < order.price or remaining <= 0:
                            break
                        available = buy_book[price]
                        fill = min(remaining, available)
                        if fill <= 0:
                            continue

                        self.position[product] = self.position.get(product, 0) - fill
                        self.pnl[product] = self.pnl.get(product, 0.0) + price * fill
                        remaining -= fill
                        buy_book[price] -= fill

                        trades.append({
                            "timestamp": timestamp,
                            "buyer": "",
                            "seller": "SUBMISSION",
                            "symbol": product,
                            "currency": CURRENCY,
                            "price": float(price),
                            "quantity": fill,
                        })

        return trades


# ---------------------------------------------------------------------------
# Position limit enforcer
# ---------------------------------------------------------------------------
def enforce_position_limits(
    orders: Dict[str, List[Order]],
    position: Dict[str, int],
    limits: Dict[str, int],
) -> Dict[str, List[Order]]:
    """
    If a product's aggregated buy (sell) volume would exceed the position
    limit if fully filled, ALL orders for that product are rejected.
    Returns the validated orders dict (products that violate are removed).
    """
    valid = {}
    for product, product_orders in orders.items():
        limit = limits.get(product, 0)
        pos = position.get(product, 0)

        total_buy = sum(o.quantity for o in product_orders if o.quantity > 0)
        total_sell = sum(-o.quantity for o in product_orders if o.quantity < 0)

        if pos + total_buy > limit or pos - total_sell < -limit:
            # Reject all orders for this product
            continue

        valid[product] = product_orders

    return valid


# ---------------------------------------------------------------------------
# Build TradingState from CSV data
# ---------------------------------------------------------------------------
def build_trading_state(
    timestamp: int,
    price_data: Dict[str, PriceRow],
    products: List[str],
    position: Dict[str, int],
    own_trades: Dict[str, List[Trade]],
    market_trades_at_ts: Dict[str, List[MarketTrade]],
    trader_data: str,
) -> TradingState:
    listings = {}
    order_depths = {}

    for product in products:
        listings[product] = Listing(
            symbol=product, product=product, denomination=CURRENCY
        )
        od = OrderDepth()
        pr = price_data.get(product)
        if pr:
            for p, v in zip(pr.bid_prices, pr.bid_volumes):
                od.buy_orders[p] = v
            for p, v in zip(pr.ask_prices, pr.ask_volumes):
                od.sell_orders[p] = -v  # sell volumes are negative
        order_depths[product] = od

    # Convert MarketTrade objects to Trade objects for market_trades
    mt_dict: Dict[str, List[Trade]] = {}
    for product, mt_list in market_trades_at_ts.items():
        mt_dict[product] = [
            Trade(
                symbol=mt.symbol, price=int(mt.price),
                quantity=mt.quantity, buyer=mt.buyer,
                seller=mt.seller, timestamp=mt.timestamp,
            )
            for mt in mt_list
        ]

    return TradingState(
        traderData=trader_data,
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=mt_dict,
        position=dict(position),
        observations=Observation({}, {}),
    )


# ---------------------------------------------------------------------------
# Activity log builder
# ---------------------------------------------------------------------------
def build_activity_row(
    day: int,
    timestamp: int,
    product: str,
    price_row: PriceRow,
    total_pnl: float,
) -> str:
    """Build a single activity log row matching the official format."""
    cols = [str(day), str(timestamp), product]

    # 3 bid levels
    for i in range(3):
        if i < len(price_row.bid_prices):
            cols.append(str(price_row.bid_prices[i]))
            cols.append(str(price_row.bid_volumes[i]))
        else:
            cols.append("")
            cols.append("")

    # 3 ask levels
    for i in range(3):
        if i < len(price_row.ask_prices):
            cols.append(str(price_row.ask_prices[i]))
            cols.append(str(price_row.ask_volumes[i]))
        else:
            cols.append("")
            cols.append("")

    cols.append(str(price_row.mid_price))
    cols.append(str(total_pnl))

    return ";".join(cols)


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------
def run_backtest(
    trader_module,
    data_reader: DataReader,
    round_num: int,
    day_num: int,
    print_output: bool = False,
) -> dict:
    """
    Runs a single day's backtest. Returns a result dict matching the
    official log format.
    """
    price_data = data_reader.read_prices(round_num, day_num)
    trade_data = data_reader.read_trades(round_num, day_num)
    timestamps = sorted(price_data.keys())

    if not timestamps:
        print(f"  No data for round {round_num} day {day_num}")
        return {}

    # Discover products from first timestamp
    products = sorted(price_data[timestamps[0]].keys())

    # State
    position: Dict[str, int] = {}
    pnl: Dict[str, float] = {p: 0.0 for p in products}
    trader_data = ""
    own_trades: Dict[str, List[Trade]] = {p: [] for p in products}

    matcher = OrderMatcher(position, pnl)
    trader = trader_module.Trader()

    # Output accumulators
    activity_header = (
        "day;timestamp;product;"
        "bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;"
        "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;"
        "mid_price;profit_and_loss"
    )
    activity_rows = [activity_header]
    sandbox_logs = []
    all_trades = []

    for ts in timestamps:
        prices_at_ts = price_data[ts]
        market_trades_at_ts = trade_data.get(ts, {})

        state = build_trading_state(
            timestamp=ts,
            price_data=prices_at_ts,
            products=products,
            position=position,
            own_trades=own_trades,
            market_trades_at_ts=market_trades_at_ts,
            trader_data=trader_data,
        )

        # Run the trader algorithm, capturing stdout
        stdout_capture = StringIO()
        try:
            if print_output:
                # Print to both console and capture
                result = trader.run(state)
                lambda_log = ""
            else:
                with redirect_stdout(stdout_capture):
                    result = trader.run(state)
                lambda_log = stdout_capture.getvalue().rstrip()
        except Exception as e:
            lambda_log = f"ERROR: {e}"
            result = ({}, 0, trader_data)

        # Unpack result
        if isinstance(result, tuple):
            if len(result) == 3:
                orders_dict, conversions, trader_data = result
            elif len(result) == 2:
                orders_dict, trader_data = result
            else:
                orders_dict = result[0] if result else {}
        else:
            orders_dict = result if isinstance(result, dict) else {}

        if orders_dict is None:
            orders_dict = {}

        # Enforce position limits
        sandbox_msg = ""
        valid_orders = enforce_position_limits(orders_dict, position, POSITION_LIMITS)
        for product in orders_dict:
            if product not in valid_orders:
                sandbox_msg += f"Orders for {product} exceeded position limit. "

        # Match orders
        new_trades = matcher.match(valid_orders, state.order_depths, ts)
        all_trades.extend(new_trades)

        # Build own_trades for next iteration
        own_trades = {p: [] for p in products}
        for t in new_trades:
            trade_obj = Trade(
                symbol=t["symbol"], price=int(t["price"]),
                quantity=t["quantity"], buyer=t["buyer"],
                seller=t["seller"], timestamp=ts,
            )
            own_trades.setdefault(t["symbol"], []).append(trade_obj)

        # Build activity log rows (one per product)
        for product in products:
            pr = prices_at_ts.get(product)
            if pr is None:
                continue
            # Total PnL = realized + unrealized (position * mid_price)
            realized = pnl.get(product, 0.0)
            pos = position.get(product, 0)
            total_pnl = realized + pos * pr.mid_price
            activity_rows.append(
                build_activity_row(day_num, ts, product, pr, total_pnl)
            )

        sandbox_logs.append({
            "sandboxLog": sandbox_msg,
            "lambdaLog": lambda_log,
            "timestamp": ts,
        })

    # Calculate final PnL
    last_ts = timestamps[-1]
    total_profit = 0.0
    for product in products:
        realized = pnl.get(product, 0.0)
        pos = position.get(product, 0)
        mid = price_data[last_ts][product].mid_price
        total_profit += realized + pos * mid

    # Build .log format (matches official Prosperity log)
    log_result = {
        "submissionId": str(uuid.uuid4()),
        "activitiesLog": "\n".join(activity_rows),
        "logs": sandbox_logs,
        "tradeHistory": all_trades,
    }

    # Build .json format (matches official Prosperity summary)
    json_result = {
        "round": str(round_num),
        "status": "FINISHED",
        "profit": total_profit,
        "activitiesLog": "\n".join(activity_rows),
        "positions": [
            {"symbol": CURRENCY, "quantity": int(-sum(
                t["price"] * t["quantity"] * (1 if t["buyer"] == "SUBMISSION" else -1)
                for t in all_trades
            ))},
        ] + [
            {"symbol": p, "quantity": position.get(p, 0)}
            for p in products
        ],
    }

    return {
        "log": log_result,
        "json": json_result,
        "profit": total_profit,
        "position": dict(position),
        "pnl_by_product": {
            p: pnl.get(p, 0.0) + position.get(p, 0) * price_data[last_ts][p].mid_price
            for p in products
        },
    }


# ---------------------------------------------------------------------------
# Algorithm loader
# ---------------------------------------------------------------------------
def load_algorithm(algo_path: Path):
    """Dynamically import a .py file containing a Trader class."""
    algo_path = algo_path.resolve()

    # Add both the algorithm's directory and the imcProsperity dir to path
    algo_dir = str(algo_path.parent)
    if algo_dir not in sys.path:
        sys.path.insert(0, algo_dir)

    spec = importlib.util.spec_from_file_location("trader_algo", algo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Trader"):
        print(f"ERROR: {algo_path} does not contain a 'Trader' class.")
        sys.exit(1)

    return module


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IMC Prosperity 4 Backtester")
    parser.add_argument("algorithm", type=str, help="Algorithm name (file in algorithms/ folder, without .py)")
    parser.add_argument("--round", type=int, default=0, help="Round number (default: 0)")
    parser.add_argument("--day", type=int, default=None, help="Day number (default: all days)")
    parser.add_argument("--print-output", action="store_true", help="Print trader stdout")

    args = parser.parse_args()

    # Resolve paths — algo from algorithms/, data from data/, logs to logs/
    algo_name = args.algorithm.removesuffix(".py")
    algo_path = SCRIPT_DIR / "algorithms" / f"{algo_name}.py"
    if not algo_path.exists():
        print(f"ERROR: Algorithm not found: {algo_path}")
        sys.exit(1)

    data_dir = SCRIPT_DIR / "data"
    out_dir = SCRIPT_DIR / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_reader = DataReader(data_dir)
    trader_module = load_algorithm(algo_path)

    # Determine which days to run
    if args.day is not None:
        days = [args.day]
    else:
        days = data_reader.available_days(args.round)
        if not days:
            print(f"No data found for round {args.round} in {data_dir}")
            sys.exit(1)

    print(f"=== IMC Prosperity 4 Backtester ===")
    print(f"Algorithm : {algo_path.name}")
    print(f"Round     : {args.round}")
    print(f"Days      : {days}")
    print(f"Data dir  : {data_dir}")
    print()

    cumulative_profit = 0.0

    for day in days:
        print(f"--- Running Round {args.round}, Day {day} ---")

        result = run_backtest(
            trader_module, data_reader, args.round, day,
            print_output=args.print_output,
        )

        if not result:
            continue

        profit = result["profit"]
        cumulative_profit += profit

        print(f"  Trades executed : {len(result['log']['tradeHistory'])}")
        print(f"  Final positions : {result['position']}")
        print(f"  PnL by product  : ", end="")
        for p, v in result["pnl_by_product"].items():
            print(f"{p}={v:.2f}  ", end="")
        print()
        print(f"  Day PnL         : {profit:.2f}")
        print()

        # Write output files
        run_id = str(uuid.uuid4())[:8]
        base_name = f"r{args.round}_d{day}_{algo_path.stem}_{run_id}"

        log_path = out_dir / f"{base_name}.log"
        with open(log_path, "w") as f:
            json.dump(result["log"], f)
        print(f"  Log written     : {log_path}")

        json_path = out_dir / f"{base_name}.json"
        with open(json_path, "w") as f:
            json.dump(result["json"], f)
        print(f"  Summary written : {json_path}")
        print()

    print(f"=== Total PnL across all days: {cumulative_profit:.2f} ===")


if __name__ == "__main__":
    main()
