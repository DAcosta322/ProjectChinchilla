"""
Backtest result analyzer for IMC Prosperity 4.

Reads log/json files produced by backtester.py and generates matplotlib charts:
  - PnL over time per product
  - Market bid/ask prices with mid-price
  - Algorithm order prices colored by fill status

Usage:
    python analyze.py <log_folder>
    python analyze.py logs/tutorial_20260405_143000

The script looks for all .log and .json files in the given folder.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_activities_log(activities_csv: str):
    """Parse the semicolon-delimited activitiesLog into per-product time series."""
    reader = csv.DictReader(StringIO(activities_csv), delimiter=";")

    # {product: {"timestamps": [], "mid_price": [], "pnl": [], "bid1": [], "ask1": [], ...}}
    series = defaultdict(lambda: defaultdict(list))

    for row in reader:
        product = row["product"]
        ts = int(row["timestamp"])
        s = series[product]
        s["timestamps"].append(ts)
        s["mid_price"].append(float(row["mid_price"]) if row["mid_price"] else None)
        s["profit_and_loss"].append(float(row["profit_and_loss"]) if row["profit_and_loss"] else 0.0)

        # Best bid/ask for market price visualization
        bid1 = row.get("bid_price_1", "")
        ask1 = row.get("ask_price_1", "")
        s["best_bid"].append(float(bid1) if bid1 else None)
        s["best_ask"].append(float(ask1) if ask1 else None)

    return dict(series)


def group_orders_by_product(orders):
    """Group order history by product symbol."""
    by_product = defaultdict(list)
    for o in orders:
        by_product[o["symbol"]].append(o)
    return dict(by_product)


def group_trades_by_product(trades):
    """Group trade history by product symbol."""
    by_product = defaultdict(list)
    for t in trades:
        by_product[t["symbol"]].append(t)
    return dict(by_product)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STATUS_COLORS = {
    "filled": "#2bc86d",      # green
    "partial": "#f39c12",     # orange
    "unfilled": "#e74c3c",    # red
    "rejected": "#95a5a6",    # grey
}

STATUS_MARKERS = {
    "filled": "o",
    "partial": "s",
    "unfilled": "x",
    "rejected": "D",
}


def plot_pnl(series_by_product, ax, day_label):
    """Plot PnL over time for each product and total."""
    total_pnl = None

    for product, s in sorted(series_by_product.items()):
        ts = np.array(s["timestamps"])
        pnl = np.array(s["profit_and_loss"])
        ax.plot(ts, pnl, label=product, linewidth=1.2)

        if total_pnl is None:
            total_pnl = np.zeros_like(pnl, dtype=float)
        # Accumulate total (timestamps should align across products)
        if len(pnl) == len(total_pnl):
            total_pnl += pnl

    if total_pnl is not None and len(series_by_product) > 1:
        ts = np.array(list(series_by_product.values())[0]["timestamps"])
        ax.plot(ts, total_pnl, label="TOTAL", linewidth=2, linestyle="--", color="black")

    ax.set_title(f"Profit & Loss — {day_label}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("PnL")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_market_and_orders(product, series, orders, trades, ax, day_label):
    """Plot market bid/ask, mid-price, and algorithm orders for one product."""
    ts = np.array(series["timestamps"])
    mid = np.array(series["mid_price"], dtype=float)
    best_bid = np.array([b if b is not None else np.nan for b in series["best_bid"]])
    best_ask = np.array([a if a is not None else np.nan for a in series["best_ask"]])

    # Market prices
    ax.fill_between(ts, best_bid, best_ask, alpha=0.15, color="steelblue", label="Bid-Ask Spread")
    ax.plot(ts, mid, color="steelblue", linewidth=1.0, label="Mid Price", alpha=0.8)
    ax.plot(ts, best_bid, color="green", linewidth=0.5, alpha=0.4, label="Best Bid")
    ax.plot(ts, best_ask, color="red", linewidth=0.5, alpha=0.4, label="Best Ask")

    # Plot order styling — prominent for filled/partial, subtle for unfilled
    order_styles = {
        "filled":   {"size": 30, "alpha": 0.9, "zorder": 5},
        "partial":  {"size": 30, "alpha": 0.9, "zorder": 5},
        "unfilled": {"size": 4,  "alpha": 0.15, "zorder": 2},
        "rejected": {"size": 8,  "alpha": 0.3, "zorder": 3},
    }

    for status in ["unfilled", "rejected", "partial", "filled"]:  # draw important on top
        color = STATUS_COLORS[status]
        style = order_styles[status]
        buy_orders = [o for o in orders if o["status"] == status and o["side"] == "BUY"]
        sell_orders = [o for o in orders if o["status"] == status and o["side"] == "SELL"]

        if buy_orders:
            ax.scatter(
                [o["timestamp"] for o in buy_orders],
                [o["price"] for o in buy_orders],
                c=color, marker="^", s=style["size"], alpha=style["alpha"],
                edgecolors="none", zorder=style["zorder"],
                label=f"Buy {status}" if buy_orders else None,
            )
        if sell_orders:
            ax.scatter(
                [o["timestamp"] for o in sell_orders],
                [o["price"] for o in sell_orders],
                c=color, marker="v", s=style["size"], alpha=style["alpha"],
                edgecolors="none", zorder=style["zorder"],
                label=f"Sell {status}" if sell_orders else None,
            )

    # Auto-scale y-axis to market price range with padding, not order extremes
    valid_bids = best_bid[~np.isnan(best_bid)]
    valid_asks = best_ask[~np.isnan(best_ask)]
    if len(valid_bids) > 0 and len(valid_asks) > 0:
        price_min = np.min(valid_bids)
        price_max = np.max(valid_asks)
        padding = (price_max - price_min) * 0.15
        ax.set_ylim(price_min - padding, price_max + padding)

    ax.set_title(f"{product} — Market & Orders — {day_label}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_run(log_dir: Path):
    """Load all day results from a run folder and produce charts."""
    log_files = sorted(log_dir.glob("*.log"))
    json_files = sorted(log_dir.glob("*.json"))

    if not log_files:
        print(f"No .log files found in {log_dir}")
        sys.exit(1)

    print(f"=== Backtest Analyzer ===")
    print(f"Run folder : {log_dir}")
    print(f"Log files  : {[f.name for f in log_files]}")
    print()

    all_days = []

    for log_path, json_path in zip(log_files, json_files):
        with open(log_path) as f:
            log_data = json.load(f)
        with open(json_path) as f:
            json_data = json.load(f)

        # Extract day label from filename (e.g., r0_d-1_tutorial_xxx)
        stem = log_path.stem
        parts = stem.split("_")
        day_label = f"Round {parts[0][1:]}, Day {parts[1][1:]}" if len(parts) >= 2 else stem

        all_days.append({
            "day_label": day_label,
            "log": log_data,
            "json": json_data,
            "profit": json_data.get("profit", 0),
        })

    # Determine number of products from first day
    first_series = parse_activities_log(all_days[0]["log"]["activitiesLog"])
    products = sorted(first_series.keys())
    n_days = len(all_days)

    # Create figure: for each day -> 1 PnL plot + 1 market/orders plot per product
    n_rows_per_day = 1 + len(products)
    total_rows = n_rows_per_day * n_days
    fig, axes = plt.subplots(total_rows, 1, figsize=(14, 4.5 * total_rows))
    if total_rows == 1:
        axes = [axes]

    row = 0
    for day_info in all_days:
        log_data = day_info["log"]
        day_label = day_info["day_label"]

        series = parse_activities_log(log_data["activitiesLog"])
        orders_by_product = group_orders_by_product(log_data.get("orderHistory", []))
        trades_by_product = group_trades_by_product(log_data.get("tradeHistory", []))

        # PnL chart
        plot_pnl(series, axes[row], day_label)
        row += 1

        # Market + orders chart per product
        for product in products:
            product_series = series.get(product)
            if product_series is None:
                row += 1
                continue

            product_orders = orders_by_product.get(product, [])
            product_trades = trades_by_product.get(product, [])

            plot_market_and_orders(
                product, product_series, product_orders, product_trades,
                axes[row], day_label,
            )
            row += 1

    fig.tight_layout(pad=2.0)

    # Save to the same run folder
    out_path = log_dir / "analysis.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"Charts saved to: {out_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="IMC Prosperity 4 Backtest Analyzer")
    parser.add_argument("log_folder", type=str, help="Path to the backtester output folder (e.g., logs/tutorial_20260405_143000)")

    args = parser.parse_args()
    log_dir = Path(args.log_folder)

    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parent / log_dir

    if not log_dir.is_dir():
        print(f"ERROR: Not a directory: {log_dir}")
        sys.exit(1)

    analyze_run(log_dir)


if __name__ == "__main__":
    main()
