"""Plot bid/ask order book and trade data for a single day.

Usage:
    python plot_data.py
    (prompts for round number and day number)
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"


def read_prices(path: Path):
    """Return {product: list of row dicts} from a prices CSV."""
    products = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            p = row["product"]
            products.setdefault(p, []).append(row)
    return products


def read_trades(path: Path):
    """Return {symbol: list of row dicts} from a trades CSV."""
    products = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            s = row["symbol"]
            products.setdefault(s, []).append(row)
    return products


def parse_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def plot_product(product, price_rows, trade_rows, out_dir: Path):
    """Create and save bid, ask, and trade figures for one product."""

    # --- Parse price data ---
    timestamps = [int(r["timestamp"]) for r in price_rows]

    bid_levels = []
    ask_levels = []
    for level in range(1, 4):
        bp = [parse_float(r[f"bid_price_{level}"]) for r in price_rows]
        bv = [parse_int(r[f"bid_volume_{level}"]) for r in price_rows]
        ap = [parse_float(r[f"ask_price_{level}"]) for r in price_rows]
        av = [parse_int(r[f"ask_volume_{level}"]) for r in price_rows]
        bid_levels.append((bp, bv))
        ask_levels.append((ap, av))

    mid = [parse_float(r["mid_price"]) for r in price_rows]

    # --- Compute price range from all data for consistent y-axis scaling ---
    all_prices = [p for p in mid if p is not None and p > 0]
    for bp, _ in bid_levels:
        all_prices.extend(p for p in bp if p is not None)
    for ap, _ in ask_levels:
        all_prices.extend(p for p in ap if p is not None)
    if trade_rows:
        all_prices.extend(float(r["price"]) for r in trade_rows)

    if all_prices:
        price_min = min(all_prices)
        price_max = max(all_prices)
        price_pad = max((price_max - price_min) * 0.05, 1)
        price_ylim = (price_min - price_pad, price_max + price_pad)
    else:
        price_ylim = None

    # --- Bid orders figure ---
    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{product} — Bid Orders", fontsize=14)

    colors = ["#2ca02c", "#98df8a", "#c7e9c0"]
    for i, (bp, bv) in enumerate(bid_levels):
        label = f"bid_{i+1}"
        valid = [(t, p, v) for t, p, v in zip(timestamps, bp, bv) if p is not None and v is not None]
        if valid:
            ts_v, bp_v, bv_v = zip(*valid)
            ax_price.plot(ts_v, bp_v, color=colors[i], linewidth=0.8, label=label, alpha=0.9)
            ax_vol.plot(ts_v, bv_v, color=colors[i], linewidth=0.8, label=label, alpha=0.9)

    mid_valid_ts = [t for t, m in zip(timestamps, mid) if m is not None and m > 0]
    mid_valid = [m for m in mid if m is not None and m > 0]
    if mid_valid:
        ax_price.plot(mid_valid_ts, mid_valid, color="gray", linewidth=0.6, linestyle="--", label="mid", alpha=0.7)

    ax_price.set_ylabel("Price")
    if price_ylim:
        ax_price.set_ylim(price_ylim)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)
    ax_vol.set_ylabel("Volume")
    ax_vol.set_xlabel("Timestamp")
    ax_vol.legend(loc="upper left", fontsize=8)
    ax_vol.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{product}_bids.png", dpi=150)
    plt.close(fig)

    # --- Ask orders figure ---
    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{product} — Ask Orders", fontsize=14)

    colors = ["#d62728", "#ff9896", "#fdd0a2"]
    for i, (ap, av) in enumerate(ask_levels):
        label = f"ask_{i+1}"
        valid = [(t, p, v) for t, p, v in zip(timestamps, ap, av) if p is not None and v is not None]
        if valid:
            ts_v, ap_v, av_v = zip(*valid)
            ax_price.plot(ts_v, ap_v, color=colors[i], linewidth=0.8, label=label, alpha=0.9)
            ax_vol.plot(ts_v, av_v, color=colors[i], linewidth=0.8, label=label, alpha=0.9)

    if mid_valid:
        ax_price.plot(mid_valid_ts, mid_valid, color="gray", linewidth=0.6, linestyle="--", label="mid", alpha=0.7)

    ax_price.set_ylabel("Price")
    if price_ylim:
        ax_price.set_ylim(price_ylim)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)
    ax_vol.set_ylabel("Volume")
    ax_vol.set_xlabel("Timestamp")
    ax_vol.legend(loc="upper left", fontsize=8)
    ax_vol.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{product}_asks.png", dpi=150)
    plt.close(fig)

    # --- Trades figure ---
    if not trade_rows:
        print(f"  {product}: no trades, skipping trade plot")
        return

    trade_ts = [int(r["timestamp"]) for r in trade_rows]
    trade_px = [float(r["price"]) for r in trade_rows]
    trade_qty = [int(r["quantity"]) for r in trade_rows]

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{product} — Trades", fontsize=14)

    ax_price.scatter(trade_ts, trade_px, c="#1f77b4", s=15, alpha=0.7, zorder=3)
    ax_price.plot(trade_ts, trade_px, color="#1f77b4", linewidth=0.5, alpha=0.4)
    if mid_valid:
        ax_price.plot(mid_valid_ts, mid_valid, color="gray", linewidth=0.6, linestyle="--", label="mid", alpha=0.5)
        ax_price.legend(loc="upper left", fontsize=8)
    ax_price.set_ylabel("Trade Price")
    if price_ylim:
        ax_price.set_ylim(price_ylim)
    ax_price.grid(True, alpha=0.3)

    ax_vol.bar(trade_ts, trade_qty, width=max(1, (max(trade_ts) - min(trade_ts)) / len(trade_ts) * 0.4),
               color="#1f77b4", alpha=0.7)
    ax_vol.set_ylabel("Trade Quantity")
    ax_vol.set_xlabel("Timestamp")
    ax_vol.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{product}_trades.png", dpi=150)
    plt.close(fig)


def main():
    round_num = input("Round number: ").strip()
    day_num = input("Day number: ").strip()

    round_dir = DATA_DIR / f"ROUND_{round_num}"
    if not round_dir.exists():
        print(f"Error: {round_dir} does not exist")
        return

    prices_file = round_dir / f"prices_round_{round_num}_day_{day_num}.csv"
    trades_file = round_dir / f"trades_round_{round_num}_day_{day_num}.csv"

    if not prices_file.exists():
        print(f"Error: {prices_file} does not exist")
        return

    print(f"Reading {prices_file.name} ...")
    price_data = read_prices(prices_file)

    trade_data = {}
    if trades_file.exists():
        print(f"Reading {trades_file.name} ...")
        trade_data = read_trades(trades_file)
    else:
        print(f"Warning: {trades_file.name} not found, skipping trade plots")

    products = sorted(price_data.keys())
    print(f"Products: {products}")

    out_dir = round_dir / f"day_{day_num}"
    out_dir.mkdir(exist_ok=True)

    for product in products:
        print(f"Plotting {product} ...")
        plot_product(product, price_data[product], trade_data.get(product, []), out_dir)
        print(f"  saved {product}_bids.png, {product}_asks.png, {product}_trades.png")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
