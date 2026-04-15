"""Generate synthetic price/trade CSV data for testing.

Creates a synthetic day with:
  - ASH_COATED_OSMIUM: mean-reverting around 10000, more volatile than real data
  - INTARIAN_PEPPER_ROOT: rises then reverses and falls

Usage:
    python generate_data.py
"""

import csv
import random
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ROUND = 1
DAY = 99  # synthetic day number
TICKS = 10000
DT = 100  # timestamp step

random.seed(42)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def generate_osmium():
    """Osmium: mean-reverts to 10000, wider swings than real data."""
    ANCHOR = 10000
    VOLATILITY = 3.0       # per-tick std (real is ~1.5)
    MEAN_REV = 0.02        # pull toward anchor
    NORMAL_SPREAD = 16
    NARROW_PROB = 0.10     # 10% narrow-spread ticks (real ~8%)
    EMPTY_PROB = 0.04      # 4% one-sided book

    mid = ANCHOR + random.gauss(0, 5)
    prices = []
    trades = []

    for i in range(TICKS):
        ts = i * DT

        # Random walk with mean reversion
        mid += random.gauss(0, VOLATILITY) - MEAN_REV * (mid - ANCHOR)
        mid = clamp(mid, ANCHOR - 40, ANCHOR + 40)

        is_narrow = random.random() < NARROW_PROB
        is_empty_bid = random.random() < EMPTY_PROB
        is_empty_ask = random.random() < EMPTY_PROB

        if is_narrow:
            spread = random.choice([5, 6, 7, 8, 9, 10, 11])
        else:
            spread = random.choice([16, 16, 16, 18, 19, 21])

        half = spread / 2
        bid1 = int(round(mid - half))
        ask1 = int(round(mid + half))
        bv1 = random.randint(3, 20)
        av1 = random.randint(3, 20)

        # Level 2 (present ~65% of time)
        has_l2 = random.random() < 0.65
        if has_l2:
            bid2 = bid1 - random.randint(1, 3)
            ask2 = ask1 + random.randint(1, 3)
            bv2 = random.randint(10, 30)
            av2 = random.randint(10, 30)
        else:
            bid2 = ask2 = bv2 = av2 = None

        # One-sided book
        if is_empty_bid:
            bid1 = bv1 = bid2 = bv2 = None
        if is_empty_ask:
            ask1 = av1 = ask2 = av2 = None

        if bid1 is not None and ask1 is not None:
            mp = (bid1 + ask1) / 2
        elif bid1 is not None:
            mp = bid1
        elif ask1 is not None:
            mp = ask1
        else:
            mp = 0

        prices.append({
            "day": DAY, "timestamp": ts, "product": "ASH_COATED_OSMIUM",
            "bid1": bid1, "bv1": bv1, "bid2": bid2, "bv2": bv2,
            "ask1": ask1, "av1": av1, "ask2": ask2, "av2": av2,
            "mid": mp,
        })

        # Market trade (~4% of ticks)
        if random.random() < 0.04 and (bid1 is not None or ask1 is not None):
            if bid1 is not None and (ask1 is None or random.random() < 0.5):
                t_price = bid1
            else:
                t_price = ask1
            trades.append({
                "timestamp": ts, "symbol": "ASH_COATED_OSMIUM",
                "price": t_price, "quantity": random.randint(2, 10),
            })

    return prices, trades


def generate_pepper():
    """Pepper: rises from 12000 to ~12080 then reverses and falls to ~11950."""
    START = 12000
    PEAK_TICK = 5000         # peaks around tick 5000 (halfway)
    RISE_RATE = 0.016        # per tick
    FALL_RATE = 0.025        # per tick (falls faster)
    VOLATILITY = 1.5
    NORMAL_SPREAD = 14
    NARROW_PROB = 0.03
    EMPTY_PROB = 0.04

    mid = START
    prices = []
    trades = []

    for i in range(TICKS):
        ts = i * DT

        if i < PEAK_TICK:
            drift = RISE_RATE
        else:
            drift = -FALL_RATE

        mid += drift + random.gauss(0, VOLATILITY)
        mid = clamp(mid, START - 100, START + 150)

        is_narrow = random.random() < NARROW_PROB
        is_empty_bid = random.random() < EMPTY_PROB
        is_empty_ask = random.random() < EMPTY_PROB

        if is_narrow:
            spread = random.choice([2, 3, 4, 5, 6])
        else:
            spread = random.choice([13, 13, 14, 14, 16, 17])

        half = spread / 2
        bid1 = int(round(mid - half))
        ask1 = int(round(mid + half))
        bv1 = random.randint(5, 20)
        av1 = random.randint(5, 20)

        has_l2 = random.random() < 0.50
        if has_l2:
            bid2 = bid1 - random.randint(1, 3)
            ask2 = ask1 + random.randint(1, 3)
            bv2 = random.randint(10, 25)
            av2 = random.randint(10, 25)
        else:
            bid2 = ask2 = bv2 = av2 = None

        if is_empty_bid:
            bid1 = bv1 = bid2 = bv2 = None
        if is_empty_ask:
            ask1 = av1 = ask2 = av2 = None

        if bid1 is not None and ask1 is not None:
            mp = (bid1 + ask1) / 2
        elif bid1 is not None:
            mp = bid1
        elif ask1 is not None:
            mp = ask1
        else:
            mp = 0

        prices.append({
            "day": DAY, "timestamp": ts, "product": "INTARIAN_PEPPER_ROOT",
            "bid1": bid1, "bv1": bv1, "bid2": bid2, "bv2": bv2,
            "ask1": ask1, "av1": av1, "ask2": ask2, "av2": av2,
            "mid": mp,
        })

        if random.random() < 0.035 and (bid1 is not None or ask1 is not None):
            if bid1 is not None and (ask1 is None or random.random() < 0.5):
                t_price = bid1
            else:
                t_price = ask1
            trades.append({
                "timestamp": ts, "symbol": "INTARIAN_PEPPER_ROOT",
                "price": t_price, "quantity": random.randint(2, 10),
            })

    return prices, trades


def write_csvs(osm_prices, osm_trades, pep_prices, pep_trades):
    out_dir = DATA_DIR / f"ROUND_{ROUND}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prices CSV
    prices_path = out_dir / f"prices_round_{ROUND}_day_{DAY}.csv"
    with open(prices_path, "w", newline="") as f:
        f.write("day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;"
                "bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;"
                "ask_price_3;ask_volume_3;mid_price;profit_and_loss\n")

        all_prices = osm_prices + pep_prices
        all_prices.sort(key=lambda r: (r["timestamp"], r["product"]))

        for r in all_prices:
            cols = [
                str(r["day"]), str(r["timestamp"]), r["product"],
                str(r["bid1"]) if r["bid1"] is not None else "",
                str(r["bv1"]) if r["bv1"] is not None else "",
                str(r["bid2"]) if r["bid2"] is not None else "",
                str(r["bv2"]) if r["bv2"] is not None else "",
                "", "",  # level 3
                str(r["ask1"]) if r["ask1"] is not None else "",
                str(r["av1"]) if r["av1"] is not None else "",
                str(r["ask2"]) if r["ask2"] is not None else "",
                str(r["av2"]) if r["av2"] is not None else "",
                "", "",  # level 3
                str(r["mid"]),
                "0.0",
            ]
            f.write(";".join(cols) + "\n")

    print(f"Wrote {prices_path} ({len(all_prices)} rows)")

    # Trades CSV
    trades_path = out_dir / f"trades_round_{ROUND}_day_{DAY}.csv"
    with open(trades_path, "w", newline="") as f:
        f.write("timestamp;buyer;seller;symbol;currency;price;quantity\n")

        all_trades = osm_trades + pep_trades
        all_trades.sort(key=lambda r: r["timestamp"])

        for t in all_trades:
            cols = [
                str(t["timestamp"]), "", "", t["symbol"],
                "XIRECS", str(float(t["price"])), str(t["quantity"]),
            ]
            f.write(";".join(cols) + "\n")

    print(f"Wrote {trades_path} ({len(all_trades)} trades)")


def main():
    print(f"Generating synthetic data: round {ROUND}, day {DAY}")
    print(f"  Osmium: mean-reverting around 10000, higher volatility")
    print(f"  Pepper: rises to ~12080, then falls to ~11950")
    print()

    osm_prices, osm_trades = generate_osmium()
    pep_prices, pep_trades = generate_pepper()

    write_csvs(osm_prices, osm_trades, pep_prices, pep_trades)

    # Print summary
    osm_mids = [r["mid"] for r in osm_prices if r["mid"] > 0]
    pep_mids = [r["mid"] for r in pep_prices if r["mid"] > 0]
    print(f"\nOsmium: {osm_mids[0]:.0f} -> {osm_mids[-1]:.0f} (range {min(osm_mids):.0f}-{max(osm_mids):.0f})")
    print(f"Pepper: {pep_mids[0]:.0f} -> {pep_mids[-1]:.0f} (range {min(pep_mids):.0f}-{max(pep_mids):.0f})")
    print(f"  Pepper peak at tick ~{max(range(len(pep_mids)), key=lambda i: pep_mids[i]) * DT}")
    print(f"Osmium trades: {len(osm_trades)}")
    print(f"Pepper trades: {len(pep_trades)}")


if __name__ == "__main__":
    main()