"""Compute theoretical max PnL via DP over the order book.

Usage:
    python theoretical_max.py --round 1
    python theoretical_max.py --round 1 --day 0
    python theoretical_max.py --round 1 --no-filter    # disable outlier filter
"""

import argparse
import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
POS_LIMIT = 80
ROLLING_WINDOW = 20
FILTER_TOLERANCE = 0  # pts of slack for the rolling-mid filter (auto-calibrated if 0)


def compute_max_pnl(prices_path: Path, product: str, filter_outliers: bool = True) -> dict:
    with open(prices_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = [r for r in reader if r["product"] == product]

    if not rows:
        return None

    # --- Pre-compute rolling mid and tolerance for outlier filtering ---
    mid_history = []
    rolling_mids = []
    for r in rows:
        mid = float(r["mid_price"]) if r["mid_price"] and float(r["mid_price"]) > 0 else None
        if mid is not None:
            mid_history.append(mid)
        rolling_mids.append(sum(mid_history[-ROLLING_WINDOW:]) / len(mid_history[-ROLLING_WINDOW:]) if mid_history else None)

    # Auto-calibrate tolerance: half the median spread
    tolerance = FILTER_TOLERANCE
    if filter_outliers and tolerance == 0:
        spreads = []
        for r in rows:
            if r["bid_price_1"] and r["ask_price_1"]:
                spreads.append(float(r["ask_price_1"]) - float(r["bid_price_1"]))
        if spreads:
            tolerance = sorted(spreads)[len(spreads) // 2] / 2

    # --- DP ---
    size = 2 * POS_LIMIT + 1
    INF = float("-inf")
    dp = [INF] * size
    dp[POS_LIMIT] = 0.0  # start at position 0

    filtered_count = 0

    for i, r in enumerate(rows):
        rm = rolling_mids[i]

        buys = []   # bids we can sell into
        sells = []  # asks we can buy from
        for lvl in range(1, 4):
            bp = r.get(f"bid_price_{lvl}", "")
            bv = r.get(f"bid_volume_{lvl}", "")
            if bp and bv:
                px = int(float(bp))
                # Filter: only sell into bids near or above rolling mid
                if filter_outliers and rm is not None and px < rm - tolerance:
                    filtered_count += 1
                    continue
                buys.append((px, int(float(bv))))
            ap = r.get(f"ask_price_{lvl}", "")
            av = r.get(f"ask_volume_{lvl}", "")
            if ap and av:
                px = int(float(ap))
                # Filter: only buy from asks near or below rolling mid
                if filter_outliers and rm is not None and px > rm + tolerance:
                    filtered_count += 1
                    continue
                sells.append((px, int(float(av))))

        if not buys and not sells:
            continue

        new_dp = dp[:]

        for p_idx in range(size):
            if dp[p_idx] == INF:
                continue
            pos = p_idx - POS_LIMIT
            base_pnl = dp[p_idx]

            # Buy from asks (ascending price)
            cum_cost = 0.0
            cum_qty = 0
            for ask_px, ask_vol in sorted(sells):
                can_buy = min(ask_vol, POS_LIMIT - pos - cum_qty)
                if can_buy <= 0:
                    break
                cum_qty += can_buy
                cum_cost += ask_px * can_buy
                new_idx = pos + cum_qty + POS_LIMIT
                val = base_pnl - cum_cost
                if val > new_dp[new_idx]:
                    new_dp[new_idx] = val

            # Sell into bids (descending price)
            cum_rev = 0.0
            cum_qty = 0
            for bid_px, bid_vol in sorted(buys, reverse=True):
                can_sell = min(bid_vol, POS_LIMIT + pos - cum_qty)
                if can_sell <= 0:
                    break
                cum_qty += can_sell
                cum_rev += bid_px * can_sell
                new_idx = pos - cum_qty + POS_LIMIT
                val = base_pnl + cum_rev
                if val > new_dp[new_idx]:
                    new_dp[new_idx] = val

        dp = new_dp

    # Mark to market at last valid mid
    last_mid = 0
    for r in reversed(rows):
        if r["mid_price"] and float(r["mid_price"]) > 0:
            last_mid = float(r["mid_price"])
            break

    best_pnl = INF
    best_pos = 0
    for p_idx in range(size):
        if dp[p_idx] == INF:
            continue
        pos = p_idx - POS_LIMIT
        total = dp[p_idx] + pos * last_mid
        if total > best_pnl:
            best_pnl = total
            best_pos = pos

    return {"pnl": best_pnl, "pos": best_pos, "last_mid": last_mid, "filtered": filtered_count}


def main():
    parser = argparse.ArgumentParser(description="Theoretical max PnL via DP")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--day", type=int, default=None)
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable outlier filtering (raw DP, allows future-peeking)")
    parser.add_argument("--tolerance", type=float, default=0,
                        help="Filter tolerance in pts (0 = auto: half median spread)")
    args = parser.parse_args()

    filter_on = not args.no_filter
    global FILTER_TOLERANCE
    if args.tolerance > 0:
        FILTER_TOLERANCE = args.tolerance

    round_dir = DATA_DIR / f"ROUND_{args.round}"
    if not round_dir.exists():
        print(f"Error: {round_dir} does not exist")
        return

    # Find available days
    if args.day is not None:
        days = [args.day]
    else:
        days = sorted(
            int(f.stem.split("_day_")[1])
            for f in round_dir.glob(f"prices_round_{args.round}_day_*.csv")
        )

    # Discover products from first file
    first_file = round_dir / f"prices_round_{args.round}_day_{days[0]}.csv"
    with open(first_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        products = sorted({r["product"] for r in reader})

    tol_str = f", tolerance={FILTER_TOLERANCE}" if FILTER_TOLERANCE > 0 else ", tolerance=auto (half median spread)"
    mode = f"filtered{tol_str}" if filter_on else "raw (no filter)"
    print(f"Round {args.round}, pos limit {POS_LIMIT}, mode: {mode}")
    print(f"Products: {products}")
    print()

    grand_total = {p: 0.0 for p in products}

    for day in days:
        path = round_dir / f"prices_round_{args.round}_day_{day}.csv"
        print(f"--- Day {day} ---")
        for product in products:
            result = compute_max_pnl(path, product, filter_outliers=filter_on)
            if result is None:
                print(f"  {product}: no data")
                continue
            filt_str = f", filtered {result['filtered']} orders" if result["filtered"] else ""
            print(f"  {product}: max PnL = {result['pnl']:.0f}  (final pos={result['pos']}, last mid={result['last_mid']}{filt_str})")
            grand_total[product] += result["pnl"]
        print()

    if len(days) > 1:
        print("--- Total across all days ---")
        for product in products:
            print(f"  {product}: {grand_total[product]:.0f}")
        print(f"  COMBINED: {sum(grand_total.values()):.0f}")


if __name__ == "__main__":
    main()
