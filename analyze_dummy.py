"""
Analyze dummy trader logs for ASH_COATED_OSMIUM from ROUND_1_DUMMY.
Dummy trader placed NO orders, so all market activity is pure bot behavior.
"""

import json
import csv
import io
from collections import Counter, defaultdict
import math

LOG_PATH = "d:/UW/1B/prosperity_imc/dump/ROUND_1_DUMMY/228836.log"
PRODUCT = "ASH_COATED_OSMIUM"

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def parse_activities(data):
    """Parse activitiesLog CSV (semicolon-separated) and filter for PRODUCT."""
    reader = csv.DictReader(io.StringIO(data["activitiesLog"]), delimiter=";")
    rows = []
    for row in reader:
        if row["product"] != PRODUCT:
            continue
        r = {
            "day": int(row["day"]),
            "timestamp": int(row["timestamp"]),
            "mid_price": float(row["mid_price"]) if row["mid_price"] else None,
            "pnl": float(row["profit_and_loss"]) if row["profit_and_loss"] else 0.0,
        }
        for side in ["bid", "ask"]:
            for level in [1, 2, 3]:
                pk = f"{side}_price_{level}"
                vk = f"{side}_volume_{level}"
                p = row.get(pk, "")
                v = row.get(vk, "")
                r[pk] = float(p) if p else None
                r[vk] = int(v) if v else None
        rows.append(r)
    return rows

def parse_trades(data):
    """Extract trades for PRODUCT from tradeHistory."""
    trades = []
    for t in data.get("tradeHistory", []):
        if t.get("symbol") == PRODUCT:
            trades.append({
                "timestamp": t["timestamp"],
                "price": t["price"],
                "quantity": t["quantity"],
                "buyer": t.get("buyer", ""),
                "seller": t.get("seller", ""),
            })
    return trades

def compute_stats(rows, trades, label="ALL"):
    print(f"\n{'='*70}")
    print(f"  {PRODUCT} — {label}")
    print(f"{'='*70}")

    n = len(rows)
    print(f"\nTotal ticks: {n}")
    if n == 0:
        return

    # --- Spread distribution ---
    spreads = []
    for r in rows:
        if r["bid_price_1"] is not None and r["ask_price_1"] is not None:
            spreads.append(r["ask_price_1"] - r["bid_price_1"])

    two_sided = len(spreads)
    print(f"Two-sided ticks (have both bid1 & ask1): {two_sided} / {n} ({100*two_sided/n:.1f}%)")

    if spreads:
        spread_counter = Counter(spreads)
        print(f"\nSpread distribution (ask1 - bid1):")
        for s in sorted(spread_counter.keys()):
            cnt = spread_counter[s]
            bar = "#" * min(cnt, 80)
            print(f"  {s:6.0f}: {cnt:4d} ({100*cnt/two_sided:5.1f}%) {bar}")

        print(f"\n  Spread stats: min={min(spreads):.0f}, max={max(spreads):.0f}, "
              f"mean={sum(spreads)/len(spreads):.1f}, median={sorted(spreads)[len(spreads)//2]:.0f}")

    # --- Key spread thresholds ---
    le10 = sum(1 for s in spreads if s <= 10)
    le13 = sum(1 for s in spreads if s <= 13)
    print(f"\n  Spread <= 10: {le10} ticks ({100*le10/n:.1f}%)")
    print(f"  Spread <= 13: {le13} ticks ({100*le13/n:.1f}%)")

    # --- Mid price stats ---
    mids = [r["mid_price"] for r in rows if r["mid_price"] is not None and r["mid_price"] > 0]
    if mids:
        mean_mid = sum(mids) / len(mids)
        var_mid = sum((m - mean_mid)**2 for m in mids) / len(mids)
        std_mid = math.sqrt(var_mid)
        print(f"\nMid price: min={min(mids):.1f}, max={max(mids):.1f}, mean={mean_mid:.1f}, std={std_mid:.1f}")

    # --- Levels count ---
    three_bid = sum(1 for r in rows if r["bid_price_3"] is not None)
    three_ask = sum(1 for r in rows if r["ask_price_3"] is not None)
    three_both = sum(1 for r in rows if r["bid_price_3"] is not None and r["ask_price_3"] is not None)
    print(f"\nTicks with 3 bid levels: {three_bid} ({100*three_bid/n:.1f}%)")
    print(f"Ticks with 3 ask levels: {three_ask} ({100*three_ask/n:.1f}%)")
    print(f"Ticks with 3 levels on BOTH sides: {three_both} ({100*three_both/n:.1f}%)")

    # --- One-sided ticks ---
    only_bid = sum(1 for r in rows if r["bid_price_1"] is not None and r["ask_price_1"] is None)
    only_ask = sum(1 for r in rows if r["bid_price_1"] is None and r["ask_price_1"] is not None)
    neither = sum(1 for r in rows if r["bid_price_1"] is None and r["ask_price_1"] is None)
    print(f"\nOne-sided ticks — bids only: {only_bid}, asks only: {only_ask}, neither: {neither}")

    # --- Imbalance on narrow spread ticks ---
    print(f"\nImbalance on narrow-spread ticks (spread <= 13):")
    narrow_imbalances = []
    for r in rows:
        if r["bid_price_1"] is not None and r["ask_price_1"] is not None:
            spread = r["ask_price_1"] - r["bid_price_1"]
            if spread <= 13:
                bid_vol = sum(r[f"bid_volume_{i}"] or 0 for i in [1, 2, 3])
                ask_vol = sum(r[f"ask_volume_{i}"] or 0 for i in [1, 2, 3])
                total = bid_vol + ask_vol
                if total > 0:
                    imb = (bid_vol - ask_vol) / total  # +1 = all bids, -1 = all asks
                    narrow_imbalances.append(imb)
    if narrow_imbalances:
        mean_imb = sum(narrow_imbalances) / len(narrow_imbalances)
        print(f"  Count: {len(narrow_imbalances)}, mean imbalance (bid-ask)/total: {mean_imb:.3f}")
        # Bucket into ranges
        buckets = Counter()
        for imb in narrow_imbalances:
            if imb < -0.3:
                buckets["ask-heavy (<-0.3)"] += 1
            elif imb < 0.0:
                buckets["slight ask (-0.3 to 0)"] += 1
            elif imb < 0.3:
                buckets["slight bid (0 to 0.3)"] += 1
            else:
                buckets["bid-heavy (>0.3)"] += 1
        for k in ["ask-heavy (<-0.3)", "slight ask (-0.3 to 0)", "slight bid (0 to 0.3)", "bid-heavy (>0.3)"]:
            print(f"    {k}: {buckets.get(k, 0)}")

    # --- Bot trades ---
    print(f"\nBot trades: {len(trades)}")
    if trades:
        total_vol = sum(t["quantity"] for t in trades)
        prices = [t["price"] for t in trades]
        print(f"  Total volume: {total_vol}")
        print(f"  Price range: {min(prices):.0f} - {max(prices):.0f}")
        print(f"  Mean trade price: {sum(prices)/len(prices):.1f}")

        # Build a lookup: timestamp -> row
        ts_to_row = {}
        for r in rows:
            ts_to_row[r["timestamp"]] = r  # last one wins if multiple days (shouldn't happen per day)

        # For per-day analysis, build per (day,ts)
        dayts_to_row = {}
        for r in rows:
            dayts_to_row[(r["day"], r["timestamp"])] = r

        print(f"\n  Trade details (price relative to bid1/ask1 at same tick):")
        trade_spreads = []
        for t in trades:
            ts = t["timestamp"]
            # Try to find matching row - trades don't have day, so search
            matching = [r for r in rows if r["timestamp"] == ts]
            if matching:
                r = matching[0]  # take first match
                b1 = r["bid_price_1"]
                a1 = r["ask_price_1"]
                spread = (a1 - b1) if (a1 is not None and b1 is not None) else None
                if spread is not None:
                    trade_spreads.append(spread)
                rel_bid = f"{t['price'] - b1:+.0f}" if b1 else "N/A"
                rel_ask = f"{t['price'] - a1:+.0f}" if a1 else "N/A"
                print(f"    ts={ts:5d} price={t['price']:.0f} qty={t['quantity']:2d} "
                      f"bid1={b1} ask1={a1} spread={spread} "
                      f"vs_bid={rel_bid} vs_ask={rel_ask}")
            else:
                print(f"    ts={ts:5d} price={t['price']:.0f} qty={t['quantity']:2d} (no matching book tick)")

        if trade_spreads:
            trade_spread_counter = Counter(trade_spreads)
            print(f"\n  Spread at trade ticks:")
            for s in sorted(trade_spread_counter.keys()):
                print(f"    spread={s:.0f}: {trade_spread_counter[s]} trades")

    # --- Theoretical max PnL (simple mid-price FV) ---
    # If we could buy at bid1 when bid1 < mid and sell at ask1 when ask1 > mid
    # This is a rough upper bound on edge from market making
    if mids:
        mean_mid_val = sum(mids) / len(mids)
        total_edge = 0.0
        edge_ticks = 0
        for r in rows:
            if r["bid_price_1"] is not None and r["ask_price_1"] is not None:
                mid = r["mid_price"] if r["mid_price"] else mean_mid_val
                # Edge from buying at bid1 and selling at ask1 if spread is positive
                spread = r["ask_price_1"] - r["bid_price_1"]
                if spread > 0:
                    # Approx: edge per unit = spread / 2 (you capture half the spread)
                    min_vol = min(r["bid_volume_1"] or 0, r["ask_volume_1"] or 0)
                    edge = (spread / 2.0) * min_vol
                    total_edge += edge
                    edge_ticks += 1
        print(f"\nTheoretical edge (spread/2 * min(bid1_vol, ask1_vol) per tick):")
        print(f"  Total edge over {edge_ticks} two-sided ticks: {total_edge:.0f}")
        print(f"  Avg edge per tick: {total_edge/edge_ticks:.1f}" if edge_ticks else "  N/A")

    # --- Exploitable opportunities ---
    print(f"\n--- EXPLOITABLE OPPORTUNITIES ---")
    opp_count = 0
    opp_by_spread = Counter()
    for r in rows:
        if r["bid_price_1"] is not None and r["ask_price_1"] is not None:
            spread = r["ask_price_1"] - r["bid_price_1"]
            # If spread > 2 (we can post inside), there's opportunity
            if spread >= 4:  # need at least 2 ticks of room on each side
                opp_count += 1
                bucket = f"{int(spread)}"
                opp_by_spread[bucket] += 1

    print(f"  Ticks where spread >= 4 (room to post inside): {opp_count} / {n} ({100*opp_count/n:.1f}%)")
    if opp_by_spread:
        print(f"  By spread value:")
        for s in sorted(opp_by_spread.keys(), key=int):
            print(f"    spread={s}: {opp_by_spread[s]} ticks")

    # Edge if posting at mid (capturing spread/2 minus some adverse selection)
    profitable_ticks = 0
    for r in rows:
        if r["bid_price_1"] is not None and r["ask_price_1"] is not None:
            spread = r["ask_price_1"] - r["bid_price_1"]
            if spread >= 10:  # wide enough to be confident
                profitable_ticks += 1
    print(f"  Ticks with spread >= 10 (high confidence): {profitable_ticks} ({100*profitable_ticks/n:.1f}%)")
    print(f"  Ticks with spread >= 16 (very wide): "
          f"{sum(1 for r in rows if r['bid_price_1'] and r['ask_price_1'] and r['ask_price_1']-r['bid_price_1']>=16)}")


def main():
    data = load_data(LOG_PATH)
    rows = parse_activities(data)
    trades = parse_trades(data)

    # Figure out which days exist
    days = sorted(set(r["day"] for r in rows))
    print(f"Product: {PRODUCT}")
    print(f"Days found: {days}")
    print(f"Total ticks: {len(rows)}, Total trades: {len(trades)}")

    # Per-day analysis
    for day in days:
        day_rows = [r for r in rows if r["day"] == day]
        # Trades don't have day field; infer from timestamp range
        # Day 0: ts 0-99999, Day 1: ts 0-99999 etc. — actually trades lack day
        # Since dummy trader, all trades are bot-to-bot; we'll assign based on timestamp
        # For multi-day, we need to figure out day boundaries
        # Let's just use all trades for overall and note this limitation
        day_trades = trades if len(days) == 1 else []  # handle below
        compute_stats(day_rows, day_trades, label=f"Day {day}")

    # Overall
    if len(days) > 1:
        compute_stats(rows, trades, label="ALL DAYS")

    # Extra: trade timing pattern
    if trades:
        print(f"\n{'='*70}")
        print(f"  TRADE TIMING ANALYSIS")
        print(f"{'='*70}")
        ts_list = [t["timestamp"] for t in trades]
        gaps = [ts_list[i+1] - ts_list[i] for i in range(len(ts_list)-1)]
        if gaps:
            gap_counter = Counter(gaps)
            print(f"\nGaps between consecutive trades:")
            for g in sorted(gap_counter.keys())[:20]:
                print(f"  gap={g}: {gap_counter[g]} times")
            print(f"  Min gap: {min(gaps)}, Max gap: {max(gaps)}, Mean: {sum(gaps)/len(gaps):.0f}")

        # Trade price movement
        print(f"\nTrade price sequence:")
        for t in trades:
            print(f"  ts={t['timestamp']:5d} price={t['price']:.0f} qty={t['quantity']}")

if __name__ == "__main__":
    main()
