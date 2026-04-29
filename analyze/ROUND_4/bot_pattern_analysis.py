"""Round-4 bot pattern analysis (deeper dive).

Five investigations:
  1. Per-bot daily PnL (mark-to-market at day-end mid).
  2. Lead-lag: does bot X's trade direction predict price moves at h ticks?
  3. Pair synchrony: how often do paired bots trade in the same ticks?
  4. Trade burst structure: inter-trade interval distribution.
  5. Cumulative edge captured per bot per day.

Reads CSVs directly; no algo invocation needed.
"""
from __future__ import annotations
import csv
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, median, pstdev

DATA = Path("data/ROUND_4")
DAYS = [1, 2, 3]
BOTS = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]
PAIRS = [
    ("Mark 01", "Mark 22"),  # voucher buyer / seller
    ("Mark 14", "Mark 38"),  # MM / aggressor (HYD)
    ("Mark 14", "Mark 55"),  # MM / aggressor (VELVET)
    ("Mark 67", "Mark 49"),  # VELVET buyer / seller
]


def load_day(day: int):
    mids = defaultdict(dict)            # product -> {ts: mid}
    book = defaultdict(dict)            # product -> {ts: (bid1, ask1)}
    trades = []
    with open(DATA / f"prices_round_4_day_{day}.csv") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                m = float(r["mid_price"])
            except (TypeError, ValueError):
                continue
            ts = int(r["timestamp"])
            mids[r["product"]][ts] = m
            try:
                bid = int(float(r.get("bid_price_1") or 0))
                ask = int(float(r.get("ask_price_1") or 0))
                if bid and ask:
                    book[r["product"]][ts] = (bid, ask)
            except (TypeError, ValueError):
                pass
    with open(DATA / f"trades_round_4_day_{day}.csv") as f:
        for r in csv.DictReader(f, delimiter=";"):
            trades.append({
                "ts": int(r["timestamp"]),
                "buyer": r["buyer"], "seller": r["seller"],
                "symbol": r["symbol"],
                "price": float(r["price"]),
                "qty": int(float(r["quantity"])),
            })
    return mids, book, trades


# ---------------------------------------------------------------------------
# 1. Per-bot daily PnL (mark-to-market)
# ---------------------------------------------------------------------------
def bot_pnl():
    print("=" * 90)
    print("1) PER-BOT DAILY PnL (cash + final mark)")
    print("=" * 90)
    print(f"{'bot':<10} " + " ".join(f"{f'D{d}':>10}" for d in DAYS) + f"  {'TOTAL':>10}")
    print("-" * 90)
    grand = defaultdict(float)
    per_day = {}
    for d in DAYS:
        mids, _, trades = load_day(d)
        cash = defaultdict(float)
        pos = defaultdict(lambda: defaultdict(int))  # bot -> {sym: pos}
        for t in trades:
            sym, qty, px = t["symbol"], t["qty"], t["price"]
            if t["buyer"] in BOTS:
                cash[t["buyer"]] -= px * qty
                pos[t["buyer"]][sym] += qty
            if t["seller"] in BOTS:
                cash[t["seller"]] += px * qty
                pos[t["seller"]][sym] -= qty
        # Mark-to-market at last available mid per product
        bot_pnl_day = {}
        for bot in BOTS:
            mtm = cash[bot]
            for sym, p in pos[bot].items():
                last_ts = max(mids[sym].keys()) if mids[sym] else None
                if last_ts is None:
                    continue
                mtm += p * mids[sym][last_ts]
            bot_pnl_day[bot] = mtm
            grand[bot] += mtm
        per_day[d] = bot_pnl_day
    for bot in BOTS:
        line = f"{bot:<10} " + " ".join(f"{per_day[d][bot]:>+10.0f}" for d in DAYS)
        line += f"  {grand[bot]:>+10.0f}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# 2. Lead-lag: does a bot's trade direction predict price moves?
# ---------------------------------------------------------------------------
def lead_lag():
    print("=" * 110)
    print("2) LEAD-LAG: avg signed price move (mid_{t+h} - mid_t) per bot trade")
    print("   Sign convention: bot was buyer → +1; bot was seller → -1.")
    print("   Positive avg = bot leads (buys before up-move). Negative = bot lags.")
    print("=" * 110)
    horizons = [1, 5, 25, 100, 500, 2000]
    print(f"{'bot':<10} {'product':<24} {'n':>5} " +
          " ".join(f"{f'h={h}':>9}" for h in horizons))
    print("-" * 110)
    rows = []
    for d in DAYS:
        mids, _, trades = load_day(d)
        for t in trades:
            sym = t["symbol"]
            ts = t["ts"]
            qty = t["qty"]
            for bot in (t["buyer"], t["seller"]):
                if bot not in BOTS:
                    continue
                sign = +1 if bot == t["buyer"] else -1
                impacts = []
                m_now = mids[sym].get(ts)
                if m_now is None:
                    continue
                for h in horizons:
                    m_fut = mids[sym].get(ts + h * 100)
                    if m_fut is None:
                        impacts.append(None)
                    else:
                        impacts.append(sign * (m_fut - m_now) * qty)
                rows.append((bot, sym, qty, impacts))
    # Aggregate
    agg = defaultdict(lambda: {"n": 0, "qty": 0, "impacts": [0.0] * len(horizons),
                                "counts": [0] * len(horizons)})
    for bot, sym, qty, imp in rows:
        a = agg[(bot, sym)]
        a["n"] += 1
        a["qty"] += qty
        for i, val in enumerate(imp):
            if val is not None:
                a["impacts"][i] += val
                a["counts"][i] += qty
    # Print bots × products with at least 200 qty
    sorted_keys = sorted(agg.keys(), key=lambda k: -agg[k]["qty"])
    for bot, sym in sorted_keys:
        a = agg[(bot, sym)]
        if a["qty"] < 200:
            continue
        avgs = [a["impacts"][i] / a["counts"][i] if a["counts"][i] else 0
                for i in range(len(horizons))]
        line = f"{bot:<10} {sym:<24} {a['n']:>5} " + " ".join(f"{x:>+9.3f}" for x in avgs)
        print(line)
    print()


# ---------------------------------------------------------------------------
# 3. Pair synchrony
# ---------------------------------------------------------------------------
def pair_synchrony():
    print("=" * 90)
    print("3) PAIR SYNCHRONY: do paired bots trade in the same tick (or near it)?")
    print("=" * 90)
    print(f"{'pair':<25} {'product':<24} {'A_ticks':>8} {'B_ticks':>8} "
          f"{'overlap':>8} {'within+/-5':>10}")
    print("-" * 90)
    for botA, botB in PAIRS:
        for d in DAYS:
            mids, _, trades = load_day(d)
            # Group trades by (sym, ts) per bot
            ticks_A = defaultdict(set)  # sym -> {ts}
            ticks_B = defaultdict(set)
            for t in trades:
                sym = t["symbol"]
                if t["buyer"] == botA or t["seller"] == botA:
                    ticks_A[sym].add(t["ts"])
                if t["buyer"] == botB or t["seller"] == botB:
                    ticks_B[sym].add(t["ts"])
            # All products that both touch
            prods = sorted(set(ticks_A.keys()) & set(ticks_B.keys()),
                          key=lambda p: -(len(ticks_A[p]) + len(ticks_B[p])))
            for sym in prods[:1]:  # report only top product per pair per day
                A, B = ticks_A[sym], ticks_B[sym]
                exact = A & B
                # within ±5 ticks (=500 timestamp units)
                near = 0
                B_sorted = sorted(B)
                import bisect
                for ta in A:
                    lo = bisect.bisect_left(B_sorted, ta - 500)
                    hi = bisect.bisect_right(B_sorted, ta + 500)
                    if hi > lo:
                        near += 1
                pair_name = f"{botA}<>{botB} D{d}"
                print(f"{pair_name:<25} {sym:<24} "
                      f"{len(A):>8} {len(B):>8} {len(exact):>8} {near:>10}")
    print()


# ---------------------------------------------------------------------------
# 4. Burst structure
# ---------------------------------------------------------------------------
def burst_structure():
    print("=" * 90)
    print("4) TRADE BURST STRUCTURE — inter-trade interval (in ticks=100 ts units)")
    print("=" * 90)
    print(f"{'bot':<10} {'product':<24} {'n':>5} {'medDt':>7} {'p10':>6} "
          f"{'p90':>6} {'p99':>6} {'maxgap':>7} {'cluster%':>9}")
    print("-" * 90)
    for d in DAYS:
        mids, _, trades = load_day(d)
        ts_by_bot_prod = defaultdict(list)
        for t in trades:
            sym = t["symbol"]
            for bot in (t["buyer"], t["seller"]):
                if bot in BOTS:
                    ts_by_bot_prod[(bot, sym)].append(t["ts"])
        for (bot, sym), ts_list in sorted(ts_by_bot_prod.items(),
                                           key=lambda kv: -len(kv[1])):
            if len(ts_list) < 50:
                continue
            ts_list.sort()
            diffs = [(ts_list[i] - ts_list[i-1]) // 100
                     for i in range(1, len(ts_list)) if ts_list[i] != ts_list[i-1]]
            if not diffs:
                continue
            diffs_sorted = sorted(diffs)
            n = len(diffs_sorted)
            med = diffs_sorted[n // 2]
            p10 = diffs_sorted[n // 10] if n > 10 else diffs_sorted[0]
            p90 = diffs_sorted[(9 * n) // 10] if n > 10 else diffs_sorted[-1]
            p99 = diffs_sorted[(99 * n) // 100] if n > 100 else diffs_sorted[-1]
            maxgap = max(diffs_sorted)
            # Cluster% = fraction of trades within 5 ticks of next trade
            cluster = sum(1 for d_ in diffs if d_ <= 5) / n * 100
            print(f"{bot:<10} {sym:<24} {len(ts_list):>5} "
                  f"{med:>7d} {p10:>6d} {p90:>6d} {p99:>6d} "
                  f"{maxgap:>7d} {cluster:>8.1f}%")
        if d != DAYS[-1]:
            print(f"  -- D{d} ↑")
    print()


# ---------------------------------------------------------------------------
# 5. Per-bot edge captured per day
# ---------------------------------------------------------------------------
def edge_captured():
    print("=" * 90)
    print("5) EDGE CAPTURED — sum (mid - price) for buys, (price - mid) for sells")
    print("   Interpretation: total dollars of mid-edge captured across all trades.")
    print("=" * 90)
    print(f"{'bot':<10} " + " ".join(f"{f'D{d}':>10}" for d in DAYS) + f"  {'TOTAL':>10}")
    print("-" * 90)
    totals = defaultdict(float)
    per_day = {}
    for d in DAYS:
        mids, _, trades = load_day(d)
        per_bot = defaultdict(float)
        for t in trades:
            m = mids[t["symbol"]].get(t["ts"])
            if m is None:
                continue
            if t["buyer"] in BOTS:
                per_bot[t["buyer"]] += (m - t["price"]) * t["qty"]
            if t["seller"] in BOTS:
                per_bot[t["seller"]] += (t["price"] - m) * t["qty"]
        per_day[d] = per_bot
        for b, v in per_bot.items():
            totals[b] += v
    for bot in BOTS:
        line = f"{bot:<10} " + " ".join(f"{per_day[d].get(bot, 0):>+10.0f}" for d in DAYS)
        line += f"  {totals[bot]:>+10.0f}"
        print(line)
    print()


def main():
    bot_pnl()
    lead_lag()
    pair_synchrony()
    burst_structure()
    edge_captured()


if __name__ == "__main__":
    main()
