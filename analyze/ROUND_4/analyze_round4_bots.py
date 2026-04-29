"""Round-4 bot behavior analysis.

For each counterparty in trades_round_4_day_*.csv, computes:
  - total volume, side balance (buy vs sell)
  - per-product specialization
  - aggressiveness: lift / hit / at-mid (vs same-tick mid)
  - avg trade size, ticks active
  - per-product net direction (drift exposure)

Output shows the distinct behavioral profiles of the bots.
"""
from __future__ import annotations
import csv
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, median

DATA = Path("data/ROUND_4")
DAYS = [1, 2, 3]


def load_mids(day: int):
    mids = {}  # (ts, product) -> mid
    with open(DATA / f"prices_round_4_day_{day}.csv") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            p = row["product"]
            try:
                mids[(ts, p)] = float(row["mid_price"])
            except (TypeError, ValueError):
                pass
    return mids


def load_trades(day: int):
    out = []
    with open(DATA / f"trades_round_4_day_{day}.csv") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            out.append({
                "ts": int(row["timestamp"]),
                "buyer": row["buyer"],
                "seller": row["seller"],
                "symbol": row["symbol"],
                "price": float(row["price"]),
                "qty": int(float(row["quantity"])),
            })
    return out


def analyze():
    all_trades = []
    mids = {}
    for d in DAYS:
        ts_off = (d - 1) * 1_000_000
        for t in load_trades(d):
            t["day"] = d
            t["ts_global"] = t["ts"] + ts_off
            all_trades.append(t)
        for (ts, p), m in load_mids(d).items():
            mids[(d, ts, p)] = m

    bots = sorted({t["buyer"] for t in all_trades} | {t["seller"] for t in all_trades})
    print(f"Total trades: {len(all_trades)} across {len(DAYS)} days")
    print(f"Distinct counterparties: {len(bots)}: {bots}")
    print()

    # Per-bot accumulators
    per_bot = defaultdict(lambda: {
        "buys_qty": 0, "sells_qty": 0, "buys_n": 0, "sells_n": 0,
        # Aggressive role: crossed the spread to take liquidity.
        # Passive role: had a quote that was traded against.
        "agg_buy": 0,    # bot lifted ask (price > mid)
        "agg_sell": 0,   # bot hit bid (price < mid)
        "passive": 0,    # bot's resting quote got lifted/hit
        "atmid": 0, "no_mid": 0,
        "by_product_qty": defaultdict(int),
        "by_product_volume": defaultdict(int),
        "qty_sizes": [],
        "ticks": set(),
        "edge_sum": 0.0,   # bot's perspective: + means traded better than mid
        "edge_n": 0,
    })

    # Pair counter
    pair_counts = Counter()

    for t in all_trades:
        b, s = t["buyer"], t["seller"]
        sym = t["symbol"]
        qty = t["qty"]
        m = mids.get((t["day"], t["ts"], sym))
        pair_counts[(b, s)] += qty

        # Buyer side
        per_bot[b]["buys_qty"] += qty
        per_bot[b]["buys_n"] += 1
        per_bot[b]["by_product_qty"][sym] += qty
        per_bot[b]["by_product_volume"][sym] += qty
        per_bot[b]["qty_sizes"].append(qty)
        per_bot[b]["ticks"].add((t["day"], t["ts"]))
        # Seller side
        per_bot[s]["sells_qty"] += qty
        per_bot[s]["sells_n"] += 1
        per_bot[s]["by_product_qty"][sym] -= qty
        per_bot[s]["by_product_volume"][sym] += qty
        per_bot[s]["qty_sizes"].append(qty)
        per_bot[s]["ticks"].add((t["day"], t["ts"]))

        # Aggressor classification vs same-tick mid.
        # price > mid  -> buyer lifted seller's ask (buyer aggressive)
        # price < mid  -> seller hit buyer's bid    (seller aggressive)
        # Bot edge (bot's perspective): + = better than mid.
        if m is None:
            per_bot[b]["no_mid"] += 1
            per_bot[s]["no_mid"] += 1
            continue
        eps = 0.5
        # Edge for buyer = mid - price; for seller = price - mid.
        per_bot[b]["edge_sum"] += (m - t["price"])
        per_bot[b]["edge_n"] += 1
        per_bot[s]["edge_sum"] += (t["price"] - m)
        per_bot[s]["edge_n"] += 1
        if t["price"] > m + eps:
            per_bot[b]["agg_buy"] += 1
            per_bot[s]["passive"] += 1
        elif t["price"] < m - eps:
            per_bot[s]["agg_sell"] += 1
            per_bot[b]["passive"] += 1
        else:
            per_bot[b]["atmid"] += 1
            per_bot[s]["atmid"] += 1

    # ---- Print summary table ----
    print("=" * 110)
    print("BOT BEHAVIOR PROFILE")
    print("=" * 110)
    print(f"{'bot':<10} {'trades':>7} {'qty':>7} {'buy%':>6} {'sell%':>6} "
          f"{'aggBuy%':>7} {'aggSell%':>8} {'pasv%':>6} {'mid%':>6} "
          f"{'avgEdge':>8} {'avgSz':>6} {'ticks':>6}")
    print("-" * 110)

    rows = []
    for bot in bots:
        s = per_bot[bot]
        n = s["buys_n"] + s["sells_n"]
        if n == 0:
            continue
        total_qty = s["buys_qty"] + s["sells_qty"]
        buy_pct = 100 * s["buys_qty"] / total_qty
        sell_pct = 100 * s["sells_qty"] / total_qty
        n_class = s["agg_buy"] + s["agg_sell"] + s["passive"] + s["atmid"]
        if n_class == 0:
            ab = asell = pas = mid_pct = 0.0
        else:
            ab = 100 * s["agg_buy"] / n_class
            asell = 100 * s["agg_sell"] / n_class
            pas = 100 * s["passive"] / n_class
            mid_pct = 100 * s["atmid"] / n_class
        avg_edge = s["edge_sum"] / s["edge_n"] if s["edge_n"] else 0.0
        avg_sz = mean(s["qty_sizes"]) if s["qty_sizes"] else 0.0
        rows.append((bot, n, total_qty, buy_pct, sell_pct,
                     ab, asell, pas, mid_pct, avg_edge, avg_sz,
                     len(s["ticks"])))

    rows.sort(key=lambda r: -r[2])
    for bot, n, qty, bp, sp, ab, asell, pas, mp, ae, sz, ticks in rows:
        print(f"{bot:<10} {n:>7} {qty:>7} {bp:>5.0f}% {sp:>5.0f}% "
              f"{ab:>6.0f}% {asell:>7.0f}% {pas:>5.0f}% {mp:>5.0f}% "
              f"{ae:>+8.2f} {sz:>6.1f} {ticks:>6}")

    # ---- Per-product specialization ----
    print()
    print("=" * 110)
    print("PRODUCT SPECIALIZATION (top product per bot, with net & total volume)")
    print("=" * 110)
    print(f"{'bot':<10}  {'top product':<24} {'vol':>6} {'net':>7}  | "
          f"{'2nd product':<24} {'vol':>6} {'net':>7}")
    for bot, *_ in rows:
        bp = per_bot[bot]["by_product_volume"]
        bn = per_bot[bot]["by_product_qty"]
        ranked = sorted(bp.items(), key=lambda kv: -kv[1])
        top = ranked[0] if len(ranked) >= 1 else ("-", 0)
        snd = ranked[1] if len(ranked) >= 2 else ("-", 0)
        print(f"{bot:<10}  {top[0]:<24} {top[1]:>6} {bn.get(top[0], 0):>+7}  | "
              f"{snd[0]:<24} {snd[1]:>6} {bn.get(snd[0], 0):>+7}")

    # ---- Net directional bet per bot per product (top abs) ----
    print()
    print("=" * 110)
    print("DIRECTIONAL BIAS — largest |net| (signed: + = net long, - = net short)")
    print("=" * 110)
    flat_net = []
    for bot in bots:
        for sym, q in per_bot[bot]["by_product_qty"].items():
            if abs(q) >= 5:
                flat_net.append((abs(q), bot, sym, q))
    flat_net.sort(reverse=True)
    print(f"{'bot':<10} {'symbol':<24} {'net':>7}")
    for _, bot, sym, q in flat_net[:25]:
        print(f"{bot:<10} {sym:<24} {q:>+7}")

    # ---- Pair flow ----
    print()
    print("=" * 110)
    print("TOP COUNTERPARTY PAIRS (buyer -> seller, total quantity)")
    print("=" * 110)
    for (b, s), q in pair_counts.most_common(15):
        print(f"  {b:<10} -> {s:<10}  qty={q}")


if __name__ == "__main__":
    analyze()
