"""Round-4 buyer-seller pair correlation analysis.

Question: do buyer-seller PAIR identities carry signal beyond what individual
identity already gives us? The active algo only uses single-bot identity
(Mark 67 buy / Mark 49 sell, optionally Mark 14 mirror). If a specific pair
(buyer X vs seller Y) has reliably different price impact than the same buyer
or seller paired with someone else, we can stratify the signal by pair.

Five outputs:
  1. Per-product pair frequency matrix (buyer x seller)
  2. Pair-conditioned lead-lag at h=10/100/500 (qty-weighted mean mid move)
  3. Aggressor-side stratification per pair
  4. Per-day stability check (does the pair signal hold across days?)
  5. Direct comparison vs identity-only signal for the same events
"""
from __future__ import annotations
import csv, json
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, pstdev

DATA = Path("data/ROUND_4")
DAYS = [1, 2, 3]
BOTS = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]
PRODUCTS_OF_INTEREST = [
    "HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100",
    "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500",
]
HORIZONS = [10, 50, 100, 500, 1000]


def load_day(day: int):
    mids = defaultdict(dict)
    book = defaultdict(dict)
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
                "day": day,
            })
    return mids, book, trades


def mid_at(mids_p, ts):
    """Return mid at the latest timestamp <= ts."""
    if ts in mids_p:
        return mids_p[ts]
    keys = sorted(mids_p.keys())
    lo, hi = 0, len(keys) - 1
    best = None
    while lo <= hi:
        m = (lo + hi) // 2
        if keys[m] <= ts:
            best = keys[m]; lo = m + 1
        else:
            hi = m - 1
    return mids_p[best] if best is not None else None


def aggr_side(price: float, mid: float):
    """Classify trade aggressor side. price > mid+0.5 -> buyer aggressed.
    price < mid-0.5 -> seller aggressed. Otherwise at-mid (passive both)."""
    if mid is None:
        return "unk"
    if price > mid + 0.5:
        return "buy_agg"
    if price < mid - 0.5:
        return "sell_agg"
    return "at_mid"


def main():
    all_data = {d: load_day(d) for d in DAYS}

    # ===== 1. Pair frequency matrix per product =====
    print("=" * 78)
    print("1. PAIR FREQUENCY (qty) per product (buyer x seller; '<2 pairs' grouped)")
    print("=" * 78)
    for prod in PRODUCTS_OF_INTEREST:
        pair_qty = Counter()
        total = 0
        for d in DAYS:
            for t in all_data[d][2]:
                if t["symbol"] != prod:
                    continue
                pair_qty[(t["buyer"], t["seller"])] += t["qty"]
                total += t["qty"]
        if total == 0:
            continue
        print(f"\n{prod}  total_qty={total}")
        rows = sorted(pair_qty.items(), key=lambda x: -x[1])
        for (b, s), q in rows[:8]:
            print(f"  {b:<10} -> {s:<10}  qty={q:>6}  ({100*q/total:>5.1f}%)")

    # ===== 2. Pair-conditioned lead-lag =====
    print("\n" + "=" * 78)
    print("2. PAIR-CONDITIONED LEAD-LAG (qty-weighted mean mid move at h ticks)")
    print("   Sign convention: positive = mid up after buyer's lift")
    print("   Only pairs with n>=30 across 3 days shown.")
    print("=" * 78)
    pair_results = {}
    for prod in PRODUCTS_OF_INTEREST:
        # Per pair: list of (qty, mid_now, [mid_now+h for h in HORIZONS])
        events = defaultdict(list)
        for d in DAYS:
            mids, book, trades = all_data[d]
            mids_p = mids.get(prod, {})
            if not mids_p:
                continue
            for t in trades:
                if t["symbol"] != prod:
                    continue
                ts = t["ts"]
                # observation point shifted by 100 ticks (matches lagged BT
                # convention: trader sees previous-tick trades, so signal arrives
                # at ts+100 from price's POV)
                t_obs = ts + 100
                mid_obs = mid_at(mids_p, t_obs)
                if mid_obs is None:
                    continue
                future = []
                for h in HORIZONS:
                    mh = mid_at(mids_p, t_obs + h)
                    future.append(mh)
                events[(t["buyer"], t["seller"])].append((t["qty"], mid_obs, future))
        # Print per-pair summary, ranked by total qty
        rows = []
        for pair, evs in events.items():
            n = len(evs)
            if n < 30:
                continue
            total_qty = sum(e[0] for e in evs)
            row = {"pair": pair, "n": n, "qty": total_qty}
            for i, h in enumerate(HORIZONS):
                num = den = 0.0
                for qty, m_now, fut in evs:
                    mh = fut[i]
                    if mh is None:
                        continue
                    num += qty * (mh - m_now)
                    den += qty
                row[f"h{h}"] = num / den if den else 0.0
            rows.append(row)
        if not rows:
            continue
        rows.sort(key=lambda r: -r["qty"])
        print(f"\n{prod}")
        hdr = f"  {'pair':<26}{'n':>6}{'qty':>8}" + "".join(f"{'h'+str(h):>9}" for h in HORIZONS)
        print(hdr)
        for r in rows:
            tag = f"{r['pair'][0]} -> {r['pair'][1]}"
            line = f"  {tag:<26}{r['n']:>6}{r['qty']:>8}" + "".join(f"{r[f'h{h}']:>+9.2f}" for h in HORIZONS)
            print(line)
        pair_results[prod] = rows

    # ===== 3. Aggressor-side stratification per pair =====
    print("\n" + "=" * 78)
    print("3. AGGRESSOR-SIDE STRATIFICATION (price vs mid at exec)")
    print("   Splits each pair's trades into buy_agg / at_mid / sell_agg buckets.")
    print("   Shows h=100 mid move per bucket.")
    print("=" * 78)
    aggr_results = {}
    for prod in PRODUCTS_OF_INTEREST:
        # bucket events by (pair, aggr_side)
        bucks = defaultdict(list)
        for d in DAYS:
            mids, book, trades = all_data[d]
            mids_p = mids.get(prod, {})
            if not mids_p:
                continue
            for t in trades:
                if t["symbol"] != prod:
                    continue
                ts = t["ts"]
                mid_exec = mid_at(mids_p, ts)  # mid at trade time (for aggressor classification)
                side = aggr_side(t["price"], mid_exec)
                t_obs = ts + 100
                m_obs = mid_at(mids_p, t_obs)
                m_h = mid_at(mids_p, t_obs + 100)
                if m_obs is None or m_h is None:
                    continue
                bucks[(t["buyer"], t["seller"], side)].append((t["qty"], m_h - m_obs))
        # Aggregate
        rows = []
        for (b, s, side), evs in bucks.items():
            n = len(evs)
            if n < 15:
                continue
            tq = sum(e[0] for e in evs)
            num = sum(e[0] * e[1] for e in evs)
            mu = num / tq if tq else 0.0
            rows.append({"pair": (b, s), "side": side, "n": n, "qty": tq, "h100": mu})
        if not rows:
            continue
        rows.sort(key=lambda r: (r["pair"], r["side"]))
        print(f"\n{prod}")
        print(f"  {'pair':<26}{'aggr':>10}{'n':>6}{'qty':>8}{'mu_h100':>10}")
        for r in rows:
            tag = f"{r['pair'][0]} -> {r['pair'][1]}"
            print(f"  {tag:<26}{r['side']:>10}{r['n']:>6}{r['qty']:>8}{r['h100']:>+10.2f}")
        aggr_results[prod] = rows

    # ===== 4. Per-day stability of pair signals =====
    print("\n" + "=" * 78)
    print("4. PER-DAY STABILITY (h=100 mean per pair per day, n>=20/day)")
    print("=" * 78)
    for prod in PRODUCTS_OF_INTEREST:
        per_pair_day = defaultdict(lambda: defaultdict(list))
        for d in DAYS:
            mids, book, trades = all_data[d]
            mids_p = mids.get(prod, {})
            if not mids_p:
                continue
            for t in trades:
                if t["symbol"] != prod:
                    continue
                ts = t["ts"]
                t_obs = ts + 100
                m_obs = mid_at(mids_p, t_obs)
                m_h = mid_at(mids_p, t_obs + 100)
                if m_obs is None or m_h is None:
                    continue
                per_pair_day[(t["buyer"], t["seller"])][d].append((t["qty"], m_h - m_obs))
        rows = []
        for pair, by_day in per_pair_day.items():
            day_means = {}
            for d, evs in by_day.items():
                if len(evs) < 20:
                    continue
                tq = sum(e[0] for e in evs)
                num = sum(e[0] * e[1] for e in evs)
                day_means[d] = (num / tq if tq else 0.0, len(evs))
            if len(day_means) >= 2:
                rows.append((pair, day_means))
        if not rows:
            continue
        print(f"\n{prod}")
        print(f"  {'pair':<26}" + "".join(f"{'D'+str(d):>14}" for d in DAYS))
        for pair, dm in sorted(rows, key=lambda x: -sum(v[1] for v in x[1].values())):
            tag = f"{pair[0]} -> {pair[1]}"
            cells = []
            for d in DAYS:
                if d in dm:
                    mu, n = dm[d]
                    cells.append(f"{mu:>+8.2f}(n={n:>3})")
                else:
                    cells.append(f"{'--':>14}")
            print(f"  {tag:<26}" + "".join(cells))


if __name__ == "__main__":
    main()
