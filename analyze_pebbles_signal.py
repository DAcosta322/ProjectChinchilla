"""Diagnostic: characterize the basket-residual signal and basket-arb fires
across PEBBLES days 2/3/4. Used to ground the deep-analysis writeup.

Outputs to data/ROUND_5/pebbles_signal/.
  - dev_dist.csv: per-day dev_XL = sum_mids - 50000 distribution stats
  - arb_count.csv: per-day basket arb fire counts (sum_bb > 50000 / sum_ba < 50000)
  - per_leg_spread.csv: per-leg spread distribution
  - dev_vs_pnl.csv: dev_XL bucket vs realized 1-tick mid change per leg
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/ROUND_5")
OUT_DIR = Path("data/ROUND_5/pebbles_signal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
DAYS = [2, 3, 4]


def load_day(day: int):
    """Returns {ts: {prod: (bb, ba, mid)}}"""
    path = DATA_DIR / f"prices_round_5_day_{day}.csv"
    rows = defaultdict(dict)
    with open(path) as f:
        rd = csv.DictReader(f, delimiter=";")
        for r in rd:
            prod = r["product"]
            if prod not in PEBBLES:
                continue
            ts = int(r["timestamp"])
            try:
                bb = int(float(r["bid_price_1"]))
                ba = int(float(r["ask_price_1"]))
            except Exception:
                continue
            mid = (bb + ba) / 2.0
            rows[ts][prod] = (bb, ba, mid)
    return rows


def stats(values):
    if not values:
        return (0, 0.0, 0.0, 0.0, 0.0)
    n = len(values)
    s = sum(values)
    mean = s / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    sd = var ** 0.5
    sv = sorted(values)
    p1 = sv[max(0, n // 100)]
    p99 = sv[min(n - 1, n - 1 - n // 100)]
    return n, mean, sd, p1, p99


def main():
    dev_rows = []
    arb_rows = []
    spread_rows = []
    dev_pnl_rows = []

    for day in DAYS:
        snaps = load_day(day)
        sorted_ts = sorted(snaps.keys())
        # build per-day series
        devs = []
        sum_bb_minus = []  # arb-sell signals
        sum_ba_minus = []  # arb-buy signals
        per_leg_spreads = defaultdict(list)
        # for dev vs forward 1-step return per leg
        dev_buckets = defaultdict(lambda: defaultdict(list))

        prev_mids = None
        for i, ts in enumerate(sorted_ts):
            rec = snaps[ts]
            if not all(p in rec for p in PEBBLES):
                prev_mids = None
                continue
            mids = {p: rec[p][2] for p in PEBBLES}
            sum_mids = sum(mids.values())
            dev = sum_mids - 50000
            devs.append(dev)
            sum_bb = sum(rec[p][0] for p in PEBBLES)
            sum_ba = sum(rec[p][1] for p in PEBBLES)
            sum_bb_minus.append(sum_bb - 50000)
            sum_ba_minus.append(sum_ba - 50000)
            for p in PEBBLES:
                per_leg_spreads[p].append(rec[p][1] - rec[p][0])
            # forward 1-step return per leg
            if prev_mids is not None:
                bucket = round(dev)  # integer dev bucket
                for p in PEBBLES:
                    dev_buckets[p][bucket].append(mids[p] - prev_mids[p])
            prev_mids = mids

        n, m, sd, p1, p99 = stats(devs)
        dev_rows.append({"day": day, "n": n, "dev_mean": m, "dev_sd": sd, "dev_p1": p1, "dev_p99": p99})

        arb_sell_fires = sum(1 for v in sum_bb_minus if v > 0)
        arb_buy_fires = sum(1 for v in sum_ba_minus if v < 0)
        arb_rows.append({
            "day": day, "n_ticks": len(sum_bb_minus),
            "arb_sell_fires": arb_sell_fires, "arb_sell_pct": 100*arb_sell_fires/max(1,len(sum_bb_minus)),
            "arb_buy_fires": arb_buy_fires, "arb_buy_pct": 100*arb_buy_fires/max(1,len(sum_ba_minus)),
            "max_sum_bb_minus": max(sum_bb_minus) if sum_bb_minus else 0,
            "min_sum_ba_minus": min(sum_ba_minus) if sum_ba_minus else 0,
        })

        for p in PEBBLES:
            sp = per_leg_spreads[p]
            n2, m2, sd2, p1_2, p99_2 = stats(sp)
            spread_rows.append({"day": day, "product": p, "n": n2, "spread_mean": m2, "spread_p1": p1_2, "spread_p99": p99_2})

        # dev → forward 1-step return: for each leg, correlation of dev with next return
        for p in PEBBLES:
            buckets = dev_buckets[p]
            # collapse to overall covariance
            xs = []  # dev values
            ys = []  # next returns
            # rebuild flat list by walking buckets
            for b, lst in buckets.items():
                for r in lst:
                    xs.append(b)
                    ys.append(r)
            if len(xs) > 100:
                n3 = len(xs)
                xm = sum(xs)/n3; ym = sum(ys)/n3
                num = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
                dx = sum((x-xm)**2 for x in xs)**0.5
                dy = sum((y-ym)**2 for y in ys)**0.5
                rho = num / max(1e-9, dx*dy)
                # slope (y on x)
                slope = num / max(1e-9, sum((x-xm)**2 for x in xs))
            else:
                rho = 0.0
                slope = 0.0
            dev_pnl_rows.append({"day": day, "product": p, "rho_dev_next_ret": rho, "slope": slope, "n": len(xs)})

    def write(path, rows, fields):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write(OUT_DIR/"dev_dist.csv", dev_rows, ["day","n","dev_mean","dev_sd","dev_p1","dev_p99"])
    write(OUT_DIR/"arb_count.csv", arb_rows, list(arb_rows[0].keys()))
    write(OUT_DIR/"per_leg_spread.csv", spread_rows, list(spread_rows[0].keys()))
    write(OUT_DIR/"dev_vs_pnl.csv", dev_pnl_rows, list(dev_pnl_rows[0].keys()))

    print("=== Per-day dev_XL distribution ===")
    for r in dev_rows:
        print(f"  D{r['day']:>2}: n={r['n']:>5d}  mean={r['dev_mean']:+7.3f}  sd={r['dev_sd']:6.3f}  p1={r['dev_p1']:+6.2f}  p99={r['dev_p99']:+6.2f}")
    print("\n=== Basket arb fire counts ===")
    for r in arb_rows:
        print(f"  D{r['day']:>2}: arb_sell={r['arb_sell_fires']:>5d} ({r['arb_sell_pct']:5.2f}%)  arb_buy={r['arb_buy_fires']:>5d} ({r['arb_buy_pct']:5.2f}%)  max_sum_bb-50K={r['max_sum_bb_minus']:+5.1f}  min_sum_ba-50K={r['min_sum_ba_minus']:+5.1f}")
    print("\n=== Per-leg spread (mean, p1, p99) ===")
    by_prod = defaultdict(list)
    for r in spread_rows:
        by_prod[r["product"]].append(r)
    for p in PEBBLES:
        rs = by_prod[p]
        m = sum(r["spread_mean"] for r in rs) / len(rs)
        print(f"  {p:>12s}: spread_mean={m:5.2f}  per-day p99={[round(r['spread_p99'],1) for r in rs]}")
    print("\n=== Predictive power: corr(dev, next 1-tick mid return) per leg ===")
    by_prod = defaultdict(list)
    for r in dev_pnl_rows:
        by_prod[r["product"]].append(r)
    for p in PEBBLES:
        rs = by_prod[p]
        for r in rs:
            print(f"  {p:>12s} D{r['day']:>2}: rho={r['rho_dev_next_ret']:+7.4f}  slope={r['slope']:+7.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
