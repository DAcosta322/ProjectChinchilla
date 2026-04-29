"""Diagnostic: book imbalance (bv-av)/(bv+av) per leg vs forward returns.
If imbalance has predictive sign on next-tick mid moves, it's a candidate
parameter-free MM gating signal.
"""
import csv
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/ROUND_5")
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
DAYS = [2, 3, 4]


def load_day(day: int):
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
                bv = int(float(r["bid_volume_1"]))
                av = int(float(r["ask_volume_1"]))
            except Exception:
                continue
            rows[ts][prod] = (bb, ba, bv, av, (bb + ba) / 2.0)
    return rows


def main():
    print("=== Per-leg book imbalance vs forward 1-tick mid return ===\n")
    print(f"{'Product':>14s} {'Day':>4s} {'rho(imb,ret_+1)':>18s} {'slope':>10s} {'rho(imb,ret_+5)':>18s}")

    for p in PEBBLES:
        for day in DAYS:
            snaps = load_day(day)
            ts_sorted = sorted(snaps.keys())
            seq = []
            for ts in ts_sorted:
                if p in snaps[ts]:
                    seq.append(snaps[ts][p])
                else:
                    seq.append(None)
            xs1, ys1 = [], []  # imb, return next 1
            xs5, ys5 = [], []  # imb, return next 5

            for i in range(len(seq) - 5):
                if seq[i] is None or seq[i+1] is None or seq[i+5] is None:
                    continue
                bb, ba, bv, av, mid = seq[i]
                if bv + av <= 0:
                    continue
                imb = (bv - av) / (bv + av)
                ret1 = seq[i+1][4] - mid
                ret5 = seq[i+5][4] - mid
                xs1.append(imb); ys1.append(ret1)
                xs5.append(imb); ys5.append(ret5)

            def corr(xs, ys):
                if len(xs) < 100:
                    return 0.0, 0.0
                n = len(xs)
                xm = sum(xs)/n; ym = sum(ys)/n
                num = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
                dx = sum((x-xm)**2 for x in xs)**0.5
                dy = sum((y-ym)**2 for y in ys)**0.5
                rho = num / max(1e-9, dx*dy)
                slope = num / max(1e-9, sum((x-xm)**2 for x in xs))
                return rho, slope

            rho1, slope1 = corr(xs1, ys1)
            rho5, _ = corr(xs5, ys5)
            print(f"{p:>14s} {day:>4d} {rho1:>+18.4f} {slope1:>+10.4f} {rho5:>+18.4f}")
        print()


if __name__ == "__main__":
    main()
