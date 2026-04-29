"""How is PISTACHIO related to the two SNACKPACK pairs?

Pairs (from prior analysis):
  pair A:  CHOCOLATE  ↔  VANILLA       (anti-correlated)
  pair B:  STRAWBERRY ↔  RASPBERRY     (anti-correlated)

Open question: where does PISTACHIO sit?
  - Is it a pure twin of STRAWBERRY (so it co-moves with STRAW vs RASP)?
  - Is it a third leg in a 3-way basket (PISTA+STRAW+RASP=const)?
  - Does it leak into the CHOC/VAN factor at all?
  - Or is its noise component independent and tradeable?

Using mid-price returns (diffs), per day 2/3/4.
"""

import csv
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ROUND_5"
DAYS = [2, 3, 4]
SNACK = ["CHOCOLATE", "VANILLA", "PISTACHIO", "STRAWBERRY", "RASPBERRY"]


def read_mids(day):
    path = DATA_DIR / f"prices_round_5_day_{day}.csv"
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                m = float(row["mid_price"])
            except (ValueError, TypeError):
                continue
            ts = int(row["timestamp"])
            rows.setdefault(row["product"], {})[ts] = m
    all_ts = sorted({t for v in rows.values() for t in v})
    out = {}
    for p, mp in rows.items():
        out[p] = np.array([mp.get(t, np.nan) for t in all_ts])
    return out


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return beta, 1 - ss_res / ss_tot


def main():
    for d in DAYS:
        mids = read_mids(d)
        snack = {f: mids[f"SNACKPACK_{f}"] for f in SNACK}
        ret = {f: np.diff(snack[f]) for f in SNACK}

        print(f"\n========== DAY {d} ==========")

        # 1) Full 5x5 return correlation
        print("\n-- 5x5 return correlation --")
        names = SNACK
        mat = np.corrcoef(np.stack([ret[f] for f in names]))
        header = "         " + " ".join(f"{n[:5]:>7}" for n in names)
        print(header)
        for i, n in enumerate(names):
            line = f"{n[:8]:8s} " + " ".join(f"{mat[i,j]:+7.3f}" for j in range(5))
            print(line)

        # 2) End-of-day drift levels (sums to detect basket invariants)
        print("\n-- per-flavor start/end levels --")
        for f in SNACK:
            print(f"  {f:11s}: start={snack[f][0]:7.1f}  end={snack[f][-1]:7.1f}  drift={snack[f][-1]-snack[f][0]:+7.1f}")
        sums = {
            "CHOC+VAN":  snack["CHOCOLATE"]+snack["VANILLA"],
            "STRAW+RASP": snack["STRAWBERRY"]+snack["RASPBERRY"],
            "PISTA+STRAW+RASP": snack["PISTACHIO"]+snack["STRAWBERRY"]+snack["RASPBERRY"],
            "PISTA+RASP":  snack["PISTACHIO"]+snack["RASPBERRY"],
            "ALL5":       sum(snack.values()),
        }
        print("\n-- candidate basket sums (mean ± std) --")
        for k, v in sums.items():
            print(f"  {k:22s}: {v.mean():9.2f} ± {v.std():7.2f}   range [{v.min():.0f}, {v.max():.0f}]")

        # 3) PISTACHIO regressed on the two factor spreads
        # factor_A = CHOC - VAN  (anti-pair A's "direction")
        # factor_B = STRAW - RASP  (anti-pair B's "direction")
        fA = ret["CHOCOLATE"] - ret["VANILLA"]
        fB = ret["STRAWBERRY"] - ret["RASPBERRY"]
        y  = ret["PISTACHIO"]
        X  = np.column_stack([np.ones_like(y), fA, fB])
        beta, r2 = ols(y, X)
        print(f"\n-- PISTACHIO_ret ~ a + b*(CHOC-VAN) + c*(STRAW-RASP) --")
        print(f"   intercept={beta[0]:+.4f}  b(A)={beta[1]:+.4f}  c(B)={beta[2]:+.4f}   R2={r2:.4f}")

        # 4) PISTACHIO regressed on each individual SNACK return
        print("\n-- PISTACHIO_ret ~ each single flavor (univariate) --")
        for f in ["CHOCOLATE", "VANILLA", "STRAWBERRY", "RASPBERRY"]:
            X1 = np.column_stack([np.ones_like(ret[f]), ret[f]])
            b, r = ols(ret["PISTACHIO"], X1)
            print(f"   {f:11s}: slope={b[1]:+.4f}  R2={r:.4f}")

        # 5) Full multi-regression: PISTA ~ all 4 others
        X4 = np.column_stack([np.ones_like(y),
                              ret["CHOCOLATE"], ret["VANILLA"],
                              ret["STRAWBERRY"], ret["RASPBERRY"]])
        b4, r4 = ols(y, X4)
        print(f"\n-- PISTACHIO_ret ~ all 4 others --")
        print(f"   intercept={b4[0]:+.4f}  CHOC={b4[1]:+.3f} VAN={b4[2]:+.3f} "
              f"STRAW={b4[3]:+.3f} RASP={b4[4]:+.3f}   R2={r4:.4f}")

        # 6) Residual: PISTACHIO once STRAWBERRY explained — is leftover idio or hits CHOC/VAN?
        Xs = np.column_stack([np.ones_like(y), ret["STRAWBERRY"]])
        bs, _ = ols(y, Xs)
        resid = y - Xs @ bs
        # regress residual on CHOC, VAN
        Xr = np.column_stack([np.ones_like(resid), ret["CHOCOLATE"], ret["VANILLA"]])
        br, rr = ols(resid, Xr)
        print(f"\n-- residual(PISTA after STRAW) ~ CHOC + VAN --")
        print(f"   CHOC={br[1]:+.4f} VAN={br[2]:+.4f}  R2={rr:.4f}")
        # also check residual vs STRAW+RASP sum (level convergence)
        Xr2 = np.column_stack([np.ones_like(resid), ret["STRAWBERRY"]+ret["RASPBERRY"]])
        br2, rr2 = ols(resid, Xr2)
        print(f"   residual ~ (STRAW+RASP):  slope={br2[1]:+.4f}  R2={rr2:.4f}")


if __name__ == "__main__":
    main()
