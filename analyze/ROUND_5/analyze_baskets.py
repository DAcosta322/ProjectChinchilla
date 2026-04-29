"""Verify the basket relationships discovered in correlations:
  PEBBLES_XL  ~ -(XS + S + M + L) ?
  SNACKPACK_VANILLA   ~ -CHOCOLATE ?
  SNACKPACK_RASPBERRY ~ -(PISTACHIO + STRAWBERRY)/2 ?

Fits OLS on returns (and on price levels with intercept) per day, reports R^2.
"""

import csv
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ROUND_5"
DAYS = [2, 3, 4]


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
    """Fit y = X @ beta + e (X already has intercept if needed). Returns beta, R^2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return beta, r2


def fit_basket(name, target, regressors, mids_by_day):
    print(f"\n=== {name} ===")
    print(f"  target:     {target}")
    print(f"  regressors: {regressors}")

    for d in DAYS:
        mids = mids_by_day[d]
        # Level fit (with intercept)
        y = mids[target]
        X = np.column_stack([np.ones_like(y)] + [mids[r] for r in regressors])
        beta, r2 = ols(y, X)
        # Returns fit (no intercept needed for diffs around 0)
        yr = np.diff(y)
        Xr = np.column_stack([np.diff(mids[r]) for r in regressors])
        beta_r, r2_r = ols(yr, Xr)
        coef_str = ", ".join(f"{r}={beta_r[i]:+.3f}" for i, r in enumerate(regressors))
        print(f"  day {d}: level R2={r2:.4f}  intercept={beta[0]:.1f}  "
              f"return-coefs[{coef_str}]  return R2={r2_r:.4f}")


def main():
    mids_by_day = {d: read_mids(d) for d in DAYS}

    # --- PEBBLES ---
    fit_basket("PEBBLES — XL vs basket(XS,S,M,L)",
               "PEBBLES_XL",
               ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"],
               mids_by_day)

    # --- SNACKPACK ---
    fit_basket("SNACKPACK — VANILLA vs CHOCOLATE",
               "SNACKPACK_VANILLA",
               ["SNACKPACK_CHOCOLATE"],
               mids_by_day)

    fit_basket("SNACKPACK — RASPBERRY vs (PISTACHIO, STRAWBERRY)",
               "SNACKPACK_RASPBERRY",
               ["SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"],
               mids_by_day)

    # Double-check the "no structure" claim on a couple other categories:
    fit_basket("SLEEP_POD — COTTON vs other 4 (control: should be near 0)",
               "SLEEP_POD_COTTON",
               ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL",
                "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON"],
               mids_by_day)

    fit_basket("ROBOT — IRONING vs other 4 (control)",
               "ROBOT_IRONING",
               ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_LAUNDRY"],
               mids_by_day)


if __name__ == "__main__":
    main()
