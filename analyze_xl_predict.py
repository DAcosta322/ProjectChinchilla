"""Predict PEBBLES_XL from the other 4. Three angles:

1. Contemporaneous regression — already known R2 ≈ 0.98 (basket invariant).
   Verify with explicit fit and show residual scale.

2. Lead-lag — does XL lead or lag the others? Compute corr(XL_ret_t, others_ret_{t+k})
   for k in [-10..+10]. Non-zero peaks at k != 0 = tradeable.

3. Implied-fair vs actual mid — at each tick compute implied_XL = 50000 - sum(others),
   compare to actual mid_XL. Dev distribution shows whether bid/ask
   microstructure ever gives a tradeable mismatch.
"""

import csv
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "ROUND_5"
DAYS = [2, 3, 4]
PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]


def read_mids(day):
    path = DATA_DIR / f"prices_round_5_day_{day}.csv"
    rows = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            try:
                m = float(row["mid_price"])
            except (ValueError, TypeError):
                continue
            rows.setdefault(row["product"], {})[int(row["timestamp"])] = m
    all_ts = sorted({t for v in rows.values() for t in v})
    out = {p: np.array([rows[p].get(t, np.nan) for t in all_ts]) for p in PRODUCTS if p in rows}
    return out, np.array(all_ts)


def per_day_analysis(day):
    print(f"\n=== DAY {day} ===")
    mids, ts = read_mids(day)
    if any(p not in mids for p in PRODUCTS):
        print("missing product"); return

    XL = mids["PEBBLES_XL"]
    others = np.column_stack([mids[p] for p in PRODUCTS if p != "PEBBLES_XL"])
    sum_others = others.sum(axis=1)

    # 1. Contemporaneous regression on LEVELS
    X = np.column_stack([np.ones_like(sum_others), sum_others])
    beta_lvl, *_ = np.linalg.lstsq(X, XL, rcond=None)
    pred_lvl = X @ beta_lvl
    resid_lvl = XL - pred_lvl
    r2_lvl = 1 - (resid_lvl ** 2).sum() / ((XL - XL.mean()) ** 2).sum()
    print(f"LEVEL regression XL ~ a + b*sum_others:")
    print(f"  intercept={beta_lvl[0]:.2f}  slope={beta_lvl[1]:.4f}  R2={r2_lvl:.6f}")
    print(f"  residual std={resid_lvl.std():.3f}  max|resid|={np.abs(resid_lvl).max():.2f}")

    # 2. Contemporaneous regression on RETURNS
    XL_ret = np.diff(XL)
    others_ret = np.diff(others, axis=0)
    sum_ret = others_ret.sum(axis=1)
    X2 = np.column_stack([sum_ret])  # no intercept (returns are 0-mean)
    beta_ret, *_ = np.linalg.lstsq(X2, XL_ret, rcond=None)
    pred_ret = X2 @ beta_ret
    resid_ret = XL_ret - pred_ret
    ss_tot = (XL_ret ** 2).sum()
    ss_res = (resid_ret ** 2).sum()
    r2_ret = 1 - ss_res / ss_tot
    print(f"RETURN regression XL_ret ~ b*sum_others_ret:")
    print(f"  slope={beta_ret[0]:.4f}  R2={r2_ret:.6f}")
    print(f"  residual std={resid_ret.std():.3f}  max|resid|={np.abs(resid_ret).max():.2f}")

    # 3. Lead-lag: corr(XL_ret_t, sum_others_ret_{t+k})
    print(f"LEAD-LAG corr(XL_ret_t, sum_others_ret_{{t+k}}):")
    print(f"  {'k':>4} {'corr':>9} {'n':>8}")
    for k in (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20):
        if k < 0:
            x = XL_ret[-k:]; y = sum_ret[:k]
        elif k > 0:
            x = XL_ret[:-k]; y = sum_ret[k:]
        else:
            x = XL_ret; y = sum_ret
        if len(x) > 10:
            c = np.corrcoef(x, y)[0, 1]
            sign = " ← others lead" if k > 0 else (" ← XL leads" if k < 0 else "")
            print(f"  {k:>+4}  {c:>+8.4f}  {len(x):>8}{sign if abs(c) > 0.05 and k != 0 else ''}")

    # 4. Implied price vs actual — if implied always equals actual, no edge
    # implied = -beta_lvl[0]/beta_lvl[1]... wait simpler form:
    # XL ≈ a + b*sum_others. Solve for "implied XL given others = sum_others".
    # Same as pred_lvl. Already computed resid_lvl.
    print(f"PRICE-LEVEL DEV distribution (XL_actual - implied):")
    pcts = np.percentile(resid_lvl, [1, 5, 50, 95, 99])
    print(f"  p1={pcts[0]:+.2f} p5={pcts[1]:+.2f} p50={pcts[2]:+.2f} p95={pcts[3]:+.2f} p99={pcts[4]:+.2f}")

    # 5. Predicting XL CHANGE from others' CHANGE — what if we use a basket
    # that EXCLUDES XL to compute an "implied next-tick XL"? At t we know
    # others at t, predict XL at t. Already done as r2_ret. Try also k=+1:
    # predict XL_change_t from sum_others_change_{t-1} (others lead)
    XL_ret_t = XL_ret[1:]
    sum_ret_lag = sum_ret[:-1]
    X3 = np.column_stack([sum_ret_lag])
    beta_lag, *_ = np.linalg.lstsq(X3, XL_ret_t, rcond=None)
    pred_lag = X3 @ beta_lag
    ss_res_lag = ((XL_ret_t - pred_lag) ** 2).sum()
    ss_tot_lag = (XL_ret_t ** 2).sum()
    r2_lag = 1 - ss_res_lag / ss_tot_lag
    print(f"PREDICTIVE regression XL_ret_t ~ b*sum_others_ret_{{t-1}}:")
    print(f"  slope={beta_lag[0]:+.4f}  R2={r2_lag:.6f}  (positive R2 = others lead by 1 tick)")


def main():
    print("PEBBLES_XL prediction from other 4 — basket regression analysis")
    for d in DAYS:
        per_day_analysis(d)


if __name__ == "__main__":
    main()
