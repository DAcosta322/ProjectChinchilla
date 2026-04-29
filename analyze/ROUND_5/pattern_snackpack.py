"""Pattern recognition on the 5 SNACKPACK products (days 2/3/4).

Each function tests one hypothesis and prints a compact finding. Run all
of them; surviving signals are candidates for v5.

Hypotheses covered:
  1. Lead-lag         — does any product lead any other?
  2. Book imbalance   — does L1 depth ratio predict short-term returns?
  3. Trade flow       — aggressive buy vs sell imbalance over rolling windows
  4. MR half-life     — actual reversion speed of each factor spread
  5. Intraday windows — PnL of MR strategy by quarter-of-day
  6. Vol clustering   — autocorrelation of |returns|
  7. Conditional corr — does pair correlation change in high-vol regimes?
  8. PISTA residual   — autocorrelation, jump structure
  9. Sum invariants   — which subsets sum to a stable constant?
 10. Spread asymmetry — bid vs ask thickness lean
"""

import csv
from pathlib import Path
import numpy as np
from collections import defaultdict


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ROUND_5"
DAYS = [2, 3, 4]
SNACK = ["CHOCOLATE", "VANILLA", "PISTACHIO", "STRAWBERRY", "RASPBERRY"]
SNACK_FULL = [f"SNACKPACK_{f}" for f in SNACK]


# ---------------------------------------------------------------- IO ----

def load_prices(day: int):
    """Returns dict[product] -> dict with arrays: ts, mid, bb, ba, bv, av, depth."""
    rows = defaultdict(list)
    with open(DATA_DIR / f"prices_round_5_day_{day}.csv") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if not r["product"].startswith("SNACKPACK"): continue
            try: m = float(r["mid_price"])
            except: continue
            try:
                bb = int(float(r["bid_price_1"])); bv = int(float(r["bid_volume_1"]))
                ba = int(float(r["ask_price_1"])); av = int(float(r["ask_volume_1"]))
            except: continue
            rows[r["product"]].append((int(r["timestamp"]), m, bb, ba, bv, av))
    out = {}
    for p, vs in rows.items():
        vs.sort()
        ts = np.array([v[0] for v in vs])
        mid = np.array([v[1] for v in vs])
        bb  = np.array([v[2] for v in vs])
        ba  = np.array([v[3] for v in vs])
        bv  = np.array([v[4] for v in vs])
        av  = np.array([v[5] for v in vs])
        out[p] = {"ts": ts, "mid": mid, "bb": bb, "ba": ba, "bv": bv, "av": av}
    return out


def load_trades(day: int):
    """Returns list of (ts, symbol, price, qty, buyer, seller)."""
    out = []
    path = DATA_DIR / f"trades_round_5_day_{day}.csv"
    if not path.exists(): return out
    with open(path) as f:
        for r in csv.DictReader(f, delimiter=";"):
            if not r["symbol"].startswith("SNACKPACK"): continue
            out.append((int(r["timestamp"]), r["symbol"], float(r["price"]),
                        int(float(r["quantity"])), r.get("buyer",""), r.get("seller","")))
    return out


# ------------------------------------------------------------ HELPERS ----

def _ema_resid_std(x, hl=4000):
    """Replay EMA, return residual std (final state)."""
    alpha = 1 - 0.5**(1/hl)
    ema = x[0]; var = 0.0
    for v in x:
        ema = ema + alpha*(v - ema)
        var = (1 - alpha)*var + alpha*(v - ema)**2
    return var ** 0.5


# ----------------------------------------------------- ANALYSES ---------

def analyze_lead_lag(prices, day):
    """For each pair, compute cross-correlation of returns at lags -5..+5."""
    print(f"\n[1] LEAD-LAG (returns) day {day}")
    # Returns
    R = {p.replace("SNACKPACK_",""): np.diff(prices[p]["mid"]) for p in SNACK_FULL}
    pairs = [("CHOCOLATE","VANILLA"), ("STRAWBERRY","RASPBERRY"),
             ("PISTACHIO","STRAWBERRY"), ("PISTACHIO","RASPBERRY"),
             ("CHOCOLATE","STRAWBERRY"), ("VANILLA","RASPBERRY")]
    for a, b in pairs:
        ra = R[a]; rb = R[b]
        n = min(len(ra), len(rb))
        # corr(ra[t], rb[t+k]) for k = -3,-2,-1,0,1,2,3
        line = f"  {a[:5]:>5} vs {b[:5]:<5}:"
        for k in (-3,-2,-1,0,1,2,3):
            if k >= 0:
                c = np.corrcoef(ra[:n-k], rb[k:n])[0,1]
            else:
                c = np.corrcoef(ra[-k:n], rb[:n+k])[0,1]
            mark = "*" if abs(c) > 0.05 and k != 0 else " "
            line += f"  k={k:+d}:{c:+.3f}{mark}"
        print(line)


def analyze_book_imbalance(prices, day):
    """Does (bv - av)/(bv + av) predict next-tick mid return?"""
    print(f"\n[2] BOOK IMBALANCE -> next-return day {day}")
    for p in SNACK_FULL:
        d = prices[p]
        bv, av, mid = d["bv"], d["av"], d["mid"]
        imb = (bv - av) / np.maximum(bv + av, 1)
        ret_next = np.diff(mid)
        n = min(len(imb)-1, len(ret_next))
        c = np.corrcoef(imb[:n], ret_next[:n])[0,1]
        # also: split by imb sign and report mean future return
        future_5 = (np.roll(mid, -5) - mid)[:n]
        pos_mask = imb[:n] > 0.3
        neg_mask = imb[:n] < -0.3
        if pos_mask.sum() > 100 and neg_mask.sum() > 100:
            r5p = future_5[pos_mask].mean()
            r5n = future_5[neg_mask].mean()
            edge = r5p - r5n
            print(f"  {p[10:]:>11s}: corr(imb, ret1)={c:+.3f}  "
                  f"E[ret5|imb>+0.3]={r5p:+.2f}  E[ret5|imb<-0.3]={r5n:+.2f}  "
                  f"edge={edge:+.2f}")
        else:
            print(f"  {p[10:]:>11s}: corr(imb, ret1)={c:+.3f}  (too few extreme)")


def analyze_trade_flow(prices, trades, day, window_ts=10000):
    """For each product, in rolling windows, aggressive buy vs sell qty.
    Then check if imbalance predicts subsequent return."""
    print(f"\n[3] TRADE FLOW imbalance (window {window_ts} ts) -> subsequent return day {day}")
    for p in SNACK_FULL:
        d = prices[p]
        ts = d["ts"]; mid = d["mid"]
        prod_trades = [t for t in trades if t[1] == p]
        # classify by mid at t
        # Build mid lookup
        mid_lookup = dict(zip(ts, mid))
        # bucket trades by 10K-ts windows
        buckets = defaultdict(lambda: [0, 0])  # [buy_vol, sell_vol]
        for ts_t, sym, px, qty, b, s in prod_trades:
            mid_at = mid_lookup.get(ts_t)
            if mid_at is None:
                # nearest
                idx = np.searchsorted(ts, ts_t)
                if idx >= len(ts): idx = len(ts)-1
                mid_at = mid[idx]
            bucket = ts_t // window_ts
            if px > mid_at:    buckets[bucket][0] += qty
            elif px < mid_at:  buckets[bucket][1] += qty
        # imbalance and next-window return
        sorted_buckets = sorted(buckets.keys())
        imbs = []
        future_rets = []
        for bk in sorted_buckets:
            buy, sell = buckets[bk]
            tot = buy + sell
            if tot < 5: continue
            imb = (buy - sell) / tot
            # mid at end of this bucket vs end of next
            t_end = (bk + 1) * window_ts
            t_far = (bk + 2) * window_ts
            i_end = np.searchsorted(ts, t_end); i_far = np.searchsorted(ts, t_far)
            if i_far >= len(mid): break
            ret = mid[min(i_far, len(mid)-1)] - mid[min(i_end, len(mid)-1)]
            imbs.append(imb); future_rets.append(ret)
        if len(imbs) >= 10:
            c = np.corrcoef(imbs, future_rets)[0,1]
            print(f"  {p[10:]:>11s}: n={len(imbs):3d}  corr(flow_imb, next_window_ret)={c:+.3f}")
        else:
            print(f"  {p[10:]:>11s}: n={len(imbs)} too small")


def analyze_mr_halflife(prices, day):
    """For each factor spread, fit AR(1) and report half-life in ticks."""
    print(f"\n[4] MR HALF-LIFE day {day}")
    F = {
        "CHOC-VAN":   prices["SNACKPACK_CHOCOLATE"]["mid"] - prices["SNACKPACK_VANILLA"]["mid"],
        "STRAW-RASP": prices["SNACKPACK_STRAWBERRY"]["mid"] - prices["SNACKPACK_RASPBERRY"]["mid"],
        "PISTA-S":    prices["SNACKPACK_PISTACHIO"]["mid"] - prices["SNACKPACK_STRAWBERRY"]["mid"],
        "PISTA-0.29(S-R)": (prices["SNACKPACK_PISTACHIO"]["mid"]
                             - 0.29*(prices["SNACKPACK_STRAWBERRY"]["mid"]
                                     - prices["SNACKPACK_RASPBERRY"]["mid"])),
    }
    for name, x in F.items():
        # AR(1) on detrended (slow EMA removed)
        alpha = 1 - 0.5**(1/4000)
        ema = np.zeros_like(x); ema[0] = x[0]
        for i in range(1, len(x)):
            ema[i] = ema[i-1] + alpha*(x[i] - ema[i-1])
        r = x - ema
        phi = np.corrcoef(r[:-1], r[1:])[0,1]
        # half-life: ln(0.5) / ln(phi). Only valid if 0 < phi < 1.
        if 0 < phi < 1:
            hl = np.log(0.5) / np.log(phi)
        else:
            hl = float('inf')
        print(f"  {name:<22s}: AR(1)phi={phi:.4f}  half_life={hl:>8.0f} ticks  resid_std={r.std():.0f}")


def analyze_intraday_windows(prices, day):
    """Split day into 4 quarters; report spread variance + factor MR PnL per quarter."""
    print(f"\n[5] INTRADAY WINDOWS (4 quarters) day {day}")
    n_total = len(prices["SNACKPACK_CHOCOLATE"]["mid"])
    quarter = n_total // 4
    fA = prices["SNACKPACK_CHOCOLATE"]["mid"] - prices["SNACKPACK_VANILLA"]["mid"]
    fB = prices["SNACKPACK_STRAWBERRY"]["mid"] - prices["SNACKPACK_RASPBERRY"]["mid"]
    print(f"  {'Q':<3} {'fA std':>8} {'fA range':>12} {'fB std':>8} {'fB range':>12}")
    for q in range(4):
        s, e = q*quarter, (q+1)*quarter
        a = fA[s:e]; b = fB[s:e]
        print(f"  Q{q+1:<2} {a.std():>8.0f} [{a.min():>5.0f},{a.max():>5.0f}] "
              f"{b.std():>8.0f} [{b.min():>5.0f},{b.max():>5.0f}]")


def analyze_volatility_clustering(prices, day):
    """Autocorrelation of |returns| at various lags."""
    print(f"\n[6] VOLATILITY CLUSTERING (autocorr |return|) day {day}")
    print(f"  {'product':<12} {'lag=1':>7} {'lag=10':>7} {'lag=50':>7} {'lag=200':>8}")
    for p in SNACK_FULL:
        r = np.abs(np.diff(prices[p]["mid"]))
        ac = []
        for lag in (1, 10, 50, 200):
            n = len(r) - lag
            if n <= 0: ac.append(np.nan); continue
            c = np.corrcoef(r[:n], r[lag:lag+n])[0,1]
            ac.append(c)
        print(f"  {p[10:]:<12} {ac[0]:>+.3f}  {ac[1]:>+.3f}  {ac[2]:>+.3f}  {ac[3]:>+.3f}")


def analyze_conditional_corr(prices, day):
    """Pair correlations in high-vol vs low-vol windows."""
    print(f"\n[7] CONDITIONAL CORR (high-vol vs low-vol) day {day}")
    # Compute realized vol of f_A in 1000-tick windows
    fA = prices["SNACKPACK_CHOCOLATE"]["mid"] - prices["SNACKPACK_VANILLA"]["mid"]
    fB = prices["SNACKPACK_STRAWBERRY"]["mid"] - prices["SNACKPACK_RASPBERRY"]["mid"]
    n = len(fA)
    win = 1000
    vols = np.array([fA[i:i+win].std() for i in range(0, n-win, win)])
    # Top 25% vs bottom 25% windows
    if len(vols) < 8:
        print("  too short")
        return
    hi_thresh = np.percentile(vols, 75)
    lo_thresh = np.percentile(vols, 25)
    R = {p: np.diff(prices[p]["mid"]) for p in SNACK_FULL}
    for (a, b) in [("CHOCOLATE","VANILLA"), ("STRAWBERRY","RASPBERRY"),
                   ("PISTACHIO","STRAWBERRY"), ("PISTACHIO","RASPBERRY")]:
        ra = R[f"SNACKPACK_{a}"]; rb = R[f"SNACKPACK_{b}"]
        hi_idx = []; lo_idx = []
        for i, v in enumerate(vols):
            i0, i1 = i*win, min((i+1)*win, len(ra))
            rng = list(range(i0, i1))
            if v >= hi_thresh: hi_idx.extend(rng)
            elif v <= lo_thresh: lo_idx.extend(rng)
        if len(hi_idx) < 100 or len(lo_idx) < 100: continue
        c_hi = np.corrcoef(ra[hi_idx], rb[hi_idx])[0,1]
        c_lo = np.corrcoef(ra[lo_idx], rb[lo_idx])[0,1]
        print(f"  {a[:5]}-{b[:5]:<6}: lo-vol rho={c_lo:+.3f}  hi-vol rho={c_hi:+.3f}  shift={c_hi-c_lo:+.3f}")


def analyze_pista_residual(prices, day):
    """PISTA's residual after factor B: jumps, AR(1), large-deviation reversion."""
    print(f"\n[8] PISTA RESIDUAL (after 0.29*(STRAW-RASP)) day {day}")
    P = prices["SNACKPACK_PISTACHIO"]["mid"]
    S = prices["SNACKPACK_STRAWBERRY"]["mid"]
    R = prices["SNACKPACK_RASPBERRY"]["mid"]
    f_P = P - 0.29*(S - R)
    # detrend
    alpha = 1 - 0.5**(1/4000)
    ema = np.zeros_like(f_P); ema[0] = f_P[0]
    for i in range(1, len(f_P)): ema[i] = ema[i-1] + alpha*(f_P[i] - ema[i-1])
    resid = f_P - ema
    # diff stats
    d = np.diff(resid)
    big = np.abs(d) > 3 * d.std()
    # after big jumps, does it revert?
    rev = []
    for i in range(len(d)):
        if big[i] and i + 50 < len(resid):
            sign = -np.sign(d[i])
            move_50 = resid[i+50] - resid[i+1]
            rev.append(sign * move_50)
    rev = np.array(rev)
    print(f"  resid_std={resid.std():.0f}  diff_std={d.std():.1f}")
    print(f"  jump count (>3sd diff): {big.sum()}  ({100*big.mean():.1f}% of ticks)")
    if len(rev) > 5:
        print(f"  post-jump 50-tick reversion: mean={rev.mean():+.2f}  median={np.median(rev):+.2f}  n={len(rev)}")


def analyze_sum_invariants(prices, day):
    """Test all 2/3/4/5 subsets — find tightest sum constants."""
    print(f"\n[9] SUM INVARIANTS day {day}  (looking for subsets with low std)")
    M = {f: prices[f"SNACKPACK_{f}"]["mid"] for f in SNACK}
    from itertools import combinations
    findings = []
    for k in (2, 3, 4, 5):
        for combo in combinations(SNACK, k):
            s = sum(M[f] for f in combo)
            findings.append((s.std(), s.mean(), combo))
    findings.sort()
    print("  Top 10 tightest sums:")
    for std, mean, combo in findings[:10]:
        cs = "+".join(c[:3] for c in combo)
        print(f"    {cs:<25s}  mean={mean:>9.0f}  std={std:>6.1f}")


def analyze_spread_asymmetry(prices, day):
    """Bid vs ask side L1 thickness lean across the day."""
    print(f"\n[10] BID/ASK ASYMMETRY day {day}")
    print(f"  {'product':<12} {'bv_mean':>8} {'av_mean':>8} {'lean':>6}  (lean = (bv-av)/(bv+av))")
    for p in SNACK_FULL:
        bv = prices[p]["bv"]; av = prices[p]["av"]
        lean = (bv.mean() - av.mean()) / (bv.mean() + av.mean())
        print(f"  {p[10:]:<12} {bv.mean():>8.1f} {av.mean():>8.1f} {lean:>+6.3f}")


# -------------------------------------------------------------------- main

def main():
    for d in DAYS:
        print(f"\n========================  DAY {d}  ========================")
        prices = load_prices(d)
        trades = load_trades(d)
        analyze_lead_lag(prices, d)
        analyze_book_imbalance(prices, d)
        analyze_trade_flow(prices, trades, d)
        analyze_mr_halflife(prices, d)
        analyze_intraday_windows(prices, d)
        analyze_volatility_clustering(prices, d)
        analyze_conditional_corr(prices, d)
        analyze_pista_residual(prices, d)
        analyze_sum_invariants(prices, d)
        analyze_spread_asymmetry(prices, d)


if __name__ == "__main__":
    main()
