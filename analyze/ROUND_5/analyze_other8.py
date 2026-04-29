"""Per-product pattern analysis for the 8 non-basket categories.

Inputs: data/ROUND_5/prices_round_5_day_{2,3,4}.csv + trades CSVs
Outputs: data/ROUND_5/other8/
  per_product.csv          — one row per (day, product) with all features
  category_summary.csv     — aggregated mean/range per category
  spread_hist.png          — spread distributions
  mr_vs_trend.png          — variance ratio + autocorr scatter
  drift_per_day.png        — drift bar chart by product
  ofi_predict.png          — OFI lead/lag returns scatter
  summary.txt              — text report w/ MM_QTY recommendation per product

Goal: surface which of the 40 "other" products deserve differentiated
treatment beyond the uniform MM_QTY=10 pull-to-zero baseline used by
round_5_combined_v1.py.
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ROUND_5"
OUT_DIR = DATA_DIR / "other8"
DAYS = [2, 3, 4]

CATEGORIES = {
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
                      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
                      "GALAXY_SOUNDS_SOLAR_FLAMES"],
    "SLEEP_POD":    ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
                     "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
    "MICROCHIP":    ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
                     "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"],
    "ROBOT":        ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
                     "ROBOT_LAUNDRY", "ROBOT_IRONING"],
    "UV_VISOR":     ["UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
                     "UV_VISOR_RED", "UV_VISOR_MAGENTA"],
    "TRANSLATOR":   ["TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
                     "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
                     "TRANSLATOR_VOID_BLUE"],
    "PANEL":        ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
    "OXYGEN_SHAKE": ["OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
                     "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"],
}
ALL_PRODUCTS = [p for ms in CATEGORIES.values() for p in ms]
CAT_OF = {p: cat for cat, ms in CATEGORIES.items() for p in ms}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_day(day):
    """Returns dict[product] -> dict with arrays: ts, mid, bb, ba, bv, av."""
    path = DATA_DIR / f"prices_round_5_day_{day}.csv"
    raw = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter=";")
        for row in rd:
            p = row["product"]
            if p not in CAT_OF:
                continue
            try:
                ts = int(row["timestamp"])
                bb = float(row["bid_price_1"])
                ba = float(row["ask_price_1"])
                bv = float(row["bid_volume_1"])
                av = float(row["ask_volume_1"])
                m  = float(row["mid_price"])
            except (ValueError, TypeError, KeyError):
                continue
            raw[p]["ts"].append(ts)
            raw[p]["mid"].append(m)
            raw[p]["bb"].append(bb)
            raw[p]["ba"].append(ba)
            raw[p]["bv"].append(bv)
            raw[p]["av"].append(av)
    out = {}
    for p, d in raw.items():
        order = np.argsort(d["ts"])
        out[p] = {k: np.asarray(v, dtype=float)[order] for k, v in d.items()}
    return out


def load_trades(day):
    """dict[product] -> list[(ts, price, qty)]."""
    path = DATA_DIR / f"trades_round_5_day_{day}.csv"
    out = defaultdict(list)
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter=";")
        for row in rd:
            sym = row["symbol"]
            if sym not in CAT_OF:
                continue
            try:
                out[sym].append((int(row["timestamp"]),
                                 float(row["price"]),
                                 int(row["quantity"])))
            except (ValueError, TypeError):
                continue
    return out


# ---------------------------------------------------------------------------
# Per-product features
# ---------------------------------------------------------------------------
def variance_ratio(returns, k):
    """Lo-MacKinlay variance ratio: var(sum k returns) / (k * var(1 return)).
    VR > 1 → trending, VR < 1 → mean-reverting, VR ≈ 1 → random walk.
    """
    r = returns[~np.isnan(returns)]
    if len(r) < 5 * k:
        return np.nan
    var1 = np.var(r, ddof=1)
    if var1 == 0:
        return np.nan
    n = (len(r) // k) * k
    rk = r[:n].reshape(-1, k).sum(axis=1)
    vark = np.var(rk, ddof=1)
    return vark / (k * var1)


def lag_autocorr(returns, lag=1):
    r = returns[~np.isnan(returns)]
    if len(r) < lag + 5:
        return np.nan
    a = r[:-lag] - r[:-lag].mean()
    b = r[lag:] - r[lag:].mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den == 0:
        return np.nan
    return float(np.sum(a * b) / den)


def ofi_predict(books, horizon=10):
    """Correlation between book imbalance now and forward return over `horizon` ticks."""
    bv = books["bv"]; av = books["av"]; mid = books["mid"]
    imb = (bv - av) / np.maximum(bv + av, 1.0)
    if len(mid) <= horizon + 5:
        return np.nan
    fwd = mid[horizon:] - mid[:-horizon]
    imb = imb[:-horizon]
    if np.std(imb) == 0 or np.std(fwd) == 0:
        return np.nan
    return float(np.corrcoef(imb, fwd)[0, 1])


def features(books, trades_for_prod):
    bb = books["bb"]; ba = books["ba"]; mid = books["mid"]
    spread = ba - bb
    rets = np.diff(mid)

    bv = books["bv"]; av = books["av"]
    imb = (bv - av) / np.maximum(bv + av, 1.0)

    feat = {
        "n_ticks":      int(len(mid)),
        "mid_start":    float(mid[0]) if len(mid) else np.nan,
        "mid_end":      float(mid[-1]) if len(mid) else np.nan,
        "drift":        float(mid[-1] - mid[0]) if len(mid) else np.nan,
        "max_excursion":float(mid.max() - mid.min()) if len(mid) else np.nan,
        "spread_mean":  float(spread.mean()),
        "spread_med":   float(np.median(spread)),
        "spread_p90":   float(np.percentile(spread, 90)),
        "ret_std":      float(np.std(rets)) if len(rets) else np.nan,
        "ret_zero_pct": float(np.mean(rets == 0)) if len(rets) else np.nan,
        "ret_ac1":      lag_autocorr(rets, 1),
        "ret_ac5":      lag_autocorr(rets, 5),
        "vr_5":         variance_ratio(rets, 5),
        "vr_20":        variance_ratio(rets, 20),
        "vr_100":       variance_ratio(rets, 100),
        "imb_mean":     float(imb.mean()),
        "imb_std":      float(imb.std()),
        "ofi_pred_h10": ofi_predict(books, 10),
        "ofi_pred_h50": ofi_predict(books, 50),
        "n_trades":     int(len(trades_for_prod)),
        "trade_qty":    int(sum(t[2] for t in trades_for_prod)),
    }
    # MM-edge proxy: spread captured each round-trip if quoting at bb+1/ba-1
    feat["mm_edge_per_rt"] = max(feat["spread_mean"] - 2, 0.0)
    return feat


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_per_product_csv(rows, out_path):
    cols = ["day", "category", "product"] + list(rows[0].keys() - {"day", "category", "product"})
    cols = ["day", "category", "product",
            "n_ticks", "mid_start", "mid_end", "drift", "max_excursion",
            "spread_mean", "spread_med", "spread_p90", "mm_edge_per_rt",
            "ret_std", "ret_zero_pct", "ret_ac1", "ret_ac5",
            "vr_5", "vr_20", "vr_100",
            "imb_mean", "imb_std", "ofi_pred_h10", "ofi_pred_h50",
            "n_trades", "trade_qty"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def write_category_summary(rows, out_path):
    by_pc = defaultdict(list)
    for r in rows:
        by_pc[(r["category"], r["product"])].append(r)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "product",
                    "spread_mean_avg", "mm_edge_avg",
                    "drift_d2", "drift_d3", "drift_d4",
                    "drift_abs_avg", "ret_ac1_avg", "vr_20_avg", "vr_100_avg",
                    "ofi_h50_avg", "n_trades_avg"])
        for cat in CATEGORIES:
            for p in CATEGORIES[cat]:
                rs = by_pc.get((cat, p), [])
                if not rs: continue
                drift_by_day = {r["day"]: r["drift"] for r in rs}
                w.writerow([
                    cat, p,
                    f"{np.mean([r['spread_mean'] for r in rs]):.2f}",
                    f"{np.mean([r['mm_edge_per_rt'] for r in rs]):.2f}",
                    f"{drift_by_day.get(2, ''):+.0f}" if drift_by_day.get(2) is not None else "",
                    f"{drift_by_day.get(3, ''):+.0f}" if drift_by_day.get(3) is not None else "",
                    f"{drift_by_day.get(4, ''):+.0f}" if drift_by_day.get(4) is not None else "",
                    f"{np.mean([abs(r['drift']) for r in rs]):.0f}",
                    f"{np.nanmean([r['ret_ac1'] for r in rs]):+.3f}",
                    f"{np.nanmean([r['vr_20'] for r in rs]):.3f}",
                    f"{np.nanmean([r['vr_100'] for r in rs]):.3f}",
                    f"{np.nanmean([r['ofi_pred_h50'] for r in rs]):+.3f}",
                    f"{np.mean([r['n_trades'] for r in rs]):.0f}",
                ])
            w.writerow([])  # blank between categories


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_spread_hist(rows, out_path):
    """One subplot per category, 5 spread histograms each (one per product, day-avg)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    by_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat[r["category"]][r["product"]].append(r["spread_mean"])
    for ax, (cat, prods) in zip(axes, sorted(by_cat.items())):
        names, vals = zip(*prods.items())
        names = [n.replace(cat + "_", "") for n in names]
        vals = [np.mean(v) for v in vals]
        ax.bar(range(len(names)), vals, color="steelblue")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{cat} — avg spread", fontsize=10)
        ax.set_ylabel("ticks")
        ax.axhline(2, color="red", lw=0.6, ls="--", label="MM no-edge floor")
        ax.legend(fontsize=7)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=7)
    fig.suptitle("Average bid-ask spread by product (day-mean)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_mr_vs_trend(rows, out_path):
    """Scatter ret_ac1 (x) vs vr_20 (y); each product = 3 dots colored by day."""
    fig, ax = plt.subplots(figsize=(13, 9))
    cats = list(CATEGORIES)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    cat2c = {c: cmap[i] for i, c in enumerate(cats)}
    by_prod = defaultdict(list)
    for r in rows:
        by_prod[r["product"]].append(r)
    for prod, rs in by_prod.items():
        cat = CAT_OF[prod]
        ac1 = np.nanmean([r["ret_ac1"] for r in rs])
        vr  = np.nanmean([r["vr_20"] for r in rs])
        ax.scatter(ac1, vr, c=[cat2c[cat]], s=80, alpha=0.85, edgecolors="black")
        ax.annotate(prod.replace(cat + "_", "")[:8], (ac1, vr),
                    fontsize=7, alpha=0.8)
    ax.axhline(1, color="black", lw=0.6, ls="--", label="VR=1 random walk")
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("lag-1 autocorrelation of returns (avg of days)")
    ax.set_ylabel("variance ratio at k=20")
    ax.set_title("Mean reversion (left/down) vs trending (right/up)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=cat2c[c], label=c, markersize=8)
               for c in cats]
    ax.legend(handles=handles, fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_drift_per_day(rows, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    by_cat = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_cat[r["category"]][r["product"]][r["day"]] = r["drift"]
    for ax, cat in zip(axes, sorted(by_cat)):
        prods = list(by_cat[cat])
        x = np.arange(len(prods))
        w = 0.27
        for i, d in enumerate(DAYS):
            vals = [by_cat[cat][p].get(d, 0) for p in prods]
            ax.bar(x + (i - 1) * w, vals, w, label=f"day {d}")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace(cat + "_", "") for p in prods], rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{cat}", fontsize=10)
        ax.set_ylabel("end_mid − start_mid")
        ax.legend(fontsize=8)
    fig.suptitle("Per-day drift by product (sign volatility = day-asymmetry)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_ofi_predictive(rows, out_path):
    """Per-product OFI predictive value at h=10 vs h=50."""
    fig, ax = plt.subplots(figsize=(13, 9))
    cats = list(CATEGORIES)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    cat2c = {c: cmap[i] for i, c in enumerate(cats)}
    by_prod = defaultdict(list)
    for r in rows:
        by_prod[r["product"]].append(r)
    for prod, rs in by_prod.items():
        cat = CAT_OF[prod]
        h10 = np.nanmean([r["ofi_pred_h10"] for r in rs])
        h50 = np.nanmean([r["ofi_pred_h50"] for r in rs])
        ax.scatter(h10, h50, c=[cat2c[cat]], s=80, alpha=0.85, edgecolors="black")
        ax.annotate(prod.replace(cat + "_", "")[:8], (h10, h50), fontsize=7)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("ρ(book imbalance, fwd return over 10 ticks)")
    ax.set_ylabel("ρ(book imbalance, fwd return over 50 ticks)")
    ax.set_title("OFI predictive value by horizon")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=cat2c[c], label=c, markersize=8)
               for c in cats]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def write_text_summary(rows, out_path):
    by_pc = defaultdict(list)
    for r in rows:
        by_pc[(r["category"], r["product"])].append(r)

    lines = ["=== Round 5 — patterns for 8 non-basket categories ===\n",
             f"40 products × {len(DAYS)} days = {len(rows)} rows\n\n"]

    # ---- Spread ranking
    lines.append("--- Spread (avg over days), highest first ---\n")
    spread_rank = sorted(
        ((np.mean([r["spread_mean"] for r in rs]), p, c)
         for (c, p), rs in by_pc.items()),
        reverse=True)
    for s, p, c in spread_rank:
        edge = max(s - 2, 0.0)
        lines.append(f"  spread={s:5.2f}  mm_edge≈{edge:5.2f}/RT  {c:14s} {p}\n")

    # ---- MR vs trend ranking
    lines.append("\n--- Lag-1 autocorrelation of returns (most-MR first) ---\n")
    ac_rank = sorted(
        ((np.nanmean([r["ret_ac1"] for r in rs]), p, c)
         for (c, p), rs in by_pc.items()))
    for ac, p, c in ac_rank:
        verdict = "MR (good for passive MM)" if ac < -0.05 else \
                  "TREND (avoid pull-to-zero)" if ac > 0.05 else "random walk"
        lines.append(f"  ac1={ac:+.3f}  {verdict:30s}  {c:14s} {p}\n")

    lines.append("\n--- Variance ratio k=20 (smaller = more MR) ---\n")
    vr_rank = sorted(
        ((np.nanmean([r["vr_20"] for r in rs]), p, c)
         for (c, p), rs in by_pc.items()))
    for vr, p, c in vr_rank:
        lines.append(f"  vr20={vr:.3f}  {c:14s} {p}\n")

    # ---- Drift consistency: products with same-sign drifts on multiple days
    lines.append("\n--- Drift sign consistency ---\n")
    for (c, p), rs in sorted(by_pc.items()):
        drifts = [r["drift"] for r in rs]
        signs = ["+" if d > 0 else "-" if d < 0 else "0" for d in drifts]
        consistency = max(signs.count("+"), signs.count("-")) / len(signs)
        lines.append(f"  {''.join(signs)}  cons={consistency:.2f}  drifts={drifts}  {c:14s} {p}\n")

    # ---- OFI rank
    lines.append("\n--- |OFI| vs h=50 forward return (most-predictive first) ---\n")
    ofi_rank = sorted(
        ((np.nanmean([r["ofi_pred_h50"] for r in rs]), p, c)
         for (c, p), rs in by_pc.items()),
        key=lambda t: -abs(t[0] if not np.isnan(t[0]) else 0))
    for o, p, c in ofi_rank[:15]:
        lines.append(f"  ofi_h50={o:+.3f}  {c:14s} {p}\n")

    # ---- Recommendations
    lines.append("\n=== Heuristic per-product MM_QTY recommendation ===\n"
                 "Heuristic: rank by mm_edge × MR-strength × volume\n"
                 " score = max(spread-2, 0) × max(0.05 - ac1, 0) × log(1+n_trades)\n\n")
    scored = []
    for (c, p), rs in by_pc.items():
        sp = np.mean([r["spread_mean"] for r in rs])
        ac = np.nanmean([r["ret_ac1"] for r in rs])
        nt = np.mean([r["n_trades"] for r in rs])
        score = max(sp - 2, 0) * max(0.05 - (ac if not np.isnan(ac) else 0), 0) * np.log1p(nt)
        scored.append((score, sp, ac, nt, p, c))
    scored.sort(reverse=True)
    lines.append("rank score   spread  ac1    trades  category       product\n")
    for i, (s, sp, ac, nt, p, c) in enumerate(scored, 1):
        lines.append(f" {i:>3} {s:5.2f}  {sp:5.2f}  {ac:+.3f}  {nt:6.0f}  {c:14s} {p}\n")

    with open(out_path, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for d in DAYS:
        print(f"loading day {d}...")
        books = load_day(d)
        trades = load_trades(d)
        for p in ALL_PRODUCTS:
            if p not in books:
                continue
            f = features(books[p], trades.get(p, []))
            f["day"] = d
            f["product"] = p
            f["category"] = CAT_OF[p]
            rows.append(f)
    print(f"computed {len(rows)} per-product-day rows")

    write_per_product_csv(rows, OUT_DIR / "per_product.csv")
    write_category_summary(rows, OUT_DIR / "category_summary.csv")
    write_text_summary(rows, OUT_DIR / "summary.txt")
    plot_spread_hist(rows, OUT_DIR / "spread_avg.png")
    plot_mr_vs_trend(rows, OUT_DIR / "mr_vs_trend.png")
    plot_drift_per_day(rows, OUT_DIR / "drift_per_day.png")
    plot_ofi_predictive(rows, OUT_DIR / "ofi_predict.png")

    print(f"\nDone. Outputs in {OUT_DIR}")
    print("Files:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
