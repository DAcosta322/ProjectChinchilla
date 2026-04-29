"""Round 5 directional + compositional patterns for the 40 non-basket products.

Looks for things the per-product analyzer missed:
  D1. Intraday drift profile — averaged across all 40 products, does the
      universe drift up/down vs flat within a day? Per fragment too.
  D2. Per-product within-day path shape — trend, U, hockey-stick, etc.
      Classify each (product, day) by drift sign + monotonicity.
  D3. Day-conditional sign — does day-2 direction predict day-3/4 direction
      per product? (At population level, ρ_drift across days is in memory:
      d2-d3 = +0.06, d2-d4 = +0.16, d3-d4 = -0.31. We confirm and break by
      product.)
  C1. PCA on de-meaned returns of the 40 — how much variance does the top
      latent factor explain? What are the loadings? Hidden factor would
      show as concentrated loadings.
  C2. Cross-category correlation clusters — re-run pairwise corr filtered
      to just the 40, look for clusters above the noise floor.
  C3. Long-horizon (h=100, 1000) cross-correlations — within-tick ρ ≈ 0
      doesn't preclude lagged cross-effects. Compute ρ(r_i(t), r_j(t+h)).

Outputs: data/ROUND_5/other8_directional/
  intraday_profile.png
  day_profile_perprod.png
  path_classification.csv
  drift_sign_persistence.csv
  pca_loadings.csv + pca_loadings.png
  pca_variance.png
  long_horizon_xcorr.csv (top pairs)
  long_horizon_xcorr.png
  summary.txt
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ROUND_5"
OUT_DIR = DATA_DIR / "other8_directional"
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
N = len(ALL_PRODUCTS)


def load_day(day):
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
                m  = float(row["mid_price"])
            except (ValueError, TypeError, KeyError):
                continue
            raw[p]["ts"].append(ts)
            raw[p]["mid"].append(m)
    out = {}
    for p, d in raw.items():
        order = np.argsort(d["ts"])
        out[p] = {
            "ts":  np.asarray(d["ts"])[order],
            "mid": np.asarray(d["mid"], dtype=float)[order],
        }
    return out


def aligned_mid_matrix(books, products):
    """Return (T, len(products)) mid matrix aligned to common ts axis. NaN-fwd-filled."""
    common_ts = None
    for p in products:
        if p not in books:
            continue
        if common_ts is None:
            common_ts = set(books[p]["ts"])
        else:
            common_ts &= set(books[p]["ts"])
    common_ts = np.array(sorted(common_ts))
    M = np.zeros((len(common_ts), len(products)), dtype=float)
    for j, p in enumerate(products):
        if p not in books:
            M[:, j] = np.nan
            continue
        ts2mid = dict(zip(books[p]["ts"], books[p]["mid"]))
        M[:, j] = [ts2mid[t] for t in common_ts]
    return common_ts, M


# ---------------------------------------------------------------------------
# D1. Intraday drift profile (population mean and per-product)
# ---------------------------------------------------------------------------
def intraday_profile(per_day_M, deciles=20):
    """For each day, compute per-product normalized cumulative drift across `deciles`
    buckets. Then average across the 40 products to get the population drift profile."""
    profiles_pop = {}
    profiles_per_prod = {}
    for d, M in per_day_M.items():
        T, P = M.shape
        bucket_edges = np.linspace(0, T - 1, deciles + 1).astype(int)
        # Per-product drift (mid - start) at each decile boundary, normalized by full-day std
        prod_profiles = np.zeros((P, deciles + 1))
        for j in range(P):
            mid = M[:, j]
            sd = np.std(np.diff(mid)) * np.sqrt(T)  # rough full-day-vol scale
            if sd == 0:
                sd = 1.0
            for k, e in enumerate(bucket_edges):
                prod_profiles[j, k] = (mid[e] - mid[0]) / sd
        profiles_per_prod[d] = prod_profiles
        # Population mean (each product equal-weight)
        profiles_pop[d] = prod_profiles.mean(axis=0)
    return profiles_pop, profiles_per_prod


# ---------------------------------------------------------------------------
# D2. Path classification per (product, day)
# ---------------------------------------------------------------------------
def classify_path(mid):
    """Classify mid path. Categories:
      - 'monotone_up' / 'monotone_down': drift dominates path (|drift|/range >= 0.7)
      - 'reverse': path peaks/troughs mid-day, returns (max_excursion >> end_drift)
      - 'noise': low |drift|/range
    """
    drift = mid[-1] - mid[0]
    rng = mid.max() - mid.min()
    if rng == 0:
        return "flat", 0.0
    ratio = drift / rng
    if abs(ratio) >= 0.7:
        return ("monotone_up" if drift > 0 else "monotone_down"), ratio
    elif abs(ratio) <= 0.2 and rng > 200:  # large range, small net drift
        return "reverse", ratio
    else:
        return "noise", ratio


# ---------------------------------------------------------------------------
# C1. PCA on returns
# ---------------------------------------------------------------------------
def pca_returns(per_day_M):
    """Concatenate returns across 3 days, run PCA on de-meaned returns."""
    Rs = []
    for d, M in per_day_M.items():
        # Returns matrix (T-1, 40)
        R = np.diff(M, axis=0)
        # Standardize each column to unit variance for PCA on "shape" not magnitude
        std = R.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        Rs.append((R - R.mean(axis=0)) / std)
    R_all = np.vstack(Rs)
    # Covariance (already standardized so this is correlation matrix)
    C = (R_all.T @ R_all) / (len(R_all) - 1)
    # Symmetric eigendecomp
    evals, evecs = np.linalg.eigh(C)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


# ---------------------------------------------------------------------------
# C3. Long-horizon lagged cross-correlations
# ---------------------------------------------------------------------------
def long_horizon_xcorr(per_day_M, lag):
    """For each ordered pair (i, j), compute mean ρ(r_i(t), r_j(t+lag)) across days.
    r is delta-mid over `lag` ticks."""
    accs = np.zeros((N, N))
    cnts = np.zeros((N, N))
    for d, M in per_day_M.items():
        T = M.shape[0]
        if T <= 2 * lag:
            continue
        # Step returns over `lag` ticks
        step = M[lag:] - M[:-lag]
        # Standardize columns
        s = step.std(axis=0, ddof=1)
        s[s == 0] = 1.0
        Z = (step - step.mean(axis=0)) / s
        # ρ(r_i(t), r_j(t+1)) under lagged step framework — shift by 1 step
        if len(Z) < 5:
            continue
        a = Z[:-1]
        b = Z[1:]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                ai = a[:, i]
                bj = b[:, j]
                if ai.std() == 0 or bj.std() == 0:
                    continue
                accs[i, j] += np.corrcoef(ai, bj)[0, 1]
                cnts[i, j] += 1
    cnts[cnts == 0] = 1
    return accs / cnts


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
CAT_COLORS = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))
CAT2COLOR = {c: CAT_COLORS[i] for i, c in enumerate(CATEGORIES)}


def plot_intraday_profile(profiles_pop, profiles_per_prod, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # Population mean per day
    for d, prof in profiles_pop.items():
        x = np.linspace(0, 100, len(prof))
        axes[0].plot(x, prof, marker="o", label=f"day {d}", linewidth=2)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xlabel("intraday progress (%)")
    axes[0].set_ylabel("mean normalized cumul drift across 40 products")
    axes[0].set_title("Population intraday drift profile")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-product day-3 (overlay)
    d_show = 3
    if d_show in profiles_per_prod:
        prof = profiles_per_prod[d_show]
        x = np.linspace(0, 100, prof.shape[1])
        for j, p in enumerate(ALL_PRODUCTS):
            axes[1].plot(x, prof[j], color=CAT2COLOR[CAT_OF[p]], alpha=0.5, lw=0.8)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_xlabel("intraday progress (%)")
        axes[1].set_ylabel("normalized cumul drift")
        axes[1].set_title(f"Per-product profile, day {d_show}")
        handles = [plt.Line2D([], [], color=CAT2COLOR[c], label=c) for c in CATEGORIES]
        axes[1].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pca_variance(evals, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    pct = 100 * evals / evals.sum()
    cum = np.cumsum(pct)
    ax.bar(range(1, len(pct) + 1), pct, color="steelblue", label="per-component %")
    ax2 = ax.twinx()
    ax2.plot(range(1, len(cum) + 1), cum, color="red", marker="o", lw=1.5, label="cumulative %")
    ax.set_xlabel("PC index")
    ax.set_ylabel("variance explained (%)")
    ax2.set_ylabel("cumulative variance (%)")
    ax.set_title(f"PCA on standardized returns — top PC = {pct[0]:.1f}% (random walk floor ~{100/N:.1f}%)")
    ax.axhline(100 / N, color="green", lw=0.8, ls="--", label="iid floor")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pca_loadings(evecs, out_path, n_components=4):
    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 8))
    for k in range(n_components):
        ax = axes[k]
        loadings = evecs[:, k]
        # Color by category
        colors = [CAT2COLOR[CAT_OF[p]] for p in ALL_PRODUCTS]
        ax.barh(range(N), loadings, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(N))
        labels = [p.replace(CAT_OF[p] + "_", "")[:14] for p in ALL_PRODUCTS]
        if k == 0:
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("loading")
        ax.set_title(f"PC{k+1}")
        ax.axvline(0, color="black", lw=0.5)
        ax.grid(True, alpha=0.3, axis="x")
        # Category boundaries (visual)
        pos = 0
        for cat, members in CATEGORIES.items():
            pos += len(members)
            if pos < N:
                ax.axhline(pos - 0.5, color="black", lw=0.4, alpha=0.5)
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=CAT2COLOR[c], label=c, markersize=10)
               for c in CATEGORIES]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_long_horizon_xcorr(xcorr_h100, xcorr_h1000, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, mat, h in zip(axes, [xcorr_h100, xcorr_h1000], [100, 1000]):
        # Clip extreme values for visual
        v = np.percentile(np.abs(mat[~np.eye(N, dtype=bool)]), 99)
        v = max(v, 0.05)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
        # Category boundaries
        pos = 0
        for cat, members in CATEGORIES.items():
            pos += len(members)
            if pos < N:
                ax.axhline(pos - 0.5, color="black", lw=0.4)
                ax.axvline(pos - 0.5, color="black", lw=0.4)
        labels = [p.replace(CAT_OF[p] + "_", "")[:8] for p in ALL_PRODUCTS]
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_title(f"ρ(r_i(t), r_j(t+1)) at lag={h} ticks")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Lagged cross-correlation — leader (rows) → follower (cols)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("loading per-day mids ...")
    per_day_books = {d: load_day(d) for d in DAYS}
    per_day_M = {}
    for d in DAYS:
        ts, M = aligned_mid_matrix(per_day_books[d], ALL_PRODUCTS)
        per_day_M[d] = M
        print(f"  day {d}: T={len(ts)}, products={M.shape[1]}")

    # ---- D1: intraday profile ----
    print("\nD1 intraday profile ...")
    profiles_pop, profiles_per_prod = intraday_profile(per_day_M, deciles=20)
    plot_intraday_profile(profiles_pop, profiles_per_prod, OUT_DIR / "intraday_profile.png")

    # ---- D2: path classification ----
    print("D2 path classification ...")
    path_rows = []
    for d in DAYS:
        M = per_day_M[d]
        for j, p in enumerate(ALL_PRODUCTS):
            mid = M[:, j]
            cls, ratio = classify_path(mid)
            path_rows.append({
                "day": d,
                "category": CAT_OF[p],
                "product": p,
                "drift": mid[-1] - mid[0],
                "max_excursion": mid.max() - mid.min(),
                "drift_over_range": ratio,
                "class": cls,
            })
    with open(OUT_DIR / "path_classification.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "category", "product", "drift", "max_excursion",
                    "drift_over_range", "class"])
        for r in path_rows:
            w.writerow([r["day"], r["category"], r["product"],
                        f"{r['drift']:+.0f}", f"{r['max_excursion']:.0f}",
                        f"{r['drift_over_range']:+.3f}", r["class"]])
    # Class frequency table
    class_freq = defaultdict(lambda: defaultdict(int))
    for r in path_rows:
        class_freq[r["category"]][r["class"]] += 1

    # ---- D3: drift sign persistence per product ----
    print("D3 drift sign persistence ...")
    drift_by_pd = defaultdict(dict)
    for r in path_rows:
        drift_by_pd[r["product"]][r["day"]] = r["drift"]
    sign_rows = []
    for p in ALL_PRODUCTS:
        signs = ["+" if drift_by_pd[p][d] > 0 else "-" for d in DAYS]
        cons = max(signs.count("+"), signs.count("-")) / len(signs)
        # Transitions
        flips = sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
        sign_rows.append((p, "".join(signs), cons, flips))
    with open(OUT_DIR / "drift_sign_persistence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "signs_d2_d3_d4", "consistency", "flips"])
        for p, s, c, fl in sign_rows:
            w.writerow([p, s, f"{c:.2f}", fl])

    # ---- C1: PCA ----
    print("C1 PCA on returns ...")
    evals, evecs = pca_returns(per_day_M)
    plot_pca_variance(evals, OUT_DIR / "pca_variance.png")
    plot_pca_loadings(evecs, OUT_DIR / "pca_loadings.png", n_components=4)
    with open(OUT_DIR / "pca_loadings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "category"] + [f"PC{k+1}" for k in range(8)])
        for j, p in enumerate(ALL_PRODUCTS):
            w.writerow([p, CAT_OF[p]] + [f"{evecs[j, k]:+.4f}" for k in range(8)])

    # ---- C3: lagged cross-correlation ----
    print("C3 long-horizon lagged xcorr ...")
    xcorr_h100  = long_horizon_xcorr(per_day_M, lag=100)
    xcorr_h1000 = long_horizon_xcorr(per_day_M, lag=1000)
    plot_long_horizon_xcorr(xcorr_h100, xcorr_h1000, OUT_DIR / "long_horizon_xcorr.png")
    # Top pairs by |ρ|
    top_pairs = []
    for h, mat in [(100, xcorr_h100), (1000, xcorr_h1000)]:
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                top_pairs.append((h, mat[i, j], ALL_PRODUCTS[i], ALL_PRODUCTS[j]))
    top_pairs.sort(key=lambda t: -abs(t[1]))
    with open(OUT_DIR / "long_horizon_xcorr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "rho", "leader", "follower"])
        for h, r, a, b in top_pairs[:50]:
            w.writerow([h, f"{r:+.4f}", a, b])

    # ---- Text summary ----
    print("writing summary ...")
    lines = []
    lines.append("=== Round 5 — directional + compositional patterns (40 non-basket products) ===\n\n")

    lines.append("D1. POPULATION-MEAN INTRADAY PROFILE (averaged across 40 products)\n")
    for d, prof in profiles_pop.items():
        lines.append(f"  day {d}: start={prof[0]:+.3f}  mid={prof[len(prof)//2]:+.3f}  "
                     f"end={prof[-1]:+.3f}  max={prof.max():+.3f}  min={prof.min():+.3f}\n")
    lines.append("  Interpretation: in a fully-random universe each value should hover near 0;\n"
                 "  large persistent excursions = day-level directional regime.\n\n")

    lines.append("D2. PATH SHAPE FREQUENCY (classes per category-day-aggregate)\n")
    classes = ["monotone_up", "monotone_down", "reverse", "noise", "flat"]
    lines.append(f"  {'category':14s}  " + "  ".join(f"{c:14s}" for c in classes) + "\n")
    for cat in CATEGORIES:
        counts = class_freq[cat]
        total = sum(counts.values())
        row = "  " + f"{cat:14s}"
        for c in classes:
            row += f"  {counts.get(c, 0):>3}/{total}={counts.get(c,0)/total*100:>4.0f}%"
        lines.append(row + "\n")
    lines.append("  Interpretation: 'monotone' shows products that drift end-to-end and\n"
                 "  are MR-fight risk; 'reverse' shows products with intraday round-trips\n"
                 "  that benefit MR/passive MM.\n\n")

    lines.append("D3. PER-PRODUCT DAY-SIGN SEQUENCES (d2 d3 d4)\n")
    sign_freq = defaultdict(int)
    for _, s, _, _ in sign_rows:
        sign_freq[s] += 1
    for s in sorted(sign_freq, key=lambda x: -sign_freq[x]):
        lines.append(f"  {s}  count={sign_freq[s]:>2}  ({sign_freq[s]/len(sign_rows)*100:.0f}%)\n")
    lines.append("\n  Per-product (alphabetical, only shows |consistency|=1.00):\n")
    for p, s, c, fl in sorted(sign_rows):
        if c == 1.00:
            lines.append(f"    {s}  {CAT_OF[p]:14s} {p}\n")

    lines.append("\nC1. PCA ON STANDARDIZED RETURNS (40-D, days concatenated)\n")
    pct = 100 * evals / evals.sum()
    iid_floor = 100 / N
    lines.append(f"  iid floor (random walk uniform): {iid_floor:.2f}% per PC\n")
    lines.append(f"  top-1 PC: {pct[0]:.2f}%   ratio over floor = {pct[0]/iid_floor:.2f}×\n")
    lines.append(f"  top-3 PCs: {pct[:3].sum():.2f}%   top-5: {pct[:5].sum():.2f}%   top-10: {pct[:10].sum():.2f}%\n")
    lines.append("  PC1 loadings (top |x|, all-positive or all-negative dominance = level factor):\n")
    pc1_sorted = sorted(zip(evecs[:, 0], ALL_PRODUCTS), key=lambda t: -abs(t[0]))
    for v, p in pc1_sorted[:15]:
        lines.append(f"    {v:+.4f}  {CAT_OF[p]:14s} {p}\n")

    lines.append("\nC3. LAGGED CROSS-CORRELATION TOP 20 (|ρ| largest)\n")
    for h, r, a, b in top_pairs[:20]:
        ca = CAT_OF[a]
        cb = CAT_OF[b]
        same = "(same cat)" if ca == cb else "(cross)"
        lines.append(f"  h={h:>4}  ρ={r:+.4f}  {a} → {b} {same}\n")

    with open(OUT_DIR / "summary.txt", "w") as f:
        f.writelines(lines)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
