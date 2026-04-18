"""Parameter sweep for trading algorithms.

Usage:
    python param_sweep.py short_warmup_ma
    python param_sweep.py ema_trend
    python param_sweep.py both
    python param_sweep.py osmium
"""

import sys
import importlib.util
import itertools
from pathlib import Path
from typing import Dict, List, Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backtester import DataReader, run_backtest

DATA_DIR = SCRIPT_DIR / "data"
ROUND = 0
DAYS = [-2, -1]

ROUND_1 = 1
DAYS_1 = [-2, -1, 0]

ROUND_2 = 2
DAYS_2 = [-1, 0, 1]


def load_algo(name: str):
    path = SCRIPT_DIR / "algorithms" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("trader_algo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_with_params(algo_name: str, param_overrides: dict) -> Dict[str, float]:
    """Load algo, patch StockTrader class attrs, run both days, return PnL."""
    module = load_algo(algo_name)

    # Patch the StockTrader class with parameter overrides
    stock_cls = module.StockTrader
    for k, v in param_overrides.items():
        setattr(stock_cls, k, v)

    reader = DataReader(DATA_DIR)
    total_pnl = 0.0
    tomato_pnl = 0.0
    emerald_pnl = 0.0

    for day in DAYS:
        result = run_backtest(module, reader, ROUND, day)
        if result:
            total_pnl += result["profit"]
            tomato_pnl += result["pnl_by_product"].get("TOMATOES", 0.0)
            emerald_pnl += result["pnl_by_product"].get("EMERALDS", 0.0)

    return {
        "total": total_pnl,
        "tomato": tomato_pnl,
        "emerald": emerald_pnl,
    }


def sweep_ema_trend():
    print("=" * 70)
    print("SWEEP: ema_trend")
    print("=" * 70)

    param_grid = {
        "FAST_ALPHA": [0.1, 0.15, 0.2, 0.25, 0.35, 0.5],
        "SLOW_ALPHA": [0.005, 0.01, 0.02, 0.04, 0.06],
        "SIGNAL_MULT": [0.5, 1.0, 1.5, 2.0, 3.0],
        "SPREAD": [1, 2, 3],
        "CLEAR_THRESHOLD": [50, 65, 80],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"Total combinations: {len(combos)}")
    print()

    results = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))

        # Skip invalid combos: fast must be more responsive than slow
        if params["FAST_ALPHA"] <= params["SLOW_ALPHA"]:
            continue

        pnl = run_with_params("ema_trend", params)
        results.append((pnl, params))

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(combos)} done")

    results.sort(key=lambda x: x[0]["total"], reverse=True)

    print()
    print("TOP 15 by TOTAL PnL (days -2 + -1):")
    print("-" * 90)
    print(f"{'Rank':>4}  {'Total':>10}  {'Tomato':>10}  {'Emerald':>10}  Parameters")
    print("-" * 90)
    for rank, (pnl, params) in enumerate(results[:15], 1):
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"{rank:>4}  {pnl['total']:>10.2f}  {pnl['tomato']:>10.2f}  {pnl['emerald']:>10.2f}  {param_str}")

    print()
    print("TOP 15 by TOMATO PnL only:")
    print("-" * 90)
    results_tom = sorted(results, key=lambda x: x[0]["tomato"], reverse=True)
    for rank, (pnl, params) in enumerate(results_tom[:15], 1):
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"{rank:>4}  {pnl['total']:>10.2f}  {pnl['tomato']:>10.2f}  {pnl['emerald']:>10.2f}  {param_str}")

    return results


def run_baseline():
    """Run the original tutorial.py as baseline for comparison."""
    print("=" * 70)
    print("BASELINE: tutorial.py (original)")
    print("=" * 70)
    module = load_algo("tutorial")
    reader = DataReader(DATA_DIR)
    total = 0.0
    for day in DAYS:
        result = run_backtest(module, reader, ROUND, day)
        if result:
            total += result["profit"]
            print(f"  Day {day}: total={result['profit']:.2f}  "
                  f"TOMATOES={result['pnl_by_product'].get('TOMATOES', 0):.2f}  "
                  f"EMERALDS={result['pnl_by_product'].get('EMERALDS', 0):.2f}")
    print(f"  TOTAL: {total:.2f}")
    print()


def run_osmium_with_params(param_overrides: dict) -> float:
    """Load round1_osmium, patch OsmiumTrader, run all round 1 days, return osmium PnL."""
    module = load_algo("round1_osmium")
    cls = module.OsmiumTrader
    for k, v in param_overrides.items():
        setattr(cls, k, v)

    reader = DataReader(DATA_DIR)
    total_pnl = 0.0
    for day in DAYS_1:
        result = run_backtest(module, reader, ROUND_1, day)
        if result:
            total_pnl += result["pnl_by_product"].get("ASH_COATED_OSMIUM", 0.0)
    return total_pnl


def sweep_osmium():
    print("=" * 70)
    print("SWEEP: round1_osmium (ASH_COATED_OSMIUM)")
    print("=" * 70)

    # Phase 1: coarse sweep (~65 combos, ~3 min)
    coarse_grid = {
        "FAST_WINDOW": [5, 15],
        "SLOW_WINDOW": [100, 300],
        "SIGNAL_MULT": [0.0, 0.5, 1.0],
        "SPREAD": [3, 7],
        "CLEAR_THRESHOLD": [20, 40],
        "GAMMA": [0.0, 0.1],
    }

    keys = list(coarse_grid.keys())
    combos = list(itertools.product(*[coarse_grid[k] for k in keys]))
    valid_combos = [c for c in combos if dict(zip(keys, c))["FAST_WINDOW"] < dict(zip(keys, c))["SLOW_WINDOW"]]
    print(f"Phase 1 (coarse): {len(valid_combos)} combinations")

    results = []
    for i, values in enumerate(valid_combos):
        params = dict(zip(keys, values))
        pnl = run_osmium_with_params(params)
        results.append((pnl, params))
        if (i + 1) % 100 == 0:
            best = max(r[0] for r in results)
            print(f"  ... {i + 1}/{len(valid_combos)} done (best: {best:.2f})")

    results.sort(key=lambda x: x[0], reverse=True)
    best_params = results[0][1]

    print(f"\nPhase 1 best: PnL={results[0][0]:.2f}")
    print(f"  {best_params}")

    # Phase 2: refine the 4 most sensitive params around best (~110 combos, ~5 min)
    # Fix SLOW_WINDOW and CLEAR_THRESHOLD from phase 1 (less sensitive)
    def refine_values(center, candidates):
        """Pick 3 values closest to center from candidates."""
        candidates = sorted(set(candidates))
        idx = min(range(len(candidates)), key=lambda i: abs(candidates[i] - center))
        lo = max(0, idx - 1)
        hi = min(len(candidates), idx + 2)
        return candidates[lo:hi]

    fine_grid = {
        "FAST_WINDOW": refine_values(best_params["FAST_WINDOW"], [3, 5, 8, 10, 12, 15, 20]),
        "SLOW_WINDOW": [best_params["SLOW_WINDOW"]],  # fixed
        "SIGNAL_MULT": refine_values(best_params["SIGNAL_MULT"], [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.2]),
        "SPREAD": refine_values(best_params["SPREAD"], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "CLEAR_THRESHOLD": [best_params["CLEAR_THRESHOLD"]],  # fixed
        "GAMMA": refine_values(best_params["GAMMA"], [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]),
    }

    combos2 = list(itertools.product(*[fine_grid[k] for k in keys]))
    valid_combos2 = [c for c in combos2 if dict(zip(keys, c))["FAST_WINDOW"] < dict(zip(keys, c))["SLOW_WINDOW"]]
    print(f"\nPhase 2 (fine): {len(valid_combos2)} combinations around best")
    print(f"  Grid: {{{', '.join(f'{k}: {fine_grid[k]}' for k in keys)}}}")

    for i, values in enumerate(valid_combos2):
        params = dict(zip(keys, values))
        pnl = run_osmium_with_params(params)
        results.append((pnl, params))
        if (i + 1) % 100 == 0:
            best = max(r[0] for r in results)
            print(f"  ... {i + 1}/{len(valid_combos2)} done (best: {best:.2f})")

    results.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate
    seen = set()
    unique = []
    for pnl, params in results:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            unique.append((pnl, params))
    results = unique

    print()
    print("TOP 30 by Osmium PnL (days -2 + -1 + 0):")
    print("-" * 110)
    print(f"{'Rank':>4}  {'PnL':>10}  Parameters")
    print("-" * 110)
    for rank, (pnl, params) in enumerate(results[:30], 1):
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"{rank:>4}  {pnl:>10.2f}  {param_str}")

    return results


def run_heatmap_with_params(param_overrides: dict) -> float:
    """Load round1_osmium_heatmap, patch module-level constants, run round 1 days."""
    module = load_algo("round1_osmium_heatmap")
    for k, v in param_overrides.items():
        setattr(module, k, v)

    reader = DataReader(DATA_DIR)
    total_pnl = 0.0
    for day in DAYS_1:
        result = run_backtest(module, reader, ROUND_1, day)
        if result:
            total_pnl += result["pnl_by_product"].get("ASH_COATED_OSMIUM", 0.0)
    return total_pnl

def run_round2_with_params(osmium_overrides: dict, pepper_overrides: dict) -> Dict[str, float]:
    """Load round_2 algo, patch OsmiumParams/PepperParams, run round 2 days."""
    module = load_algo("round_2")
    for k, v in osmium_overrides.items():
        setattr(module.OsmiumParams, k, v)
    for k, v in pepper_overrides.items():
        setattr(module.PepperParams, k, v)

    reader = DataReader(DATA_DIR)
    pnl = {"total": 0.0, "osmium": 0.0, "pepper": 0.0}
    for day in DAYS_2:
        result = run_backtest(module, reader, ROUND_2, day)
        if result:
            pnl["total"] += result["profit"]
            pnl["osmium"] += result["pnl_by_product"].get("ASH_COATED_OSMIUM", 0.0)
            pnl["pepper"] += result["pnl_by_product"].get("INTARIAN_PEPPER_ROOT", 0.0)
    return pnl


def sweep_round2():
    print("=" * 80)
    print("SWEEP: round_2 (INV_SKEW, BUY_LIMIT) - two independent 1D sweeps")
    print(f"Days: {DAYS_2}")
    print("=" * 80)

    # Baseline params (held fixed for the other axis)
    base_skew = 3
    base_buy = 13008

    inv_values = [0, 1, 2, 3, 4, 5, 6, 8, 10]
    buy_values = [13004, 13006, 13007, 13008, 13009, 13010, 13012, 13015]

    print(f"\n--- Sweep 1: INV_SKEW (BUY_LIMIT fixed at {base_buy}) ---")
    print(f"{'INV_SKEW':>10}  {'Total':>10}  {'Osmium':>10}  {'Pepper':>10}")
    print("-" * 46)
    skew_results = []
    for s in inv_values:
        p = run_round2_with_params({"INV_SKEW": s}, {"BUY_LIMIT": base_buy})
        skew_results.append((s, p))
        print(f"{s:>10}  {p['total']:>10.2f}  {p['osmium']:>10.2f}  {p['pepper']:>10.2f}")
    best_skew = max(skew_results, key=lambda r: r[1]["osmium"])
    print(f"Best INV_SKEW by osmium PnL: {best_skew[0]}  (osmium={best_skew[1]['osmium']:.2f})")

    print(f"\n--- Sweep 2: BUY_LIMIT (INV_SKEW fixed at {base_skew}) ---")
    print(f"{'BUY_LIMIT':>10}  {'Total':>10}  {'Osmium':>10}  {'Pepper':>10}")
    print("-" * 46)
    buy_results = []
    for b in buy_values:
        p = run_round2_with_params({"INV_SKEW": base_skew}, {"BUY_LIMIT": b})
        buy_results.append((b, p))
        print(f"{b:>10}  {p['total']:>10.2f}  {p['osmium']:>10.2f}  {p['pepper']:>10.2f}")
    best_buy = max(buy_results, key=lambda r: r[1]["pepper"])
    print(f"Best BUY_LIMIT by pepper PnL: {best_buy[0]}  (pepper={best_buy[1]['pepper']:.2f})")

    # Combined optimum
    print(f"\n--- Combined optimum ---")
    combo = run_round2_with_params({"INV_SKEW": best_skew[0]}, {"BUY_LIMIT": best_buy[0]})
    print(f"INV_SKEW={best_skew[0]}  BUY_LIMIT={best_buy[0]}  "
          f"total={combo['total']:.2f}  osmium={combo['osmium']:.2f}  pepper={combo['pepper']:.2f}")

    # Sweep 3: DRIFT_STRENGTH (directional mean-reversion bias)
    print(f"\n--- Sweep 3: DRIFT_STRENGTH (INV_SKEW={best_skew[0]}, BUY_LIMIT={best_buy[0]}) ---")
    print(f"{'DRIFT':>8}  {'Total':>10}  {'Osmium':>10}  {'Pepper':>10}")
    print("-" * 44)
    drift_values = [0, 1, 2, 3, 5, 8, 12, 20, 40]
    drift_results = []
    for d in drift_values:
        p = run_round2_with_params(
            {"INV_SKEW": best_skew[0], "DRIFT_STRENGTH": d},
            {"BUY_LIMIT": best_buy[0]})
        drift_results.append((d, p))
        print(f"{d:>8}  {p['total']:>10.2f}  {p['osmium']:>10.2f}  {p['pepper']:>10.2f}")
    best_drift = max(drift_results, key=lambda r: r[1]["osmium"])
    print(f"Best DRIFT_STRENGTH by osmium PnL: {best_drift[0]}  (osmium={best_drift[1]['osmium']:.2f})")

    # Joint refine: (INV_SKEW, DRIFT_STRENGTH) around best
    print(f"\n--- Joint refine: INV_SKEW x DRIFT_STRENGTH (BUY_LIMIT={best_buy[0]}) ---")
    print(f"{'SKEW':>5} {'DRIFT':>6}  {'Total':>10}  {'Osmium':>10}")
    print("-" * 40)
    joint = []
    for s in [1, 2, 3, 4]:
        for d in [0, 2, 3, 5, 8, 12]:
            p = run_round2_with_params(
                {"INV_SKEW": s, "DRIFT_STRENGTH": d},
                {"BUY_LIMIT": best_buy[0]})
            joint.append(((s, d), p))
            print(f"{s:>5} {d:>6}  {p['total']:>10.2f}  {p['osmium']:>10.2f}")
    best_joint = max(joint, key=lambda r: r[1]["osmium"])
    print(f"Best joint (skew, drift) by osmium PnL: {best_joint[0]}  osmium={best_joint[1]['osmium']:.2f}")

    return skew_results, buy_results, drift_results, joint


def sweep_round2_full():
    """Full sweep across all osmium + pepper params. Axis-aligned 1D sweeps
    around the current best, then joint refine on the most impactful pair."""
    print("=" * 80)
    print("SWEEP: round_2 FULL - all parameters")
    print(f"Days: {DAYS_2}")
    print("=" * 80)

    # Current best baseline (algorithm defaults)
    base_osm = {
        "MA_WINDOW": 40, "ANCHOR_WEIGHT": 0.15, "HALF_SPREAD": 8,
        "NARROW_SPREAD": 13, "NARROW_EDGE": 1,
        "INV_SKEW": 2, "DRIFT_STRENGTH": 20,
    }
    base_pep = {"BUY_LIMIT": 13008}

    baseline = run_round2_with_params(base_osm, base_pep)
    print(f"\nBaseline: total={baseline['total']:.2f}  osmium={baseline['osmium']:.2f}  pepper={baseline['pepper']:.2f}")

    axes_osm = {
        "MA_WINDOW":      [10, 20, 30, 40, 60, 80, 120],
        "ANCHOR_WEIGHT":  [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        "HALF_SPREAD":    [4, 6, 8, 10, 12, 15],
        "NARROW_SPREAD":  [7, 10, 13, 16, 20, 25],
        "NARROW_EDGE":    [0, 1, 2, 3],
        "INV_SKEW":       [0, 1, 2, 3, 4, 6],
        "DRIFT_STRENGTH": [0, 5, 10, 15, 20, 25, 30, 40, 60],
    }

    best_per_axis = {}  # axis -> (value, pnl)
    for axis, values in axes_osm.items():
        print(f"\n--- {axis} (others at baseline) ---")
        print(f"{axis:>16} {'Total':>10} {'Osmium':>10}")
        best_v, best_p = base_osm[axis], baseline["osmium"]
        for v in values:
            ov = dict(base_osm); ov[axis] = v
            p = run_round2_with_params(ov, base_pep)
            marker = " *" if v == base_osm[axis] else ""
            print(f"{v!s:>16} {p['total']:>10.2f} {p['osmium']:>10.2f}{marker}")
            if p["osmium"] > best_p:
                best_v, best_p = v, p["osmium"]
        best_per_axis[axis] = (best_v, best_p)
        print(f"  best {axis}={best_v}  osmium={best_p:.2f}  (vs baseline {baseline['osmium']:.2f})")

    # Pepper
    print(f"\n--- BUY_LIMIT (osmium params at baseline) ---")
    best_buy_v, best_buy_p = base_pep["BUY_LIMIT"], baseline["pepper"]
    for v in [13005, 13006, 13007, 13008, 13009, 13010, 13012]:
        p = run_round2_with_params(base_osm, {"BUY_LIMIT": v})
        marker = " *" if v == base_pep["BUY_LIMIT"] else ""
        print(f"{v:>10}  total={p['total']:>10.2f}  pepper={p['pepper']:>10.2f}{marker}")
        if p["pepper"] > best_buy_p:
            best_buy_v, best_buy_p = v, p["pepper"]

    # Summary of 1D wins
    print(f"\n=== 1D AXIS BESTS ===")
    for axis, (v, p) in best_per_axis.items():
        delta = p - baseline["osmium"]
        print(f"  {axis}={v}  osmium={p:.2f}  ({delta:+.2f} vs base)")
    print(f"  BUY_LIMIT={best_buy_v}  pepper={best_buy_p:.2f}")

    # Apply all 1D bests simultaneously
    combined_osm = {k: best_per_axis[k][0] for k in axes_osm}
    combined_pep = {"BUY_LIMIT": best_buy_v}
    all_best = run_round2_with_params(combined_osm, combined_pep)
    print(f"\n=== All 1D bests combined ===")
    for k, v in combined_osm.items(): print(f"  {k} = {v}")
    print(f"  BUY_LIMIT = {best_buy_v}")
    print(f"  total={all_best['total']:.2f}  osmium={all_best['osmium']:.2f}  pepper={all_best['pepper']:.2f}")
    print(f"  (baseline total={baseline['total']:.2f}  delta={all_best['total']-baseline['total']:+.2f})")

    # Joint refine: INV_SKEW x DRIFT_STRENGTH (known to interact)
    print(f"\n=== Joint refine: INV_SKEW x DRIFT_STRENGTH (other params = combined best) ===")
    print(f"{'SKEW':>5} {'DRIFT':>7} {'Osmium':>10}")
    joint = []
    for s in [1, 2, 3, 4]:
        for d in [10, 15, 20, 25, 30, 40]:
            ov = dict(combined_osm); ov["INV_SKEW"] = s; ov["DRIFT_STRENGTH"] = d
            p = run_round2_with_params(ov, combined_pep)
            joint.append(((s, d), p))
            print(f"{s:>5} {d:>7} {p['osmium']:>10.2f}")
    best_joint = max(joint, key=lambda r: r[1]["osmium"])
    print(f"Best (skew, drift): {best_joint[0]}  osmium={best_joint[1]['osmium']:.2f}  total={best_joint[1]['total']:.2f}")


def run_variant_per_day(algo_name: str, osmium_overrides: dict, pepper_overrides: dict):
    """Run a round_2 variant per-day and return per-day + total PnL dicts."""
    module = load_algo(algo_name)
    for k, v in osmium_overrides.items():
        setattr(module.OsmiumParams, k, v)
    for k, v in pepper_overrides.items():
        setattr(module.PepperParams, k, v)

    reader = DataReader(DATA_DIR)
    per_day = {}
    for day in DAYS_2:
        result = run_backtest(module, reader, ROUND_2, day)
        if result:
            per_day[day] = {
                "total": result["profit"],
                "osmium": result["pnl_by_product"].get("ASH_COATED_OSMIUM", 0.0),
                "pepper": result["pnl_by_product"].get("INTARIAN_PEPPER_ROOT", 0.0),
            }
    agg = {"total": sum(d["total"] for d in per_day.values()),
           "osmium": sum(d["osmium"] for d in per_day.values()),
           "pepper": sum(d["pepper"] for d in per_day.values())}
    return per_day, agg


def _print_strategy_header(name, param_name, param_values):
    print("=" * 90)
    print(f"STRATEGY: {name}")
    print(f"Sweeping {param_name} over {param_values}  |  days: {DAYS_2}")
    print("=" * 90)


def _print_per_day_table(rows, param_name):
    # rows: list of (param_value, per_day_dict, agg_dict)
    days = DAYS_2
    header = f"{param_name:>14}  " + "  ".join(f"day{d:>3}:osm  day{d:>3}:pep" for d in days) + f"  {'TOT osm':>9}  {'TOT pep':>9}  {'TOT':>10}"
    print(header)
    print("-" * len(header))
    for v, per, agg in rows:
        cells = "  ".join(f"{per[d]['osmium']:>10.1f}  {per[d]['pepper']:>10.1f}" for d in days)
        print(f"{v!s:>14}  {cells}  {agg['osmium']:>9.1f}  {agg['pepper']:>9.1f}  {agg['total']:>10.1f}")


def _per_day_winners(rows, focus_key):
    """Pick best param value per-day by focus_key ('osmium'|'pepper'|'total')."""
    winners = {}
    for d in DAYS_2:
        best = max(rows, key=lambda r: r[1][d][focus_key])
        winners[d] = (best[0], best[1][d][focus_key])
    overall = max(rows, key=lambda r: r[2][focus_key])
    return winners, overall


def sweep_strategy_1():
    _print_strategy_header("1. Skip resting on tight spread", "MIN_POST_SPREAD",
                           [0, 1, 2, 3, 4, 5, 6, 8, 10])
    rows = []
    for v in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        per, agg = run_variant_per_day("round_2_v1_spread", {"MIN_POST_SPREAD": v}, {})
        rows.append((v, per, agg))
    _print_per_day_table(rows, "MIN_POST_SPREAD")
    winners, overall = _per_day_winners(rows, "osmium")
    print("\nBest MIN_POST_SPREAD per day (by osmium PnL):")
    for d, (v, p) in winners.items():
        print(f"  day {d}: {v}  osmium={p:.2f}")
    print(f"Best by 3-day osmium total: {overall[0]}  osmium={overall[2]['osmium']:.2f}  total={overall[2]['total']:.2f}")
    return rows


def sweep_strategy_2():
    _print_strategy_header("2. Tranched pepper accumulation", "TRANCHE_RATE",
                           [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 9.9])
    rows = []
    for v in [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 9.9]:
        per, agg = run_variant_per_day("round_2_v2_tranche", {}, {"TRANCHE_RATE": v})
        rows.append((v, per, agg))
    _print_per_day_table(rows, "TRANCHE_RATE")
    winners, overall = _per_day_winners(rows, "pepper")
    print("\nBest TRANCHE_RATE per day (by pepper PnL):")
    for d, (v, p) in winners.items():
        print(f"  day {d}: {v}  pepper={p:.2f}")
    print(f"Best by 3-day pepper total: {overall[0]}  pepper={overall[2]['pepper']:.2f}  total={overall[2]['total']:.2f}")
    return rows


def sweep_strategy_3():
    _print_strategy_header("3. Asymmetric DRIFT (long side)", "DRIFT_LONG",
                           [10, 20, 25, 35, 50, 70, 100])
    rows = []
    for v in [10, 20, 25, 35, 50, 70, 100]:
        per, agg = run_variant_per_day("round_2_v3_asym", {"DRIFT_LONG": v, "DRIFT_SHORT": 25}, {})
        rows.append((v, per, agg))
    _print_per_day_table(rows, "DRIFT_LONG")
    winners, overall = _per_day_winners(rows, "osmium")
    print("\nBest DRIFT_LONG per day (DRIFT_SHORT=25, by osmium PnL):")
    for d, (v, p) in winners.items():
        print(f"  day {d}: {v}  osmium={p:.2f}")
    print(f"Best by 3-day osmium total: {overall[0]}  osmium={overall[2]['osmium']:.2f}  total={overall[2]['total']:.2f}")
    return rows


def sweep_strategy_7():
    _print_strategy_header("7. Urgency-widened takes", "URGENCY_EDGE",
                           [0, 1, 2, 3, 4, 5, 6, 8, 12])
    rows = []
    for v in [0, 1, 2, 3, 4, 5, 6, 8, 12]:
        per, agg = run_variant_per_day("round_2_v7_urgency", {"URGENCY_EDGE": v}, {})
        rows.append((v, per, agg))
    _print_per_day_table(rows, "URGENCY_EDGE")
    winners, overall = _per_day_winners(rows, "osmium")
    print("\nBest URGENCY_EDGE per day (by osmium PnL):")
    for d, (v, p) in winners.items():
        print(f"  day {d}: {v}  osmium={p:.2f}")
    print(f"Best by 3-day osmium total: {overall[0]}  osmium={overall[2]['osmium']:.2f}  total={overall[2]['total']:.2f}")
    return rows


def sweep_strategy_6():
    _print_strategy_header("6. Never-cross-mid resting clamp", "MID_MARGIN",
                           [-2, -1, 0, 1, 2, 3])
    rows = []
    for v in [-2, -1, 0, 1, 2, 3]:
        per, agg = run_variant_per_day("round_2_v6_nocross", {"MID_MARGIN": v}, {})
        rows.append((v, per, agg))
    _print_per_day_table(rows, "MID_MARGIN")
    winners, overall = _per_day_winners(rows, "osmium")
    print("\nBest MID_MARGIN per day (by osmium PnL):")
    for d, (v, p) in winners.items():
        print(f"  day {d}: {v}  osmium={p:.2f}")
    print(f"Best by 3-day osmium total: {overall[0]}  osmium={overall[2]['osmium']:.2f}  total={overall[2]['total']:.2f}")
    return rows


def sweep_strategy_5():
    _print_strategy_header("5. Dynamic pepper BUY_LIMIT (first_ask + margin)", "BUY_MARGIN",
                           [0, 1, 2, 3, 5, 8, 12, 20])
    rows = []
    for v in [0, 1, 2, 3, 5, 8, 12, 20]:
        per, agg = run_variant_per_day("round_2_v5_dyncap", {}, {"BUY_MARGIN": v})
        rows.append((v, per, agg))
    _print_per_day_table(rows, "BUY_MARGIN")
    winners, overall = _per_day_winners(rows, "pepper")
    print("\nBest BUY_MARGIN per day (by pepper PnL):")
    for d, (v, p) in winners.items():
        print(f"  day {d}: {v}  pepper={p:.2f}")
    print(f"Best by 3-day pepper total: {overall[0]}  pepper={overall[2]['pepper']:.2f}  total={overall[2]['total']:.2f}")
    return rows


def sweep_strategy_4():
    _print_strategy_header("4. Microprice fair value", "MICROPRICE_WEIGHT",
                           [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        per, agg = run_variant_per_day("round_2_v4_micro", {"MICROPRICE_WEIGHT": v}, {})
        rows.append((v, per, agg))
    _print_per_day_table(rows, "MICROPRICE_WEIGHT")
    winners, overall = _per_day_winners(rows, "osmium")
    print("\nBest MICROPRICE_WEIGHT per day (by osmium PnL):")
    for d, (v, p) in winners.items():
        print(f"  day {d}: {v}  osmium={p:.2f}")
    print(f"Best by 3-day osmium total: {overall[0]}  osmium={overall[2]['osmium']:.2f}  total={overall[2]['total']:.2f}")
    return rows


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    if target == "osmium":
        sweep_osmium()
    elif target == "osmium_heatmap":
        sweep_osmium_heatmap()
    elif target == "round2":
        sweep_round2()
    elif target == "round2_full":
        sweep_round2_full()
    elif target == "strategy_1":
        sweep_strategy_1()
    elif target == "strategy_2":
        sweep_strategy_2()
    elif target == "strategy_3":
        sweep_strategy_3()
    elif target == "strategy_4":
        sweep_strategy_4()
    elif target == "strategy_5":
        sweep_strategy_5()
    elif target == "strategy_6":
        sweep_strategy_6()
    elif target == "strategy_7":
        sweep_strategy_7()
    elif target == "strategies_all":
        sweep_strategy_1()
        print()
        sweep_strategy_2()
        print()
        sweep_strategy_3()
        print()
        sweep_strategy_4()
    else:
        run_baseline()

        if target in ("short_warmup_ma", "both"):
            sweep_short_warmup_ma()
            print()

        if target in ("ema_trend", "both"):
            sweep_ema_trend()
