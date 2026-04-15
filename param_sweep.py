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


def sweep_osmium_heatmap():
    print("=" * 70)
    print("SWEEP: round1_osmium_heatmap (ASH_COATED_OSMIUM)")
    print("=" * 70)

    # Phase 1: coarse sweep
    coarse_grid = {
        "MA_WINDOW": [5, 10, 20, 40, 80],
        "HALF_SPREAD": [10, 15, 20, 25, 30],
    }

    keys = list(coarse_grid.keys())
    combos = list(itertools.product(*[coarse_grid[k] for k in keys]))
    print(f"Phase 1 (coarse): {len(combos)} combinations")

    results = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        pnl = run_heatmap_with_params(params)
        results.append((pnl, params))
        if (i + 1) % 10 == 0:
            best = max(r[0] for r in results)
            print(f"  ... {i + 1}/{len(combos)} done (best: {best:.2f})")

    results.sort(key=lambda x: x[0], reverse=True)
    best_params = results[0][1]

    print(f"\nPhase 1 best: PnL={results[0][0]:.2f}")
    print(f"  {best_params}")

    # Phase 2: fine sweep around best
    def refine_values(center, candidates):
        candidates = sorted(set(candidates))
        idx = min(range(len(candidates)), key=lambda i: abs(candidates[i] - center))
        lo = max(0, idx - 1)
        hi = min(len(candidates), idx + 2)
        return candidates[lo:hi]

    fine_grid = {
        "MA_WINDOW": refine_values(best_params["MA_WINDOW"],
                                   [3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80]),
        "HALF_SPREAD": refine_values(best_params["HALF_SPREAD"],
                                     [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35]),
    }

    combos2 = list(itertools.product(*[fine_grid[k] for k in keys]))
    print(f"\nPhase 2 (fine): {len(combos2)} combinations around best")
    print(f"  Grid: {{{', '.join(f'{k}: {fine_grid[k]}' for k in keys)}}}")

    for i, values in enumerate(combos2):
        params = dict(zip(keys, values))
        pnl = run_heatmap_with_params(params)
        results.append((pnl, params))

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
    print("TOP 20 by Osmium PnL (days -2 + -1 + 0):")
    print("-" * 80)
    print(f"{'Rank':>4}  {'PnL':>10}  Parameters")
    print("-" * 80)
    for rank, (pnl, params) in enumerate(results[:20], 1):
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"{rank:>4}  {pnl:>10.2f}  {param_str}")

    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    if target == "osmium":
        sweep_osmium()
    elif target == "osmium_heatmap":
        sweep_osmium_heatmap()
    else:
        run_baseline()

        if target in ("short_warmup_ma", "both"):
            sweep_short_warmup_ma()
            print()

        if target in ("ema_trend", "both"):
            sweep_ema_trend()
