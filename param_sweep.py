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
