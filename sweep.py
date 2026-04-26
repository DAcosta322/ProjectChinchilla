"""Parallel parameter sweep tool.

Drives `backtester.run_backtest` or `mc_backtester.run_one_path` over a
cartesian product of class-attribute patches. Caches per-day data parses
(and MC model fits) once on disk, then loads them once per worker process
so the heavy work is amortized across all configs.

Usage
-----
    python sweep.py <config.py> [--workers N] [--top K]
                                [--cache-dir .sweep_cache] [--rebuild-cache]

Config file (Python module) — required attrs:
    ALGO   : str               path to algorithm .py (relative to repo root ok)
    ROUND  : int
    DAYS   : list[int]
    MODE   : "bt" | "mc"       which backtester to run
    GRID   : dict[str, list]   keys are "ClassName.ATTR"; cartesian product

Optional:
    MC_PATHS : int = 30        paths per (config, day) in MC mode
    MC_SEED  : int = 42
    SKIP     : callable(params: dict) -> bool   filter combos
    TRACK    : list[str]       products to show per-product columns for
"""

from __future__ import annotations
import argparse
import importlib.util
import io
import contextlib
import itertools
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("sweep_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "ALGO": getattr(mod, "ALGO"),
        "ROUND": int(getattr(mod, "ROUND")),
        "DAYS": list(getattr(mod, "DAYS")),
        "MODE": getattr(mod, "MODE", "bt"),
        "GRID": dict(getattr(mod, "GRID")),
        "MC_PATHS": int(getattr(mod, "MC_PATHS", 30)),
        "MC_SEED": int(getattr(mod, "MC_SEED", 42)),
        "SKIP": getattr(mod, "SKIP", lambda _params: False),
        "TRACK": list(getattr(mod, "TRACK", [])),
    }


# ---------------------------------------------------------------------------
# Per-day cache: parses CSVs once, fits MC models once
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: Path, mode: str, round_num: int, day_num: int) -> Path:
    return cache_dir / f"{mode}_r{round_num}_d{day_num}.pkl"


def build_cache(cfg: dict, cache_dir: Path) -> Dict[int, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    import backtester as BT
    reader = BT.DataReader(SCRIPT_DIR / "data")
    out: Dict[int, Path] = {}
    for d in cfg["DAYS"]:
        path = _cache_path(cache_dir, cfg["MODE"], cfg["ROUND"], d)
        if path.exists():
            out[d] = path
            print(f"  [cache] day {d}: hit -> {path.name}")
            continue
        t0 = time.time()
        price_data = reader.read_prices(cfg["ROUND"], d)
        trade_data = reader.read_trades(cfg["ROUND"], d)
        timestamps = sorted(price_data.keys())
        if not timestamps:
            print(f"  [cache] day {d}: NO DATA, skipping")
            continue
        products = sorted(price_data[timestamps[0]].keys())
        payload: Dict[str, Any] = {
            "price_data": price_data,
            "trade_data": trade_data,
            "timestamps": timestamps,
            "products": products,
        }
        if cfg["MODE"] == "mc":
            from mc_backtester import MarketModel
            models = {}
            for p in products:
                prs = [price_data[ts][p] for ts in timestamps if p in price_data[ts]]
                mts = []
                for ts in timestamps:
                    mts.extend(trade_data.get(ts, {}).get(p, []))
                m = MarketModel(p)
                m.fit(prs, mts, timestamps)
                models[p] = m
            payload["models"] = models
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [cache] day {d}: built in {time.time()-t0:.1f}s -> {path.name}")
        out[d] = path
    return out


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
def apply_patches(module, params: Dict[str, Any]) -> None:
    """params keys are 'ClassName.ATTR' -> setattr on the loaded module's class."""
    for key, val in params.items():
        cls_name, attr = key.split(".", 1)
        cls = getattr(module, cls_name)
        setattr(cls, attr, val)


# ---------------------------------------------------------------------------
# Worker (uses ProcessPoolExecutor initializer to load data once per process)
# ---------------------------------------------------------------------------
_WORKER: Dict[str, Any] = {}


def _worker_init(algo_path: str, mode: str, round_num: int,
                 day_paths: Dict[int, str], mc_paths: int, mc_seed: int) -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    payloads: Dict[int, Any] = {}
    for d, p in day_paths.items():
        with open(p, "rb") as f:
            payloads[d] = pickle.load(f)
    _WORKER.update({
        "algo_path": algo_path,
        "mode": mode,
        "round_num": round_num,
        "mc_paths": mc_paths,
        "mc_seed": mc_seed,
        "payloads": payloads,
    })


def _run_config(params: Dict[str, Any]) -> Tuple[Dict[str, Any],
                                                  Dict[int, float],
                                                  Dict[int, Dict[str, float]]]:
    import backtester as BT
    module = BT.load_algorithm(Path(_WORKER["algo_path"]))
    apply_patches(module, params)

    mode = _WORKER["mode"]
    round_num = _WORKER["round_num"]
    profit_by_day: Dict[int, float] = {}
    pnl_by_prod: Dict[int, Dict[str, float]] = {}

    if mode == "bt":
        reader = BT.DataReader(SCRIPT_DIR / "data")
        for d, payload in _WORKER["payloads"].items():
            reader.read_prices = lambda rn, dn, _p=payload["price_data"]: _p
            reader.read_trades = lambda rn, dn, _t=payload["trade_data"]: _t
            with contextlib.redirect_stdout(io.StringIO()):
                r = BT.run_backtest(module, reader, round_num, d, print_output=False)
            if r:
                profit_by_day[d] = r["profit"]
                pnl_by_prod[d] = dict(r["pnl_by_product"])
    elif mode == "mc":
        import random
        from mc_backtester import run_one_path
        n = _WORKER["mc_paths"]
        seed0 = _WORKER["mc_seed"]
        for d, payload in _WORKER["payloads"].items():
            models = payload["models"]
            timestamps = payload["timestamps"]
            products = payload["products"]
            totals: List[float] = []
            prod_sums: Dict[str, float] = {p: 0.0 for p in products}
            for i in range(n):
                rng = random.Random(seed0 + i)
                with contextlib.redirect_stdout(io.StringIO()):
                    r = run_one_path(module, models, timestamps, products, rng)
                totals.append(r["total"])
                for p, v in r["by_product"].items():
                    prod_sums[p] += v
            profit_by_day[d] = sum(totals) / n
            pnl_by_prod[d] = {p: v / n for p, v in prod_sums.items()}
    else:
        raise ValueError(f"unknown MODE: {mode}")

    return params, profit_by_day, pnl_by_prod


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_combos(grid: Dict[str, List[Any]], skip) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    out: List[Dict[str, Any]] = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        if skip(params):
            continue
        out.append(params)
    return out


def _print_table(results: List[dict], days: List[int], track: List[str], top: int) -> None:
    if not results:
        print("(no results)")
        return
    cols = ["TOTAL"] + [f"D{d}" for d in days] + track
    header = f"{'rank':>4}  " + " ".join(f"{c:>10}" for c in cols) + "  PARAMS"
    print(header)
    print("-" * min(len(header), 200))
    for rank, r in enumerate(results[:top], 1):
        line = f"{rank:>4}  {r['total']:>10.0f}"
        for d in days:
            line += f" {r['by_day'].get(d, 0):>10.0f}"
        for p in track:
            v = sum(r["by_prod"].get(d, {}).get(p, 0) for d in days)
            line += f" {v:>10.0f}"
        param_str = "  ".join(f"{k}={v}" for k, v in r["params"].items())
        line += f"  {param_str}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel parameter sweep")
    parser.add_argument("config", type=str, help="Path to sweep config .py")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--cache-dir", type=str, default=".sweep_cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cache_dir = Path(args.cache_dir)
    if args.rebuild_cache and cache_dir.exists():
        for f in cache_dir.glob("*.pkl"):
            f.unlink()

    print(f"Sweep: algo={cfg['ALGO']}  round={cfg['ROUND']}  days={cfg['DAYS']}  mode={cfg['MODE']}")
    print("Building / loading per-day cache...")
    day_paths = build_cache(cfg, cache_dir)
    if not day_paths:
        print("ERROR: no day data available")
        sys.exit(1)

    combos = _build_combos(cfg["GRID"], cfg["SKIP"])
    workers = min(args.workers, max(1, len(combos)))
    print(f"Configs: {len(combos)}  |  workers: {workers}  |  "
          f"per-day mc paths: {cfg['MC_PATHS'] if cfg['MODE']=='mc' else 'n/a'}")

    init_args = (
        cfg["ALGO"], cfg["MODE"], cfg["ROUND"],
        {d: str(p) for d, p in day_paths.items()},
        cfg["MC_PATHS"], cfg["MC_SEED"],
    )

    t_sweep = time.time()
    results: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init,
                             initargs=init_args) as ex:
        futures = {ex.submit(_run_config, p): p for p in combos}
        done = 0
        report_every = max(1, len(combos) // 20)
        for fut in as_completed(futures):
            done += 1
            try:
                params, by_day, by_prod = fut.result()
            except Exception as e:
                print(f"  [{done}/{len(combos)}] FAILED {futures[fut]}: {e}")
                continue
            total = sum(by_day.values())
            results.append({"params": params, "by_day": by_day,
                            "by_prod": by_prod, "total": total})
            if done % report_every == 0 or done == len(combos):
                best = max(r["total"] for r in results)
                print(f"  [{done}/{len(combos)}] best so far: {best:>10.0f}")

    print(f"Sweep finished in {time.time()-t_sweep:.1f}s")
    print()

    results.sort(key=lambda r: r["total"], reverse=True)
    days = sorted(cfg["DAYS"])
    print(f"TOP {args.top} BY TOTAL")
    _print_table(results, days, cfg["TRACK"], args.top)

    if cfg["TRACK"]:
        for prod in cfg["TRACK"]:
            print()
            print(f"TOP {args.top} BY {prod}")
            results.sort(key=lambda r, _p=prod:
                         sum(r["by_prod"].get(d, {}).get(_p, 0) for d in days),
                         reverse=True)
            _print_table(results, days, cfg["TRACK"], args.top)


if __name__ == "__main__":
    main()
