"""Sweep v5e PISTA-MM-with-imbalance-lean params (8-worker parallel)."""
import sys
import os
from pathlib import Path
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def _load_module(path):
    spec = importlib.util.spec_from_file_location(f"mod_{os.getpid()}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_one(args):
    hl_imb, imb_full, main_qty, opp_qty = args
    sys.path.insert(0, str(ROOT))
    from backtester import DataReader, run_backtest

    algo_path = ROOT / "algorithms" / "ROUND_5" / "SNACKPACK" / "snackpack_v5e.py"
    mod = _load_module(algo_path)
    mod.P.HL_IMB = hl_imb
    mod.P.IMB_FULL = imb_full
    mod.P.PISTA_MAIN_QTY = main_qty
    mod.P.PISTA_OPP_QTY = opp_qty
    mod.ALPHA_IMB = 1.0 - 0.5 ** (1.0 / hl_imb)

    reader = DataReader(ROOT / "data")
    pnls = []; pista = 0.0
    for d in [2, 3, 4]:
        r = run_backtest(mod, reader, 5, d, print_output=False)
        pnls.append(r["profit"])
        pista += r["pnl_by_product"].get("SNACKPACK_PISTACHIO", 0.0)
    return (args, pnls, pista)


def main():
    configs = []
    for hl_imb in (10, 30, 80, 200):
        for imb_full in (0.2, 0.4, 0.6, 1.0):
            for main_qty in (3, 5, 8):
                for opp_qty in (1, 3, 5):
                    if opp_qty >= main_qty: continue
                    configs.append((hl_imb, imb_full, main_qty, opp_qty))

    print(f"Running {len(configs)} configs across 8 workers...\n", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, c): c for c in configs}
        for fut in as_completed(futs):
            cfg, pnls, pista = fut.result()
            tot = sum(pnls)
            hl, imf, mq, oq = cfg
            results.append((tot, pista, hl, imf, mq, oq, pnls))
            print(f"HL_IMB={hl:>3d}  IMB_FULL={imf:.2f}  MAIN={mq} OPP={oq}  "
                  f"tot={tot:>9.0f}  pista={pista:>+7.0f}  "
                  f"({pnls[0]:.0f},{pnls[1]:.0f},{pnls[2]:.0f})",
                  flush=True)

    results.sort(reverse=True)
    print("\nTop 10 by total:")
    for tot, pista, hl, imf, mq, oq, pnls in results[:10]:
        print(f"  HL_IMB={hl:>3d}  IMB_FULL={imf:.2f}  MAIN={mq} OPP={oq}  "
              f"tot={tot:>9.0f}  pista={pista:>+7.0f}")


if __name__ == "__main__":
    main()
