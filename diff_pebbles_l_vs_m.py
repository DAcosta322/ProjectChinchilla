"""Compare base_l vs base_m per-product PnL."""
import sys, contextlib, io, importlib.util
from pathlib import Path
sys.path.insert(0, '.')

import backtester as BT
reader = BT.DataReader(Path('data'))

def run(algo_path):
    spec = importlib.util.spec_from_file_location('m', algo_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prod_pnl = {}
    for d in [2, 3, 4]:
        with contextlib.redirect_stdout(io.StringIO()):
            r = BT.run_backtest(mod, reader, 5, d)
        for p, v in r['pnl_by_product'].items():
            prod_pnl[p] = prod_pnl.get(p, 0) + v
    return prod_pnl

l_pnl = run('algorithms/ROUND_5/PEBBLES/pebbles_combined_v3.py')   # XL-only phantom
m_pnl = run('algorithms/ROUND_5/PEBBLES/pebbles_base_m.py')         # all-5 phantom

print(f"{'product':14} {'XL-only':>10} {'all-5':>10} {'delta':>10}")
print('-' * 48)
peb = ['PEBBLES_XS','PEBBLES_S','PEBBLES_M','PEBBLES_L','PEBBLES_XL']
for p in peb:
    a = l_pnl.get(p, 0); b = m_pnl.get(p, 0)
    print(f"{p:14} {a:>+10.0f} {b:>+10.0f} {b-a:>+10.0f}")
print('-' * 48)
print(f"{'TOTAL':14} {sum(l_pnl.get(p,0) for p in peb):>+10.0f} {sum(m_pnl.get(p,0) for p in peb):>+10.0f} {sum(m_pnl.get(p,0)-l_pnl.get(p,0) for p in peb):>+10.0f}")
