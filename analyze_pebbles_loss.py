"""Diagnose PEBBLES_M and PEBBLES_XS bleed under combined_v3.

Looks at:
  1. Per-day per-product PnL split (realized vs MTM)
  2. Per-fragment per-product PnL
  3. Trade-level avg buy/sell prices (spread/RT)
  4. Trade timing — did fills cluster on bad ticks?
"""
import sys, contextlib, io, importlib.util, csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '.')

ALGO = 'algorithms/ROUND_5/round_5_combined_v3.py'
TARGETS = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']

spec = importlib.util.spec_from_file_location('m', ALGO)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import backtester as BT
reader = BT.DataReader(Path('data'))

print('=== Per-day per-product PnL (combined_v3) ===')
for d in [2, 3, 4]:
    with contextlib.redirect_stdout(io.StringIO()):
        r = BT.run_backtest(m, reader, 5, d)
    print(f'\nDAY {d}:')
    for p in TARGETS:
        v = r['pnl_by_product'].get(p, 0)
        print(f'  {p:14}  total={v:>+10.2f}')

print()
print('=== Per-fragment per-product (Day 2/3/4 × 10 frags) ===')
for d in [2, 3, 4]:
    with contextlib.redirect_stdout(io.StringIO()):
        r = BT.run_backtest(m, reader, 5, d)
    pnl_at_ts = r['pnl_at_ts']
    by_prod_at = r['pnl_by_prod_at_ts']
    tss = sorted(pnl_at_ts.keys())
    print(f'\nDAY {d}:')
    hdr = ['frag','XS','S','M','L','XL','sum']
    print('  ' + ' '.join(f'{h:>9}' for h in hdr))
    cur = tss[0]; i = 0
    while cur <= tss[-1]:
        last = max((t for t in tss if t < cur + 100000), default=None)
        if last is None: break
        deltas = []
        prev = max(t for t in tss if t < cur) if cur > tss[0] else None
        for p in TARGETS:
            sp = by_prod_at.get(p, {}).get(prev, 0) if prev else 0
            ep = by_prod_at.get(p, {}).get(last, 0)
            deltas.append(ep - sp)
        print(f'  {i:>4} ' + ' '.join(f'{d:>+9.0f}' for d in deltas) + f' {sum(deltas):>+9.0f}')
        cur += 100000; i += 1

print()
print('=== Trade-level breakdown ===')
log_dir = sorted(Path('logs/ROUND_5').glob('round_5_combined_v3*'), reverse=True)[0]
all_trades = defaultdict(lambda: {'b':0,'bq':0,'bv':0.0,'s':0,'sq':0,'sv':0.0})
for log_file in sorted(log_dir.glob('*.log')):
    import json as _j
    with open(log_file) as f:
        log = _j.load(f)
    day = int(log_file.stem.split('_d')[1].split('_')[0])
    for t in log['tradeHistory']:
        p = t['symbol']
        if p not in TARGETS: continue
        key = (p, day)
        s = all_trades[key]
        if t['buyer'] == 'SUBMISSION':
            s['b'] += 1; s['bq'] += t['quantity']; s['bv'] += t['price']*t['quantity']
        else:
            s['s'] += 1; s['sq'] += t['quantity']; s['sv'] += t['price']*t['quantity']

print(f'{"product":14} {"day":>3} {"buys":>5} {"qty":>5} {"avgB":>10} {"sells":>5} {"qty":>5} {"avgS":>10} {"spread/RT":>10}')
for (p, day), s in sorted(all_trades.items()):
    avgB = s['bv']/s['bq'] if s['bq'] else 0
    avgS = s['sv']/s['sq'] if s['sq'] else 0
    rt = avgS - avgB if (s['bq'] and s['sq']) else 0
    print(f'{p:14} {day:>3} {s["b"]:>5} {s["bq"]:>5} {avgB:>10.2f} {s["s"]:>5} {s["sq"]:>5} {avgS:>10.2f} {rt:>+10.2f}')
