"""Look at the actual mid trajectories during PEBBLES_M's worst fragments."""
import csv
from pathlib import Path

DATA_DIR = Path('data') / 'ROUND_5'

def fragment_summary(day, prod, t0, t1):
    path = DATA_DIR / f'prices_round_5_day_{day}.csv'
    mids = []
    spreads = []
    with open(path, newline='') as f:
        r = csv.DictReader(f, delimiter=';')
        for row in r:
            if row['product'] != prod: continue
            ts = int(row['timestamp'])
            if ts < t0 or ts >= t1: continue
            mids.append(float(row['mid_price']))
            try:
                bb = int(float(row['bid_price_1'])); ba = int(float(row['ask_price_1']))
                spreads.append(ba - bb)
            except (ValueError, TypeError):
                pass
    if not mids: return None
    return {
        'start': mids[0], 'end': mids[-1], 'min': min(mids), 'max': max(mids),
        'range': max(mids) - min(mids), 'drift': mids[-1] - mids[0],
        'avg_spread': sum(spreads)/len(spreads) if spreads else 0,
        'n': len(mids),
    }


cases = [
    (4, 'PEBBLES_M',  200000, 300000, 'D4 frag 2 (-$3,490)'),
    (4, 'PEBBLES_L',  200000, 300000, 'D4 frag 2 (-$1,620)'),
    (4, 'PEBBLES_XL', 200000, 300000, 'D4 frag 2 (XL +$5,806)'),
    (3, 'PEBBLES_M',  600000, 700000, 'D3 frag 6 (-$3,228)'),
    (3, 'PEBBLES_L',  600000, 700000, 'D3 frag 6 (-$1,256)'),
    (3, 'PEBBLES_XL', 600000, 700000, 'D3 frag 6 (XL +$373)'),
    # Compare to GOOD M fragments
    (4, 'PEBBLES_M',  100000, 200000, 'D4 frag 1 M=+$2,106 (good)'),
    (2, 'PEBBLES_M',  500000, 600000, 'D2 frag 5 M=-$1,845 (bad)'),
]

for day, prod, t0, t1, label in cases:
    s = fragment_summary(day, prod, t0, t1)
    if s:
        print(f'{label}')
        print(f'  {prod}: start={s["start"]:.0f} end={s["end"]:.0f} min={s["min"]:.0f} max={s["max"]:.0f}')
        print(f'    range={s["range"]:.0f} drift={s["drift"]:+.0f} spread_avg={s["avg_spread"]:.1f} n={s["n"]}')
