"""Compare combined_v3 BT vs theoretical_max per category."""
import sys, contextlib, io, importlib.util
from pathlib import Path
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('m', 'algorithms/ROUND_5/round_5_combined_v4.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import backtester as BT
reader = BT.DataReader(Path('data'))
prod_pnl = {}
for d in [2, 3, 4]:
    with contextlib.redirect_stdout(io.StringIO()):
        r = BT.run_backtest(m, reader, 5, d)
    for p, v in r['pnl_by_product'].items():
        prod_pnl[p] = prod_pnl.get(p, 0) + v

theo = {
  'PEBBLES_XS': 794500, 'PEBBLES_S': 589498, 'PEBBLES_M': 572976,
  'PEBBLES_L': 579303, 'PEBBLES_XL': 1153974,
  'SNACKPACK_CHOCOLATE': 225371, 'SNACKPACK_VANILLA': 219735,
  'SNACKPACK_PISTACHIO': 167734, 'SNACKPACK_STRAWBERRY': 291496,
  'SNACKPACK_RASPBERRY': 273976,
  'GALAXY_SOUNDS_DARK_MATTER': 355186, 'GALAXY_SOUNDS_BLACK_HOLES': 425148,
  'GALAXY_SOUNDS_PLANETARY_RINGS': 420586, 'GALAXY_SOUNDS_SOLAR_WINDS': 389182,
  'GALAXY_SOUNDS_SOLAR_FLAMES': 397542,
  'SLEEP_POD_SUEDE': 432252, 'SLEEP_POD_LAMB_WOOL': 423071,
  'SLEEP_POD_POLYESTER': 455549, 'SLEEP_POD_NYLON': 516962, 'SLEEP_POD_COTTON': 478328,
  'MICROCHIP_CIRCLE': 466248, 'MICROCHIP_OVAL': 712622,
  'MICROCHIP_SQUARE': 599143, 'MICROCHIP_RECTANGLE': 816921, 'MICROCHIP_TRIANGLE': 600021,
  'ROBOT_VACUUMING': 663416, 'ROBOT_MOPPING': 667838, 'ROBOT_DISHES': 658367,
  'ROBOT_LAUNDRY': 749892, 'ROBOT_IRONING': 279300,
  'UV_VISOR_YELLOW': 421425, 'UV_VISOR_AMBER': 320806, 'UV_VISOR_ORANGE': 434720,
  'UV_VISOR_RED': 403118, 'UV_VISOR_MAGENTA': 412200,
  'TRANSLATOR_SPACE_GRAY': 538540, 'TRANSLATOR_ASTRO_BLACK': 544227,
  'TRANSLATOR_ECLIPSE_CHARCOAL': 484301, 'TRANSLATOR_GRAPHITE_MIST': 455493,
  'TRANSLATOR_VOID_BLUE': 402079,
  'PANEL_1X2': 363631, 'PANEL_2X2': 514670, 'PANEL_1X4': 543247,
  'PANEL_2X4': 417398, 'PANEL_4X4': 473708,
  'OXYGEN_SHAKE_MORNING_BREATH': 390590, 'OXYGEN_SHAKE_EVENING_BREATH': 384183,
  'OXYGEN_SHAKE_MINT': 392707, 'OXYGEN_SHAKE_CHOCOLATE': 384219,
  'OXYGEN_SHAKE_GARLIC': 481830,
}

cats = {}
for p, v in theo.items():
    if p.startswith('GALAXY'): cat = 'GALAXY_SOUNDS'
    elif p.startswith('SLEEP'): cat = 'SLEEP_POD'
    elif p.startswith('MICROCHIP'): cat = 'MICROCHIP'
    elif p.startswith('ROBOT'): cat = 'ROBOT'
    elif p.startswith('UV_'): cat = 'UV_VISOR'
    elif p.startswith('TRANSLATOR'): cat = 'TRANSLATOR'
    elif p.startswith('PANEL'): cat = 'PANEL'
    elif p.startswith('OXYGEN'): cat = 'OXYGEN_SHAKE'
    elif p.startswith('PEBBLES'): cat = 'PEBBLES'
    elif p.startswith('SNACKPACK'): cat = 'SNACKPACK'
    cats.setdefault(cat, {'theo': 0, 'bt': 0})
    cats[cat]['theo'] += v
    cats[cat]['bt'] += prod_pnl.get(p, 0)

print(f"{'Category':14}{'BT':>10}{'Theo':>12}{'Capture %':>10}{'Headroom':>12}")
print('-' * 58)
for cat in sorted(cats, key=lambda c: -cats[c]['theo']):
    bt = cats[cat]['bt']; theo_v = cats[cat]['theo']
    pct = bt / theo_v * 100
    headroom = theo_v - bt
    print(f"{cat:14}{bt:>10.0f}{theo_v:>12.0f}{pct:>9.2f}%{headroom:>12.0f}")
total_bt = sum(c['bt'] for c in cats.values())
total_theo = sum(c['theo'] for c in cats.values())
print('-' * 58)
print(f"{'TOTAL':14}{total_bt:>10.0f}{total_theo:>12.0f}{total_bt/total_theo*100:>9.2f}%{total_theo-total_bt:>12.0f}")

# Per-product top headroom
print()
print("Top per-product theoretical headroom (theo - BT):")
print(f"  {'product':30}{'BT':>10}{'Theo':>12}{'Headroom':>12}{'Capture %':>10}")
prod_data = [(p, prod_pnl.get(p, 0), theo[p]) for p in theo]
prod_data.sort(key=lambda x: -(x[2] - x[1]))
for p, bt, t in prod_data[:15]:
    pct = bt / t * 100
    print(f"  {p:30}{bt:>10.0f}{t:>12.0f}{t-bt:>12.0f}{pct:>9.2f}%")
