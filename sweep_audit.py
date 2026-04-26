"""Single-signal ablation audit via sweep.py.

Each row in the result table is "this one signal flipped off, everything else
default" — find the contribution of each individual signal to total PnL.

GRID has axes for every togglable signal. SKIP restricts the cartesian product
to (a) the all-default baseline and (b) exactly-one-off ablations.
"""
ALGO = "algorithms/round_4_botflow.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

# Each axis has [default, off]. Default=ON, off=0.
GRID = {
    "HydrogelParams.OFI_GAIN":         [1.0, 0.0],
    "HydrogelParams.PROFIT_DIST":      [40, 0],
    "HydrogelParams.VOL_SPAN":         [200, 0],
    "HydrogelParams.BOOST_GAIN":       [3.0, 0.0],
    "HydrogelParams.NEUTRAL_GAIN":     [0.05, 0.0],
    "HydrogelParams.TREND_STRENGTH":   [0.3, 0.0],
    "VelvetParams.OFI_GAIN":           [1.0, 0.0],
    "VelvetParams.PROFIT_DIST":        [15, 0],
    "VelvetParams.BOOST_GAIN":         [5.0, 0.0],
    "VelvetParams.NEUTRAL_GAIN":       [0.05, 0.0],
    "VelvetParams.TREND_STRENGTH":     [0.6, 0.0],
    "BotFlowParams.VELVET_EVENT_GAIN": [0.20, 0.0],
}

DEFAULTS = {k: v[0] for k, v in GRID.items()}

def SKIP(p):
    # count axes that differ from default
    diffs = sum(1 for k, v in p.items() if v != DEFAULTS[k])
    return diffs > 1   # keep only baseline (0 diffs) + each-singly-off (1 diff)

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
