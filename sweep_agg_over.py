"""Sweep AGG_OVERSHOOT - finer."""
ALGO = "algorithms/round_4_botflow.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "HydrogelParams.AGG_OVERSHOOT": [0, 2, 5, 10, 20],
    "VelvetParams.AGG_OVERSHOOT":   [0, 2, 5, 10, 20],
    "_VevBase.AGG_OVERSHOOT":       [0, 2, 5, 10, 20],
}

def SKIP(p):
    h = p["HydrogelParams.AGG_OVERSHOOT"]
    v = p["VelvetParams.AGG_OVERSHOOT"]
    e = p["_VevBase.AGG_OVERSHOOT"]
    return not (h == v == e)

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
