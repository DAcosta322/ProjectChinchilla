"""Push scale + frequency knobs on max_scale algo."""

ALGO = "algorithms/ROUND_4/round_4_max_scale.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "HydrogelParams.MR_STRENGTH": [5, 7, 10],
    "VelvetParams.MR_STRENGTH":   [10, 15, 20],
    "HydrogelParams.PROFIT_DIST": [40, 0],   # 0 = decay off
    "VelvetParams.PROFIT_DIST":   [15, 0],
}

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
