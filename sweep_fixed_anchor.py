"""Sweep FIXED_ANCHOR for HYD/VEL EMA initialization."""
ALGO = "algorithms/round_3_voucher.py"
ROUND = 3
DAYS = [0, 1, 2]
MODE = "bt"

GRID = {
    "HydrogelParams.FIXED_ANCHOR": [None, 9985, 9990, 9995, 10000],
    "VelvetParams.FIXED_ANCHOR":   [None, 5245, 5250, 5255, 5260],
}

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
