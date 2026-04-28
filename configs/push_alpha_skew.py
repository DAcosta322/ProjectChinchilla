"""Push ALPHA_SPAN (voucher fair adaptation) and INV_SKEW (quote tightness)."""

ALGO = "algorithms/ROUND_4/round_4_max_scale.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

# We push ALPHA_SPAN globally on _VevBase by patching every voucher class.
GRID = {
    "_VevBase.ALPHA_SPAN":      [500, 1000, 2000, 5000, 10000],
    "HydrogelParams.INV_SKEW":  [10, 15, 20],
    "VelvetParams.INV_SKEW":    [2, 3, 5],
}

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
