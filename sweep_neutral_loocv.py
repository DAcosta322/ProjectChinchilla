"""LOOCV verification of top neutral configs."""
ALGO = "algorithms/round_3_voucher.py"
ROUND = 3
DAYS = [0, 1, 2]
MODE = "bt"

# Just the top candidates from prior sweep + baseline
GRID = {
    "HydrogelParams.NEUTRAL_BAND": [0, 5, 10, 20],
    "HydrogelParams.NEUTRAL_GAIN": [0.0, 0.05, 0.1, 0.2],
    "VelvetParams.NEUTRAL_BAND":   [0, 3, 5],
    "VelvetParams.NEUTRAL_GAIN":   [0.0, 0.05, 0.1],
}

def SKIP(p):
    h_off = p["HydrogelParams.NEUTRAL_BAND"] == 0 or p["HydrogelParams.NEUTRAL_GAIN"] == 0
    v_off = p["VelvetParams.NEUTRAL_BAND"] == 0 or p["VelvetParams.NEUTRAL_GAIN"] == 0
    # Disallow inconsistent (band>0 gain=0 or band=0 gain>0)
    if (p["HydrogelParams.NEUTRAL_BAND"] == 0) != (p["HydrogelParams.NEUTRAL_GAIN"] == 0):
        return True
    if (p["VelvetParams.NEUTRAL_BAND"] == 0) != (p["VelvetParams.NEUTRAL_GAIN"] == 0):
        return True
    return False

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
