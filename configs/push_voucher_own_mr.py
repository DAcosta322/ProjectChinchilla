"""Enable voucher OWN_MR — own mean-reversion on alpha residual.

Currently OWN_MR=0 (vouchers don't MR on their own residual). Theoretical max
shows huge unrealized PnL on ATM vouchers — maybe their residuals oscillate
enough that own-MR captures it.
"""

ALGO = "algorithms/ROUND_4/round_4_max_scale.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "_VevBase.OWN_MR": [0.0, 0.5, 1.0, 2.0, 5.0],
}

TRACK = ["VEV_5000", "VEV_5100", "VEV_5200"]
