"""Sweep the event-based VELVET bot signal (Mark 67 buy / Mark 49 sell EWMA).

Lead-lag rationale: Mark 67 prints precede +2.0 mid moves at h=1, +1.6 at h=100,
and Mark 49 mirrors. Half-life 25-100 ticks should capture this window.
"""

ALGO = "algorithms/round_4_botflow.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "BotFlowParams.VELVET_EVENT_GAIN": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    "BotFlowParams.HALF_LIFE":          [25, 50, 100, 200],
}

TRACK = ["VELVETFRUIT_EXTRACT"]
