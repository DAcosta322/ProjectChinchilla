import json
from pathlib import Path
subnum = "4470"

dd = "dump/" + subnum
jj = subnum + ".json"
ll = subnum + ".log"
d = Path(dd)
j = json.loads((d/jj).read_text())
log = json.loads((d/ll).read_text())

print("profit", j["profit"])
print("positions", j["positions"])
print("activities rows", len(j["activitiesLog"].splitlines()))

# trade history stats
trade = log["tradeHistory"]
print("trades", len(trade))
print("symbols", {s: sum(1 for t in trade if t["symbol"]==s) for s in {"TOMATOES","EMERALDS"}})