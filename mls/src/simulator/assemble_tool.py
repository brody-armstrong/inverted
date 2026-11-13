"""Assemble the Roster Investment Simulator page: src/simulator/tool_template.html + out/tool/data.json -> out/tool/index.html.

Run after:  python3 src/simulator/tool_data.py && python3 src/simulator/roster_2027.py
Every placeholder is filled from data.json (its "meta" block), so the page cannot disagree with its data.
"""
import json

d = json.load(open("out/tool/data.json"))
m = d["meta"]
fill = {
    "__DATA__": json.dumps(d, separators=(",", ":")),
    "__MATCHES__": f"{m['matches']:,}",
    "__SEASONS__": m["seasons"],
    "__BT_MODEL__": f"{m['backtest_spearman_model']:.2f}",
    "__BT_NAIVE__": f"{m['backtest_spearman_naive']:.2f}",
    "__F25__": f"{m['futures']['2025']['model_vs_market']:.2f}",
    "__F26__": f"{m['futures']['2026']['model_vs_market']:.2g}",
    "__RMSE__": f"{round(m['rmse_pts'])}",
    "__ROSTER_DATE__": m["roster_date"],
}
t = open("src/simulator/tool_template.html").read()
for k, v in fill.items():
    assert t.count(k) >= 1, k
    t = t.replace(k, v)
open("out/tool/index.html", "w").write(t)
print("wrote out/tool/index.html")
