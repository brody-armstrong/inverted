"""Fill src/playoffs/page_template.html from the pipeline outputs -> out/playoffs/index.html.

Every number on the page comes from out/playoffs/bracket.json, results/playoffs_2026.csv,
results/playoffs_backtest.json, results/playoffs_market_check.json and out/playoffs/validate_js.json.
Usage: python3 src/playoffs/assemble_page.py
"""
import json, re, pathlib
import pandas as pd

B = json.load(open("out/playoffs/bracket.json"))
D = pd.read_csv("results/playoffs_2026.csv").set_index("abbr")
BT = json.load(open("results/playoffs_backtest.json"))
MK = json.load(open("results/playoffs_market_check.json"))
JS = json.load(open("out/playoffs/validate_js.json"))
assert JS["pass"]

SHORT = {"NSH": "Nashville", "MIA": "Inter Miami", "NER": "New England", "CHI": "Chicago", "CLT": "Charlotte", "PHI": "Philadelphia",
         "CLB": "Columbus", "ORL": "Orlando", "CIN": "Cincinnati", "VAN": "Vancouver", "FCD": "FC Dallas", "STL": "St. Louis",
         "SJE": "San Jose", "HOU": "Houston", "LAFC": "LAFC", "POR": "Portland", "SEA": "Seattle", "RSL": "Real Salt Lake"}
for k, t in B["teams"].items():
    t["short"] = SHORT[k]
B["market"] = MK["pct"]
p = lambda x: str(round(100 * x))
top = D.p_champ.idxmax()
fav_n = round(BT["tie_favourite_advanced"]["market_hl90"] * BT["ties"])
vals = {
    "matches": B["meta"]["matches"], "draws": B["meta"]["draws"], "home_mult": f'{B["meta"]["home_goals_mult"]:.2f}'.rstrip("0"),
    "p_top": p(D.p_champ[top]), "top_name": SHORT[top], "p_fav_not": p(1 - D.p_champ[top]),
    "p_shield": p(D.p_champ["NSH"]), "nsh_pts": int(D.pts["NSH"]), "rho_market": f'{MK["spearman_model"]:.2f}',
    "p_van": p(D.p_champ["VAN"]), "pm_van": MK["pct"]["VAN"], "p_nsh": p(D.p_champ["NSH"]), "pm_nsh": MK["pct"]["NSH"],
    "p_mia": p(D.p_champ["MIA"]), "pm_mia": MK["pct"]["MIA"],
    "dc_nsh": p(D.p_champ_dc["NSH"]), "dc_van": p(D.p_champ_dc["VAN"]), "dc_mia": p(D.p_champ_dc["MIA"]),
    "bt_matches": BT["matches"], "bt_ties": BT["ties"],
    "ll_close": f'{BT["match_logloss"]["closing_odds"]:.3f}', "ll_model": f'{BT["match_logloss"]["market_hl90"]:.3f}',
    "ll_dc": f'{BT["match_logloss"]["results_dc"]:.3f}', "ll_base": f'{BT["match_logloss"]["base_rates"]:.3f}',
    "ll_hl60": f'{BT["match_logloss"]["market_hl60"]:.3f}',
    "fav_n": fav_n, "fav_pct": p(fav_n / BT["ties"]), "tie_ll": f'{BT["tie_logloss"]["market_hl90"]:.3f}',
    "coin_ll": f'{BT["tie_logloss_coinflip"]:.3f}', "js_diff": f'{100 * JS["max_abs_diff"]:.1f}',
    "data_json": json.dumps(B, separators=(",", ":"), ensure_ascii=False),
    "engine_js": re.sub(r"\nif \(typeof module.*\n?", "\n", pathlib.Path("src/playoffs/engine.js").read_text()),
}
html = pathlib.Path("src/playoffs/page_template.html").read_text()
for k, v in vals.items():
    html = html.replace("{{" + k + "}}", str(v))
left = re.findall(r"\{\{\w+\}\}", html)
assert not left, left
pathlib.Path("out/playoffs/index.html").write_text(html)
print("out/playoffs/index.html", len(html) // 1024, "KB |", {k: v for k, v in vals.items() if k not in ("data_json", "engine_js")})
