"""Fill src/playoffs/page_template.html from the pipeline outputs -> out/playoffs/index.html.

Every number on the page comes from out/playoffs/bracket.json, out/playoffs/profile_bracket.json, results/playoffs_2026.csv,
results/playoffs_profile_2026.csv, results/playoffs_backtest.json, results/playoffs_profile_backtest.json,
results/playoffs_profile_test.json, results/playoffs_market_check.json and out/playoffs/validate_js.json.
Usage: python3 src/playoffs/assemble_page.py
"""
import json, re, pathlib
import pandas as pd

B = json.load(open("out/playoffs/bracket.json"))
PB = json.load(open("out/playoffs/profile_bracket.json"))
D = pd.read_csv("results/playoffs_2026.csv").set_index("abbr")
PF = pd.read_csv("results/playoffs_profile_2026.csv").set_index("abbr")
BT = json.load(open("results/playoffs_backtest.json"))
PBT = json.load(open("results/playoffs_profile_backtest.json"))
PT = json.load(open("results/playoffs_profile_test.json"))
MK = json.load(open("results/playoffs_market_check.json"))
JS = json.load(open("out/playoffs/validate_js.json"))
assert JS["pass"] and not PT["supported"]   # page copy is written for the not-supported outcome

SHORT = {"NSH": "Nashville", "MIA": "Inter Miami", "NER": "New England", "CHI": "Chicago", "CLT": "Charlotte", "PHI": "Philadelphia",
         "CLB": "Columbus", "ORL": "Orlando", "CIN": "Cincinnati", "VAN": "Vancouver", "FCD": "FC Dallas", "STL": "St. Louis",
         "SJE": "San Jose", "HOU": "Houston", "LAFC": "LAFC", "POR": "Portland", "SEA": "Seattle", "RSL": "Real Salt Lake"}
for k, t in B["teams"].items():
    t["short"] = SHORT[k]

# places in the bracket where the two forecasts pick different teams
def slots(c):
    s = B["seeds"][c]; f = lambda i: s[i - 1]
    return [([f(8), f(9)], "r1"), ([f(1), f(8), f(9)], "sf"), ([f(4), f(5)], "sf"), ([f(2), f(7)], "sf"), ([f(3), f(6)], "sf"),
            ([f(1), f(8), f(9), f(4), f(5)], "cf"), ([f(2), f(7), f(3), f(6)], "cf"), (s, "final")]
allt = B["seeds"]["East"] + B["seeds"]["West"]
split = [(ts, k) for c in ("East", "West") for ts, k in slots(c)] + [(allt, "champ")]
n_split = sum(max(ts, key=lambda t: D.at[t, "p_" + k]) != max(ts, key=lambda t: PF.at[t, "p_" + k]) for ts, k in split)

data = {"teams": B["teams"], "seeds": B["seeds"], "market": MK["pct"],
        "lenses": {"market": {"probs": B["probs"], "draws": B["draws"]}, "profile": {"probs": PB["probs"], "draws": PB["draws"]}},
        "profile": {"contrib": PB["contrib"], "point": PB["point"]},
        "hyp": [{k: h[k] for k in ("label", "coef", "lo", "hi")} for h in PT["hypotheses"]]}
p = lambda x: str(round(100 * x))
top = D.p_champ.idxmax(); assert PF.p_champ.idxmax() == top
fav_n = round(BT["tie_favourite_advanced"]["market_hl90"] * BT["ties"])
vals = {
    "draws": B["meta"]["draws"], "home_mult": f'{B["meta"]["home_goals_mult"]:.2f}'.rstrip("0"),
    "p_top": p(D.p_champ[top]), "pp_top": p(PF.p_champ[top]), "top_name": SHORT[top], "n_split": n_split,
    "rho_market": f'{MK["spearman_model"]:.2f}', "rho_profile": f'{MK["spearman_profile"]:.2f}',
    "p_nsh": p(D.p_champ["NSH"]), "pp_nsh": p(PF.p_champ["NSH"]), "p_chi": p(D.p_champ["CHI"]), "pp_chi": p(PF.p_champ["CHI"]),
    "pf_chi_f": p(PF.p_final["CHI"]), "pf_nsh_f": p(PF.p_final["NSH"]), "pf_mia_f": p(PF.p_final["MIA"]), "mk_nsh_f": p(D.p_final["NSH"]),
    "mk_hou_sf": p(D.p_sf["HOU"]), "pf_sje_sf": p(PF.p_sf["SJE"]),
    "pt_matches": f'{PT["matches"]:,}', "pt_imp": PT["seasons_improved"],
    "bt_matches": BT["matches"], "bt_ties": BT["ties"],
    "ll_close": f'{BT["match_logloss"]["closing_odds"]:.3f}', "ll_model": f'{BT["match_logloss"]["market_hl90"]:.3f}',
    "ll_dc": f'{BT["match_logloss"]["results_dc"]:.3f}', "ll_base": f'{BT["match_logloss"]["base_rates"]:.3f}',
    "ll_hl60": f'{BT["match_logloss"]["market_hl60"]:.3f}', "fav_n": fav_n, "tie_ll": f'{BT["tie_logloss"]["market_hl90"]:.3f}',
    "coin_ll": f'{BT["tie_logloss_coinflip"]:.3f}', "js_diff": f'{100 * JS["max_abs_diff"]:.1f}',
    "pf_ll": f'{PBT["match_logloss"]:.3f}', "pf_tie_ll": f'{PBT["tie_logloss"]:.3f}', "pf_fav_n": PBT["tie_favourite_advanced_n"],
    "data_json": json.dumps(data, separators=(",", ":"), ensure_ascii=False),
    "engine_js": re.sub(r"\nif \(typeof module.*\n?", "\n", pathlib.Path("src/playoffs/engine.js").read_text()),
}
html = pathlib.Path("src/playoffs/page_template.html").read_text()
for k, v in vals.items():
    html = html.replace("{{" + k + "}}", str(v))
left = re.findall(r"\{\{\w+\}\}", html)
assert not left, left
pathlib.Path("out/playoffs/index.html").write_text(html)
print("out/playoffs/index.html", len(html) // 1024, "KB |", {k: v for k, v in vals.items() if k not in ("data_json", "engine_js")})
