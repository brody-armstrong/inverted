"""Stage 4: write the readiness passport page into the site repo (inverted/passport/index.html).
Usage: python -I page.py <model_out_dir> <db_path> <site_repo>
"""
import sys, os, json, html, hashlib
import numpy as np, pandas as pd
import duckdb

OUT, DB, SITE = sys.argv[1:4]
# When the commits first reached GitHub: the only timestamp anyone else can check. Filled in after the push.
_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/passport/first_public.json")
# {"github_time": "...", "github_record": GitHub's own push record,
#  "wayback_prereg": Wayback snapshot of PREREGISTRATION.md as committed, "wayback_hash": Wayback snapshot of this page with the 2027 hash}
FP = json.load(open(_fp)) if os.path.exists(_fp) else None
def _a(url, text): return f'<a class="src" href="{html.escape(url)}">{html.escape(text)}</a>'
if FP:
    FIRST_PUBLIC = f"{html.escape(FP['github_time'])}, as recorded by GitHub"
    RECORDS = (f'<p class="small">Records anyone can check: {_a(FP["github_record"], "GitHub push record")} · '
               f'{_a(FP["wayback_prereg"], "pre-registration as committed (Wayback Machine, 18:25 PT)")} · '
               f'{_a(FP["wayback_hash"], "this page with the 2027 hash (Wayback Machine, 18:11 PT)")}</p>')
else:
    FIRST_PUBLIC, RECORDS = None, ""
DISPLAY = {"Xj5YjkWAqb": "Lucas Fernandez-Kim"}   # keyed by ASA player_id: the club's spelling differs from ASA's
ORDER = {"Strikers": 0, "Wingers": 1, "Midfield": 2, "Fullbacks": 3, "Centre-backs": 4}

pl = pd.read_csv(f"{OUT}/passport_players.csv")
kp = pd.read_csv(f"{OUT}/passport_kpis.csv")
tl = pd.read_csv(f"{OUT}/passport_timeline.csv")
rel = pd.read_csv(f"{OUT}/reliability.csv")
bt = pd.read_csv(f"{OUT}/backtest_a.csv")
cal = pd.read_csv(f"{OUT}/calibration_test.csv")
sc = pd.read_csv(f"{OUT}/kpi_scale_2026.csv")
pred = pd.read_csv(f"{OUT}/predictions_2027.csv")
PRED_HASH = hashlib.sha256(open(f"{OUT}/predictions_2027.csv", "rb").read()).hexdigest()
ex = pd.read_csv(f"{OUT}/backtest_a_zero_mls_exploratory.csv").set_index("model")
summ = json.load(open(f"{OUT}/summary.json"))
con = duckdb.connect(DB, read_only=True)
contracts = con.execute("SELECT * FROM contracts").df()
con.close()
ctr = contracts.set_index("player_id")

pl["o"] = pl.group.map(ORDER)
pl = pl.sort_values(["o", "p_two"], ascending=[True, False])
relmap = rel.set_index("label")
players = []
for r in pl.itertuples():
    k = kp[kp.player_id == r.player_id]
    t = tl[tl.player_id == r.player_id]
    kk = []
    for x in k.itertuples():
        s = sc[(sc.kpi == x.kpi) & (sc.group == r.group)].iloc[0]
        kk.append(dict(k=x.kpi, ok=bool(x.reliable), rel=float(relmap.loc[x.kpi, "r_full_sb"]), raw=x.raw_per96, est=x.shrunk_per96,
                       lo=x.lo90, hi=x.hi90, pct=None if pd.isna(x.raw_pct) else int(round(x.raw_pct)),
                       apct=None if pd.isna(x.age_adj_pct) else int(round(x.age_adj_pct)), s0=s.p05, s1=s.p95, sm=s.p50))
    c = ctr.loc[r.player_id] if r.player_id in ctr.index else None
    players.append(dict(id=r.player_id, n=DISPLAY.get(r.player_id, r.name), g=r.group, p=r.position, a=round(r.age, 1),
                        np=int(r.minutes), mls=int(r.mls_S), cum=int(r.mls_cum),
                        pt=r.p_two, plo=r.p_two_lo, phi=r.p_two_hi, p1=r.p2027, p1lo=r.p2027_lo, p1hi=r.p2027_hi,
                        proj=round(r.proj_mls_g96, 3), psrc=r.proj_source, k=kk,
                        st=None if c is None else c.event, su=None if c is None else c.source_url,
                        t=[dict(s=int(x.season), tier=x.tier, m=int(x.minutes)) for x in t.itertuples()]))
data = json.dumps(players, ensure_ascii=False, separators=(",", ":"))

b = bt.set_index("model")
NAMES = {"new": "Readiness model", "raw_pct": "Raw NEXT Pro percentile", "minutes": "Minutes only"}
best_base = b.drop("new").logloss.idxmin()
new_ok = summ["new_passes"]
ob = summ["outcome_b"]
fail_kpis = rel[~rel.passes]
stamp = pred.written_utc.iloc[0]


def pct(x): return f"{100 * x:.0f}%"


rows_bt = "".join(f"<tr><td>{NAMES[m]}</td><td class='num'>{b.loc[m, 'brier']:.3f}</td><td class='num'>{b.loc[m, 'logloss']:.3f}</td>"
                  f"<td class='num'>{pct(b.loc[m, 'mean_pred'])}</td><td class='num'>{pct(b.loc[m, 'observed'])}</td></tr>"
                  for m in ["new", "raw_pct", "minutes"])
rows_rel = "".join(f"<tr><td>{x.label}</td><td class='num'>{x.r_full_sb:.2f}</td><td>{'Kept' if x.passes else 'Did not meet the reliability threshold'}</td></tr>"
                   for x in rel.itertuples())

# calibration chart (SVG)
W, H, P0 = 340, 300, 44
mx = 0.5
def X(v): return P0 + (W - P0 - 12) * min(v, mx) / mx
def Y(v): return H - 36 - (H - 36 - 12) * min(v, mx) / mx
col = {"new": "#1d5e47", "minutes": "#a0522d", "raw_pct": "#6b7280"}
dash = {"new": "", "minutes": "6 4", "raw_pct": "2 4"}
svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:420px" role="img" aria-labelledby="calt"><title id="calt">Calibration on the 2024 test cohort: predicted against observed share reaching 500 MLS minutes, five bins per model</title>']
for v in (0, .1, .2, .3, .4, .5):
    svg.append(f'<line x1="{X(v):.1f}" y1="{Y(0):.1f}" x2="{X(v):.1f}" y2="{Y(mx):.1f}" stroke="#eceef0"/><line x1="{X(0):.1f}" y1="{Y(v):.1f}" x2="{X(mx):.1f}" y2="{Y(v):.1f}" stroke="#eceef0"/>')
    svg.append(f'<text x="{X(v):.1f}" y="{H - 20}" font-size="11" text-anchor="middle" fill="#545a60">{int(v * 100)}%</text><text x="{P0 - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#545a60">{int(v * 100)}%</text>')
svg.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(mx)}" y2="{Y(mx)}" stroke="#9aa0a6" stroke-dasharray="3 3"/>')
for m in ["raw_pct", "minutes", "new"]:
    c = cal[cal.model == m].sort_values("bin")
    pts = " ".join(f"{X(a):.1f},{Y(o):.1f}" for a, o in zip(c.mean_pred, c.observed))
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{col[m]}" stroke-width="2" stroke-dasharray="{dash[m]}"/>')
    svg += [f'<circle cx="{X(a):.1f}" cy="{Y(o):.1f}" r="3.5" fill="{col[m]}"/>' for a, o in zip(c.mean_pred, c.observed)]
svg.append(f'<text x="{(X(0) + X(mx)) / 2:.0f}" y="{H - 4}" font-size="12" text-anchor="middle" fill="#1b1d1f">Predicted</text>')
svg.append(f'<text x="12" y="{(Y(0) + Y(mx)) / 2:.0f}" font-size="12" text-anchor="middle" fill="#1b1d1f" transform="rotate(-90 12 {(Y(0) + Y(mx)) / 2:.0f})">Observed</text></svg>')
legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px"><svg width="26" height="8" aria-hidden="true"><line x1="0" y1="4" x2="26" y2="4" stroke="{col[m]}" stroke-width="2" stroke-dasharray="{dash[m]}"/></svg>{NAMES[m]}</span>' for m in ["new", "minutes", "raw_pct"])

fk = fail_kpis.iloc[0] if len(fail_kpis) else None
promoted = [p for p in players if p["st"] and "first team" in p["st"].lower()]
rest = sorted([p for p in players if p not in promoted], key=lambda p: -p["pt"])
default_id = max([p for p in players if p["cum"] == 0], key=lambda p: p["pt"])["id"]
def fmt(p): return f"{html.escape(p['n'])} {pct(p['pt'])} ({pct(p['plo'])}–{pct(p['phi'])})"
leaving = [p for p in rest if p["st"] and "will not return" in p["st"]]
nxt = [p for p in rest if p not in leaving][:3]
lead_bits = ["The group to watch is " + "; ".join(fmt(p) for p in nxt) + "."
             + ("" if not leaving else " " + "; ".join(f"{html.escape(p['n'])} is at {pct(p['pt'])} but is out of contract and leaving" for p in leaving) + ".")
             + f" Everyone else is between {pct(min(p['pt'] for p in rest))} and {pct(sorted(p['pt'] for p in rest if p not in nxt + leaving)[-1])}, against a {pct(b.loc['new', 'observed'])} base rate."]
for p in promoted:
    lead_bits.append(f"At the top, the model agrees with the club: it rates {html.escape(p['n'])} at {pct(p['pt'])}, mostly because of the {p['mls']} MLS minutes he already had in 2026, and the club moved him to the first team in September.")
lead_html = " ".join(lead_bits)
exn, exr, exm = ex.loc["new"], ex.loc["raw_pct"], ex.loc["minutes"]
_lo, _hi = [float(v) for v in str(exr.ll_diff_vs_new_90ci).strip("[]").split(",")]
ci_txt = f"{_lo:.3f} to {_hi:+.3f}".replace("-", "−")
ex_rows = "".join(f"<tr><td>{NAMES[m]}</td><td class='num'>{ex.loc[m, 'brier']:.3f}</td><td class='num'>{ex.loc[m, 'logloss']:.3f}</td>"
                  f"<td class='num'>{pct(ex.loc[m, 'mean_pred'])}</td><td class='num'>{pct(ex.loc[m, 'observed'])}</td></tr>" for m in ["new", "raw_pct", "minutes"])

verdict_a = (f"On the 2024 test cohort ({int(b.loc['new', 'n'])} players), the readiness model had lower Brier score and log loss than both simpler baselines, so the page uses it."
             if new_ok else
             f"On the 2024 test cohort, the readiness model did not beat both simpler baselines, so it is not supported; the page shows the {NAMES[best_base].lower()} baseline instead.")
verdict_b = ("The new projection of MLS g+ per 96 was supported." if ob.get("supported") else
             (f"Outcome (b) is not supported: for the {ob['n_test']} test players who reached 500 MLS minutes, the new projection of MLS g+ per 96 "
              f"was less accurate than the published pathway model (RMSE {ob['rmse_new']:.3f} vs {ob['rmse_pathway']:.3f}). "
              "The pathway model was fitted on move-ups that likely include these players, which favours it. The page keeps the pathway projection."
              if "rmse_new" in ob else "Outcome (b): too few test players reached 500 MLS minutes to judge it."))

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Timbers2: who is ready for first-team minutes? · Brody Armstrong</title>
<meta name="description" content="A pre-registered, backtested readiness passport for Portland Timbers 2: each regular's chance of reaching 500 MLS minutes over 2027–28 with an interval, and every KPI as a shrunk estimate with a 90% interval and an age-adjusted percentile.">
<link rel="stylesheet" href="../../assets/style.css">
<style>
tbody tr{{cursor:pointer}} tbody tr:hover,tbody tr[aria-selected="true"]{{background:var(--soft)}}
td button.pick{{all:unset;cursor:pointer;color:var(--accent);font-weight:600;min-height:44px;display:inline-flex;align-items:center}}
td button.pick:focus-visible{{outline:3px solid #2f7dd1;outline-offset:2px}}
.pbar{{display:inline-block;width:90px;height:10px;background:var(--soft);border-radius:5px;position:relative;vertical-align:middle;margin-right:8px}}
.pbar i{{position:absolute;top:0;bottom:0;background:#9fc5b6;border-radius:5px}} .pbar b{{position:absolute;top:-3px;width:3px;height:16px;background:var(--accent);border-radius:2px}}
.pcard{{border:1px solid var(--line);border-radius:10px;padding:20px;margin-top:18px}}
.pcard h3{{font-size:22px}} .pcard h4{{margin:16px 0 8px}}
.grid2{{display:grid;gap:18px;grid-template-columns:1fr}} @media (min-width:880px){{.grid2{{grid-template-columns:5fr 6fr}}}}
.tl{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px}} .tl div{{border:1px solid var(--line);border-radius:8px;padding:8px;font-size:13px}} .tl b{{display:block;font-size:14px}}
@media (max-width:520px){{.tl{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}
.big{{font-size:30px;font-weight:700;line-height:1.1}}
.kpi{{display:grid;grid-template-columns:118px 1fr;gap:4px 10px;align-items:center;padding:8px 0;border-top:1px solid var(--line);font-size:14px}}
.kpi:first-of-type{{border-top:0}} .kpi .meta{{grid-column:2;color:var(--muted);font-size:13px}}
.kpi.off{{color:var(--muted)}} .kpi.off .lab{{text-decoration:line-through dotted}}
.slots{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}} .slots div{{border:1.5px dashed #9aa0a6;border-radius:8px;padding:10px;font-size:13px;color:var(--muted)}} .slots b{{display:block;color:var(--ink)}}
.src{{display:inline-flex;align-items:center;min-height:44px}}
.note{{background:var(--soft);border-radius:8px;padding:10px 12px;font-size:14px}}
</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-head"><div class="wrap">
  <a class="brand" href="../../">Brody Armstrong</a>
  <nav class="site-nav" aria-label="Main"><a class="opt" href="../">Research</a><a class="opt" href="../../#contact">Contact</a><a class="cta" href="../simulator/">Open simulator</a></nav>
</div></header>
<main id="main">
<div class="wrap"><nav class="crumbs" aria-label="Breadcrumb"><ol><li><a href="../../">Home</a></li><li><a href="../">Research</a></li><li aria-current="page">Timbers2 readiness</li></ol></nav></div>

<section class="hero" style="padding-top:8px"><div class="wrap measure">
  <p class="kicker">Prototype · public data · pre-registered and backtested</p>
  <h1>Timbers2: who is ready for first-team minutes, and how sure are we?</h1>
  <p class="lead">For each 2026 Timbers2 regular: the chance of reaching 500 MLS minutes over 2027–28, with a 90% interval, and every KPI as an estimate shrunk toward the position average, with its own interval and an age-adjusted percentile.</p>
  <div class="actions"><a class="btn" href="#squad">See the squad</a><a class="btn ghost" href="#backtest">Has this worked before?</a></div>
</div></section>

<div class="wrap"><ul class="facts" aria-label="Key results">
  <li><strong>{b.loc['new', 'logloss']:.3f} vs {b.loc[best_base, 'logloss']:.3f}</strong><span>Pre-registered test, all 2024 regulars: log loss, readiness model vs best simpler baseline (lower is better)</span></li>
  <li><strong>{exn.logloss:.3f} vs {min(exr.logloss, exm.logloss):.3f}</strong><span>Exploratory: players with no MLS minutes yet; the edge over the raw percentile is within noise</span></li>
  <li><strong>{len(rel) - len(fail_kpis)} of {len(rel)}</strong><span>KPIs met the reliability threshold{'' if fk is None else ' (' + fk.label + ' did not)'}</span></li>
  <li><strong>{'Supported' if ob.get('supported') else 'Not supported'}</strong><span>New MLS g+ projection vs the published pathway model</span></li>
</ul></div>

<div class="wrap">
<section class="part" id="squad" aria-labelledby="sq">
  <h2 id="sq">The squad</h2>
  <p>{lead_html}</p>
  <p class="small">2026 MLS NEXT Pro regulars (900+ regular-season minutes for Timbers2). Readiness is the chance of reaching 500 MLS minutes in total over 2027 and 2028; the shaded bar is the 90% interval and the mark is the estimate. It partly measures whether coaches pick a player, not only ability. Roster status is from the club's end-of-season announcement and is shown for context only: no model uses it. Select a player to open the passport.</p>
  <div class="tablebox"><table style="min-width:780px">
    <thead><tr><th>Player</th><th class="num">Age</th><th>Pos</th><th class="num">NEXT Pro min</th><th class="num">MLS min 2026</th><th>Readiness 2027–28 (90% interval)</th><th>Roster status</th></tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
  <div class="pcard" id="card" aria-live="polite"></div>
</section>

<section class="part" id="backtest" aria-labelledby="bt">
  <h2 id="bt">Has this worked before?</h2>
  <p>The rules were written and committed locally at 16:23 PT on 18 November 2026, before any outcome data was fetched; {"first public when pushed to GitHub on " + FIRST_PUBLIC if FIRST_PUBLIC else "first public when pushed to GitHub (time added after the push)"}. A local commit's date is set by whoever makes it, so the push is the first time anyone else can check, and it came after the outcome data was fetched. The rules: train on the 2022 and 2023 NEXT Pro cohorts, test on 2024 (outcomes through 2026), and use the new model only if it beats both simpler baselines on both scores.</p>
  {RECORDS}
  <p><strong>{verdict_a}</strong></p>
  <div class="tablebox"><table>
    <thead><tr><th>Model (2024 test cohort)</th><th class="num">Brier</th><th class="num">Log loss</th><th class="num">Mean predicted</th><th class="num">Observed</th></tr></thead>
    <tbody>{rows_bt}</tbody></table></div>
  <figure class="fig">{''.join(svg)}<figcaption>{legend}<br>Calibration on the test cohort: each point is a fifth of players, by predicted chance. Points on the dashed diagonal mean the predicted chance matched what happened.</figcaption></figure>
  <h3 style="margin-top:18px">Exploratory: players with no MLS minutes yet</h3>
  <p>This check was not pre-registered; it was added after the backtest. The model uses MLS minutes a player has already had, so some of its advantage could simply be that players already in MLS stay in MLS. Among the {int(exn.n)} test players with no MLS minutes at all ({int(exn.events)} of them reached 500), the readiness model still scored best on both measures. The margin is smaller: against minutes alone the 90% interval for the log-loss difference excludes zero; against the raw NEXT Pro percentile it does not ({ci_txt}). For these players the model is ahead of the simpler baselines, but not clearly ahead of the raw percentile.</p>
  <div class="tablebox"><table>
    <thead><tr><th>Model (no prior MLS minutes, n={int(exn.n)})</th><th class="num">Brier</th><th class="num">Log loss</th><th class="num">Mean predicted</th><th class="num">Observed</th></tr></thead>
    <tbody>{ex_rows}</tbody></table></div>
  <p>{verdict_b}</p>
  <h3 style="margin-top:18px">Which KPIs met the reliability threshold?</h3>
  <p class="small">Split-half reliability: odd against even matches within a season (2022–23, players with 450+ minutes in each half), corrected to a full season. A KPI is kept if it reaches 0.40; one that does not is shown greyed out and is not used in any model.</p>
  <div class="tablebox"><table><thead><tr><th>KPI</th><th class="num">Reliability</th><th>Result</th></tr></thead><tbody>{rows_rel}</tbody></table></div>
</section>

<section class="part" aria-labelledby="p27">
  <h2 id="p27">Predictions for 2027, written in advance</h2>
  <p>On {stamp} (UTC), before the 2027 season, the model rated every MLS NEXT Pro regular of 2026 with fewer than 500 career MLS minutes ({len(pred)} players) on their chance of reaching 500 MLS minutes in 2027. Many of them are 16 or 17, so the list is kept private for now. Only its SHA-256 fingerprint is published here; any change to the list, however small, would change it. The full list will be published when it is scored after the 2027 season. The 14 Timbers2 regulars' 2027 chances are on their passports above.</p>
  <p style="overflow-wrap:anywhere"><code>SHA-256 {PRED_HASH}</code></p>
</section>

<section class="part" aria-labelledby="how">
  <h2 id="how">How it is built</h2>
  <ul>
    <li><strong>Data:</strong> American Soccer Analysis public API, MLS NEXT Pro match by match (2022–2026, regular season) and MLS minutes and g+ by season. Thanks to ASA for making it public. Birth dates are ASA's (checked against the club's announcements where they give one).</li>
    <li><strong>Shrinkage:</strong> each KPI is pulled toward the average for the player's position group, more strongly the noisier his own match-to-match record. The interval is a 90% empirical-Bayes interval.</li>
    <li><strong>Age-adjusted percentile:</strong> where the estimate sits among players of the same position and season after allowing for age.</li>
    <li><strong>Readiness model:</strong> a season-by-season model of the chance of crossing 500 MLS minutes, from age, shrunk g+, NEXT Pro minutes, MLS minutes already given and position. Intervals come from refitting it on 500 resamples of players, with each player's own g+ drawn from its interval.</li>
    <li><strong>Baselines:</strong> the raw NEXT Pro g+ percentile alone, and minutes alone.</li>
    <li><strong>MLS projection:</strong> {html.escape(players[0]['psrc'])}.</li>
  </ul>
  <details><summary>Limits</summary><ul>
    <li>Reaching 500 MLS minutes depends on coaching decisions, depth charts, injuries and the first team's needs, not only on the player.</li>
    <li>One test cohort (2024, {int(b.loc['new', 'n'])} players). The result could look different in another year.</li>
    <li>Public event-derived data only: no tracking, physical or medical data.</li>
    <li>The model learns from what happened to past players, including who clubs chose to promote. It describes those patterns; it does not show what causes readiness.</li>
    <li>Birth dates come from ASA rather than official rosters, which do not publish them in a usable form; three were checked against club announcements and matched.</li>
  </ul></details>
</section>
</div>
</main>
<footer class="site-foot"><div class="wrap">
  <p style="margin:0"><strong>Get in touch:</strong> <a href="mailto:brodyarmstrong123@gmail.com">brodyarmstrong123@gmail.com</a></p>
  <nav aria-label="Elsewhere"><a href="https://www.linkedin.com/in/brody-armstrong-0a905b2a8">LinkedIn</a><a href="https://github.com/brody-armstrong">GitHub</a><a href="https://github.com/brody-armstrong/inverted">Project code</a></nav>
</div></footer>
<script>
const PL = {data};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const pc = x => Math.round(100*x) + "%";
const bar = (p,lo,hi) => `<span class="pbar" aria-hidden="true"><i style="left:${{100*lo}}%;width:${{Math.max(1,100*(hi-lo))}}%"></i><b style="left:calc(${{100*p}}% - 1px)"></b></span>`;
const rows = document.getElementById("rows"), card = document.getElementById("card");
rows.innerHTML = PL.map((p,i)=>`<tr data-i="${{i}}" aria-selected="false"><td><button type="button" class="pick" data-i="${{i}}">${{esc(p.n)}}</button></td>
 <td class="num">${{p.a}}</td><td>${{esc(p.p)}}</td><td class="num">${{p.np.toLocaleString()}}</td><td class="num">${{p.mls ? p.mls.toLocaleString() : "—"}}</td>
 <td>${{bar(p.pt,p.plo,p.phi)}}<span>${{pc(p.pt)}} <small>(${{pc(p.plo)}}–${{pc(p.phi)}})</small></span></td><td class="small">${{esc(p.st || "Not in the club announcement")}}</td></tr>`).join("");
function kpiRow(k){{
  const span = (k.s1 - k.s0) || 1, pos = v => Math.max(0, Math.min(100, 100*(v - k.s0)/span));
  if(!k.ok) return `<div class="kpi off"><span class="lab">${{esc(k.k)}}</span><span>Raw ${{k.raw.toFixed(3)}} per 96</span><span class="meta">Did not meet the reliability threshold (split-half reliability ${{k.rel.toFixed(2)}}; threshold 0.40). Not used in any model.</span></div>`;
  return `<div class="kpi"><span class="lab">${{esc(k.k)}}</span>
   <span><svg viewBox="0 0 200 18" width="100%" height="18" role="img" aria-label="estimate ${{k.est.toFixed(3)}}, 90% interval ${{k.lo.toFixed(3)}} to ${{k.hi.toFixed(3)}}">
   <line x1="0" y1="9" x2="200" y2="9" stroke="#d9dcdf" stroke-width="2"/><line x1="${{2*pos(k.sm)}}" y1="2" x2="${{2*pos(k.sm)}}" y2="16" stroke="#9aa0a6"/>
   <line x1="${{2*pos(k.lo)}}" y1="9" x2="${{2*pos(k.hi)}}" y2="9" stroke="#1d5e47" stroke-width="4" stroke-linecap="round" opacity=".45"/>
   <circle cx="${{2*pos(k.est)}}" cy="9" r="5" fill="#1d5e47"/></svg></span>
   <span class="meta">${{k.est.toFixed(3)}} per 96 (90%: ${{k.lo.toFixed(3)}} to ${{k.hi.toFixed(3)}}) · raw ${{k.raw.toFixed(3)}} · percentile ${{k.pct ?? "—"}} · age-adjusted ${{k.apct ?? "—"}}</span></div>`;
}}
function show(i){{
  const p = PL[i];
  [...rows.children].forEach(r=>r.setAttribute("aria-selected", r.dataset.i==i));
  const yrs = [2022,2023,2024,2025,2026].map(s=>{{const n=p.t.find(x=>x.s===s&&x.tier==="MLS NEXT Pro"), m=p.t.find(x=>x.s===s&&x.tier==="MLS");
    return `<div><b>${{s}}</b>NP ${{n?n.m.toLocaleString():"—"}}<br>MLS ${{m?m.m.toLocaleString():"—"}}</div>`;}}).join("");
  card.innerHTML = `<h3>${{esc(p.n)}}</h3><p class="small">${{esc(p.g)}} · ${{esc(p.p)}} · age ${{p.a}} (1 July 2026) · ${{p.cum}} career MLS minutes</p>
  <div class="grid2"><div>
    <h4>Readiness</h4><p class="big">${{pc(p.pt)}}</p><p>chance of 500+ MLS minutes over 2027–28 (90% interval ${{pc(p.plo)}}–${{pc(p.phi)}}). For 2027 alone: ${{pc(p.p1)}} (${{pc(p.p1lo)}}–${{pc(p.p1hi)}}).</p>
    <p class="note">${{p.st && p.st.toLowerCase().includes("first team") ? "The club moved him to the first team in September; this estimate agrees with that decision and comes mostly from MLS minutes already given, so it adds little. " : ""}}This partly measures coach selection. It counts MLS minutes already given${{p.mls?` (${{p.mls}} in 2026)`:""}}, which weigh heavily when a player is already in the first-team picture.</p>
    <h4>Minutes by season</h4><div class="tl">${{yrs}}</div>
    <h4>MLS projection</h4><p>Projected MLS g+ ${{p.proj >= 0 ? "+" : ""}}${{p.proj}} per 96 (${{esc(p.psrc)}}).</p>
    <h4>Roster status</h4><p>${{p.st ? esc(p.st) + ` <a class=\"src\" href="${{esc(p.su)}}">(club announcement)</a>` : "Not mentioned in the club's end-of-season announcement."}}</p>
  </div><div>
    <h4>2026 KPIs vs MLS NEXT Pro ${{esc(p.g.toLowerCase())}}</h4>
    <p class="small">Dot: shrunk estimate; band: 90% interval; grey tick: the position-group median; the line spans the group's 5th to 95th percentile.</p>
    ${{p.k.map(kpiRow).join("")}}
  </div></div>
  <h4>Club data slots</h4><div class="slots">${{["GPS high-speed running per 90","Sprints per 90","Acute:chronic workload","Wellness (7-day)","Maturation (PHV status)","Development plan focus areas","Video clips","Coach evaluation"].map(s=>`<div><b>${{s}}</b>From club systems</div>`).join("")}}</div>`;
}}
rows.addEventListener("click", e=>{{const t=e.target.closest("[data-i]"); if(t) show(+t.dataset.i);}});
show(Math.max(0, PL.findIndex(p=>p.id==="{default_id}")));
</script>
</body>
</html>
"""
os.makedirs(f"{SITE}/inverted/passport", exist_ok=True)
open(f"{SITE}/inverted/passport/index.html", "w", encoding="utf-8").write(page)
print("page written:", len(page), "bytes;", len(players), "players")
