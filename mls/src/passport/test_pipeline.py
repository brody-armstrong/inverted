"""pytest checks for each stage of the readiness passport pipeline.
Paths come from environment variables set by run.sh: PASSPORT_RAW, PASSPORT_DB, PASSPORT_OUT, SITE_REPO.
"""
import html
import os, glob, json, re
import pandas as pd
import duckdb
import pytest

RAW, DB, OUT, SITE = (os.environ[k] for k in ("PASSPORT_RAW", "PASSPORT_DB", "PASSPORT_OUT", "SITE_REPO"))


# ---- stage 1: fetch ----
def test_raw_files_present_and_under_cap():
    for s in range(2022, 2027):
        xg = glob.glob(f"{RAW}/mlsnp_players_xgoals_season_name-{s}_split_by_games-true_team_id-*.json")
        ga = glob.glob(f"{RAW}/mlsnp_players_goals-added_season_name-{s}_split_by_games-true_team_id-*.json")
        assert len(xg) >= 20 and len(xg) == len(ga), s
        for f in xg + ga:
            assert len(json.load(open(f))) < 10000, f   # ASA's row cap would mean truncation


# ---- stage 2: database ----
@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(DB, read_only=True)
    yield c
    c.close()


def test_tables(con):
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"players", "teams", "matches", "player_match", "mls_season", "contracts"} <= names


def test_player_match_integrity(con):
    dup = con.execute("SELECT COUNT(*) FROM (SELECT player_id, game_id FROM player_match GROUP BY 1,2 HAVING COUNT(*)>1)").fetchone()[0]
    assert dup == 0
    assert con.execute("SELECT COUNT(*) FROM player_match WHERE season IS NULL").fetchone()[0] == 0
    for ko, cap in ((False, 125), (True, 150)):   # playoff games can go to extra time
        lo, hi = con.execute("SELECT MIN(minutes), MAX(minutes) FROM player_match WHERE knockout = ?", [ko]).fetchone()
        assert lo >= 0 and hi <= cap


def test_team_minutes_plausible(con):
    # a team has at most 11 players on the field: per-team-match sums above 11 x the longest
    # possible match would mean duplicated rows
    for ko, cap in ((False, 125), (True, 150)):
        m = con.execute("SELECT MAX(s) FROM (SELECT team_id, game_id, SUM(minutes) s FROM player_match WHERE knockout = ? GROUP BY 1,2)", [ko]).fetchone()[0]
        assert m <= 11 * cap


def test_contracts_have_sources(con):
    rows = con.execute("SELECT player_id, source_url FROM contracts").fetchall()
    assert rows and all(r[0] and r[1].startswith("https://www.timbers.com/") for r in rows)


# ---- stage 3: models ----
def test_reliability():
    r = pd.read_csv(f"{OUT}/reliability.csv")
    assert len(r) == 8 and r.r_full_sb.between(-1, 1).all()
    assert (r.passes == (r.r_full_sb >= 0.40)).all()


def test_backtest_and_selection():
    b = pd.read_csv(f"{OUT}/backtest_a.csv").set_index("model")
    s = json.load(open(f"{OUT}/summary.json"))
    passes = all(b.loc["new", m] < b.loc[x, m] for x in ("raw_pct", "minutes") for m in ("brier", "logloss"))
    assert s["new_passes"] == passes
    assert s["selected"] == ("new" if passes else b.drop("new").logloss.idxmin())


def test_calibration_bins():
    c = pd.read_csv(f"{OUT}/calibration_test.csv")
    assert set(c.model) == {"new", "raw_pct", "minutes"} and c.groupby("model").n.sum().nunique() == 1


def test_probabilities_and_intervals():
    p = pd.read_csv(f"{OUT}/passport_players.csv")
    assert len(p) == 14
    for a, lo, hi in (("p_two", "p_two_lo", "p_two_hi"), ("p2027", "p2027_lo", "p2027_hi")):
        assert p[[a, lo, hi]].stack().between(0, 1).all()
        assert ((p[lo] <= p[a] + 1e-9) & (p[a] <= p[hi] + 1e-9)).all()


def test_kpi_intervals():
    k = pd.read_csv(f"{OUT}/passport_kpis.csv")
    assert ((k.lo90 <= k.shrunk_per96) & (k.shrunk_per96 <= k.hi90)).all()
    assert len(k) == 14 * 8


def test_predictions_2027_eligibility():
    p = pd.read_csv(f"{OUT}/predictions_2027.csv")
    assert p.written_utc.nunique() == 1 and (p.minutes >= 900).all() and (p.mls_cum < 500).all()
    assert p.p2027.between(0, 1).all()


# ---- stage 4: page ----
def test_page():
    h = open(f"{SITE}/inverted/passport/index.html", encoding="utf-8").read()
    data = json.loads(re.search(r"const PL = (\[.*?\]);\n", h, re.S).group(1))
    assert len(data) == 14
    assert "American Soccer Analysis" in h
    low = re.sub(r"<[^>]+>", " ", h).lower()
    assert " failed" not in low and "driven by" not in low
    assert "did not meet the reliability threshold" in low and "not reliable enough" not in low
    assert "not supported" in low and "exploratory" in low


def test_page_publishes_hash_not_list():
    import hashlib
    h = open(f"{SITE}/inverted/passport/index.html", encoding="utf-8").read()
    pred = pd.read_csv(f"{OUT}/predictions_2027.csv")
    digest = hashlib.sha256(open(f"{OUT}/predictions_2027.csv", "rb").read()).hexdigest()
    assert digest in h
    t2 = {d["id"] for d in json.loads(re.search(r"const PL = (\[.*?\]);\n", h, re.S).group(1))}
    others = pred[~pred.player_id.isin(t2)]
    assert not any(n in h for n in others.name.dropna().unique() if len(n) > 6)


def test_passports_match_frozen_2027_list():
    """The 2027 figures on the public passports must be the frozen ones, to the digit."""
    f = pd.read_csv(f"{OUT}/predictions_2027.csv").set_index("player_id")
    p = pd.read_csv(f"{OUT}/passport_players.csv").set_index("player_id")
    cols = ["p2027", "p2027_lo", "p2027_hi"]
    assert ((p[cols] - f.loc[p.index, cols]).abs().max() < 1e-12).all()


def test_no_unverifiable_timestamp_claim():
    h = open(f"{SITE}/inverted/passport/index.html", encoding="utf-8").read().lower()
    assert "timestamped" not in h and "committed locally at 16:23 pt" in h and "first public when pushed to github" in h


def test_first_public_record_is_third_party():
    """The push time must come from GitHub's record, with a Wayback Machine archive; never our own word."""
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/passport/first_public.json")
    if not os.path.exists(fp):
        return
    r = json.load(open(fp))
    assert r["github_record"].startswith("https://github.com/") or r["github_record"].startswith("https://api.github.com/")
    assert all(r[k].startswith("https://web.archive.org/web/") for k in ("wayback_prereg", "wayback_hash"))
    h = open(f"{SITE}/inverted/passport/index.html", encoding="utf-8").read()
    assert all(html.escape(r[k]) in h for k in ("github_record", "wayback_prereg", "wayback_hash"))
