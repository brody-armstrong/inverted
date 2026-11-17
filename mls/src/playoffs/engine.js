// Playoff engine used by the web page (one random run of the whole bracket).
// Same rules as src/playoffs/engine.py; validated against its exact probabilities in src/playoffs/validate_js.mjs.
// DATA: out/playoffs/bracket.json. rng: function returning uniform [0,1).
function poisson(l, rng) { const L = Math.exp(-l); let k = 0, p = 1; do { k++; p *= rng(); } while (p > L); return k - 1; }

function playGame(D, host, guest, extraTime, rng) {
  const lh = Math.exp(D.c + D.h + D.a[host] - D.d[guest]), la = Math.exp(D.c + D.a[guest] - D.d[host]);
  const g = { host, guest, hg: poisson(lh, rng), ag: poisson(la, rng), et: false, pens: false };
  if (g.hg === g.ag && extraTime) { g.et = true; g.hg += poisson(lh / 3, rng); g.ag += poisson(la / 3, rng); }
  if (g.hg === g.ag) { g.pens = true; g.winner = rng() < 0.5 ? host : guest; }
  else g.winner = g.hg > g.ag ? host : guest;
  return g;
}

function playBo3(D, hi, lo, rng) {
  const games = [], wins = { [hi]: 0, [lo]: 0 }, hosts = [[hi, lo], [lo, hi], [hi, lo]];
  for (const [h, a] of hosts) {
    const g = playGame(D, h, a, false, rng); games.push(g); wins[g.winner]++;
    if (wins[g.winner] === 2) return { hi, lo, games, winner: g.winner };
  }
}

function playBracket(DATA, rng) {
  const D = DATA.draws[Math.floor(rng() * DATA.draws.length)];
  const T = DATA.teams, out = { rounds: { wc: [], r1: [], sf: [], cf: [], cup: [] } }, champs = {};
  const better = (x, y) => { const a = T[x], b = T[y];
    const k = t => [t.pts, t.w, t.gf - t.ga]; const ka = k(a), kb = k(b);
    for (let i = 0; i < 3; i++) if (ka[i] !== kb[i]) return ka[i] > kb[i] ? x : y; return x; };
  for (const conf of ["East", "West"]) {
    const s = DATA.seeds[conf], seed = Object.fromEntries(s.map((t, i) => [t, i + 1]));
    const hostBySeed = (x, y) => seed[x] < seed[y] ? [x, y] : [y, x];
    const wc = playGame(D, s[7], s[8], false, rng); wc.conf = conf; out.rounds.wc.push(wc);
    const pairs = [[s[0], wc.winner], [s[3], s[4]], [s[1], s[6]], [s[2], s[5]]];
    const r1 = pairs.map(([h, l]) => { const t = playBo3(D, h, l, rng); t.conf = conf; out.rounds.r1.push(t); return t.winner; });
    const sf = [[r1[0], r1[1]], [r1[2], r1[3]]].map(([x, y]) => {
      const [h, a] = hostBySeed(x, y); const g = playGame(D, h, a, true, rng); g.conf = conf; out.rounds.sf.push(g); return g.winner; });
    const [h, a] = hostBySeed(sf[0], sf[1]); const cf = playGame(D, h, a, true, rng); cf.conf = conf; out.rounds.cf.push(cf);
    champs[conf] = cf.winner;
  }
  const host = better(champs.East, champs.West), guest = host === champs.East ? champs.West : champs.East;
  const cup = playGame(D, host, guest, true, rng); out.rounds.cup.push(cup); out.champion = cup.winner;
  return out;
}

if (typeof module !== "undefined") module.exports = { playBracket, playGame, playBo3 };
