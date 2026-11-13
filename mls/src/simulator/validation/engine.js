// Runs the simulator page's own engine (teamQ + simulate, copied unchanged from src/simulator/tool_template.html) in Node.
// Usage: node src/simulator/validation/engine.js <data.json> [roster overrides json]  -> per-club Q, projected points, ranges, playoff and cup odds
const fs=require('fs');const DATA=new Function('return '+fs.readFileSync(process.argv[2],'utf8'))(); // same as the page: data embedded as a JS literal (allows NaN)
const N_SIM = 3000;
const T = DATA.teams, CURVE0 = DATA.curve, H = DATA.h, C = DATA.c, TOT = DATA.total, TEAM_SD = DATA.team_sd, NC = DATA.new_curve;
// scale the rank curve so a typical-size roster's shares sum to 1 (the scale the team model was fitted on)
const NMED = T.map(t => t.players.length).sort((a, b) => a - b)[Math.floor(T.length / 2)];
const CURVE = (() => { const s = CURVE0.slice(0, NMED).reduce((a, b) => a + b, 0); return CURVE0.map(x => x / s); })();
// replacement level: average projected impact of players ranked 21st or lower on their own club
const REPL = (() => { let s = 0, n = 0; T.forEach(t => t.players.slice().sort((a, b) => b.r - a.r).slice(20).forEach(p => { s += p.r; n++; })); return n ? s / n : 0; })();
let rosters = T.map(t => t.players.map(p => Object.assign({}, p)));

// ---------- engine ----------
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const pmf = (k, l) => { let p = Math.exp(-l); for (let i = 1; i <= k; i++) p *= l / i; return p; };
const LUT = (() => { const out = []; for (let g = -500; g <= 500; g++) { const gd = g / 100; const lh = Math.max((TOT + gd) / 2, .15), la = Math.max((TOT - gd) / 2, .15);
  let ph = 0, pd = 0; const A = [], B = []; for (let i = 0; i <= 10; i++) { A.push(pmf(i, lh)); B.push(pmf(i, la)); }
  for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) { if (i > j) ph += A[i] * B[j]; else if (i === j) pd += A[i] * B[j]; } out.push([ph, pd]); } return out; })();
const probs = gd => LUT[Math.max(0, Math.min(1000, Math.round(gd * 100) + 500))];
function teamQ(roster) {
  const act = roster.filter(p => !p.out).slice().sort((a, b) => b.r - a.r);
  // minutes share by rank within the club (historical MLS average); a new player takes his rank's share and pushes
  // everyone below him down one rank, so only the players he displaces lose minutes
  let q = 0; act.forEach((p, i) => { p.share = i < CURVE.length ? CURVE[i] : 0; q += p.share * 11 * p.r; });
  // a roster shorter than typical leaves minutes that a replacement-level player fills
  for (let i = act.length; i < NMED; i++) q += CURVE[i] * 11 * REPL;
  roster.filter(p => p.out).forEach(p => p.share = 0);
  return q;
}
function simulate(Qs, seed, n = N_SIM) {
  const rng = mulberry(seed), nt = Qs.length, sch = DATA.sched, east = T.map(t => t.conf === "East");
  const pts = Array.from({ length: nt }, () => new Float64Array(n)), po = new Float64Array(nt), cup = new Float64Array(nt);
  const ko = (a, b, q, home) => { const p = probs((home ? H : -H) + C * (q[a] - q[b])); return rng() < p[0] + p[1] / 2; };
  for (let s = 0; s < n; s++) {
    const q = Qs.map(x => x + TEAM_SD * gauss(rng)); const p = new Float64Array(nt);
    for (const [h, a] of sch) { const [ph, pd] = probs(H + C * (q[h] - q[a])); const u = rng(); if (u < ph) p[h] += 3; else if (u < ph + pd) { p[h] += 1; p[a] += 1; } else p[a] += 3; }
    for (let i = 0; i < nt; i++) { pts[i][s] = p[i]; p[i] += rng() * 1e-3; }
    const champs = [];
    for (const conf of [true, false]) {
      const seeds = [...Array(nt).keys()].filter(i => east[i] === conf).sort((x, y) => p[y] - p[x]).slice(0, 9);
      seeds.forEach(i => po[i]++);
      const wc = ko(seeds[7], seeds[8], q, true) ? seeds[7] : seeds[8];
      const bo3 = (a, b) => { let w = 0; for (let g = 0; g < 3; g++) { w += ko(a, b, q, g !== 1) ? 1 : 0; if (w === 2 || g + 1 - w === 2) break; } return w >= 2 ? a : b; };
      const r1 = [bo3(seeds[0], wc), bo3(seeds[3], seeds[4]), bo3(seeds[1], seeds[6]), bo3(seeds[2], seeds[5])];
      const rk = new Map([...seeds, wc].map((t, i) => [t, i]));
      const g1 = (a, b) => { const [x, y] = rk.get(a) <= rk.get(b) ? [a, b] : [b, a]; return ko(x, y, q, true) ? x : y; };
      champs.push(g1(g1(r1[0], r1[1]), g1(r1[2], r1[3])));
    }
    const [a, b] = p[champs[0]] >= p[champs[1]] ? champs : [champs[1], champs[0]];
    cup[ko(a, b, q, true) ? a : b]++;
  }
  return Qs.map((_, i) => { const arr = Array.from(pts[i]).sort((x, y) => x - y);
    return { mean: arr.reduce((x, y) => x + y, 0) / n, lo: arr[Math.floor(n * .1)], hi: arr[Math.floor(n * .9)], po: po[i] / n, cup: cup[i] / n }; });
}
// rosters may be overridden (A7): argv[3] = JSON file of {team index: [players]}
if (process.argv[3]) { const o = new Function('return '+fs.readFileSync(process.argv[3],'utf8'))(); for (const k in o) rosters[+k] = o[k]; }
const Qs = rosters.map(r => teamQ(r)); const res = simulate(Qs, 7, 20000);
console.log(JSON.stringify(T.map((t, i) => ({ name: t.name, Q: Qs[i], pts: res[i].mean, lo: res[i].lo, hi: res[i].hi, po: res[i].po, cup: res[i].cup, n: rosters[i].length }))));
