// Check the browser engine against the exact Python probabilities: 400,000 random brackets.
// Usage (from mls/): node src/playoffs/validate_js.js
const fs = require("fs");
const { playBracket } = require("./engine.js");
const DATA = JSON.parse(fs.readFileSync("out/playoffs/bracket.json"));
let s = 20261119; const rng = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const N = 400000, cnt = {}, finals = {};
for (const t of Object.keys(DATA.probs)) cnt[t] = { r1: 0, sf: 0, cf: 0, final: 0, champ: 0 };
const allSeeds = [...DATA.seeds.East, ...DATA.seeds.West];
for (let i = 0; i < N; i++) {
  const o = playBracket(DATA, rng), R = o.rounds;
  for (const t of allSeeds) if (DATA.seeds.East.indexOf(t) < 7 && DATA.seeds.East.includes(t) || DATA.seeds.West.indexOf(t) < 7 && DATA.seeds.West.includes(t)) cnt[t].r1++;
  for (const g of R.wc) cnt[g.winner].r1++;
  for (const t of R.r1) cnt[t.winner].sf++;
  for (const g of R.sf) cnt[g.winner].cf++;
  for (const g of R.cf) cnt[g.winner].final++;
  cnt[o.champion].champ++;
  const k = R.cf.map(g => g.winner).join("-"); finals[k] = (finals[k] || 0) + 1;
}
let worst = 0;
for (const t of Object.keys(cnt)) for (const k of Object.keys(cnt[t])) {
  const d = Math.abs(cnt[t][k] / N - DATA.probs[t][k]); worst = Math.max(worst, d);
  if (d > 0.005) console.log("MISMATCH", t, k, cnt[t][k] / N, DATA.probs[t][k]);
}
const top = Object.entries(finals).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => [k, +(v / N).toFixed(4)]);
const champTop = Object.entries(cnt).map(([t, c]) => c.champ / N).sort((a, b) => b - a)[0];
const res = { sims: N, max_abs_diff: +worst.toFixed(4), pass: worst < 0.005, likeliest_finals: top, top_seed_final:
  +((finals[`${DATA.seeds.East[0]}-${DATA.seeds.West[0]}`] || 0) / N).toFixed(4), favourite_champion_prob: +champTop.toFixed(4) };
console.log(JSON.stringify(res, null, 1));
fs.writeFileSync("out/playoffs/validate_js.json", JSON.stringify(res, null, 1));
