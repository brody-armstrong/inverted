"""Parse the uploaded 2026 salary workbook (club blocks: header row with club + 'Last updated', then a table)."""
import sys, re, csv
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_excel(src, header=None)


def money(v):
    if pd.isna(v) or str(v).strip() in ("-", ""):
        return None
    s = str(v).replace("$", "").replace(",", "").replace("+", "").strip()
    m = re.fullmatch(r"([\d.]+)\s*([MK]?)", s, re.I)
    if not m:
        return None
    return float(m.group(1)) * {"M": 1e6, "K": 1e3, "": 1}[m.group(2).upper()]


rows, club, updated = [], None, None
for _, r in d.iterrows():
    a, b = r[0], r[1]
    if isinstance(b, str) and b.startswith("Last updated"):
        club, updated = str(a).strip(), b.replace("Last updated:", "").strip()
        continue
    if str(a).strip() == "#" or club is None or pd.isna(b):
        continue
    rows.append([club, updated, str(b).strip(), money(r[2]), money(r[3]), str(r[3]), money(r[4]),
                 None if pd.isna(r[5]) else int(r[5]), None if pd.isna(r[6]) else str(r[6])])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["club", "updated", "player", "weekly", "annual", "annual_as_shown", "bonus", "age", "contract_end"])
    w.writerows(rows)
print(len(rows), "players,", len({r[0] for r in rows}), "clubs, updated dates:", sorted({r[1] for r in rows}))
