"""Parse the saved Wikipedia 'Designated Player Rule' page: club DP history tables + current DPs table."""
import sys, re, csv
from bs4 import BeautifulSoup

path, out_hist, out_cur = sys.argv[1], sys.argv[2], sys.argv[3]
b = BeautifulSoup(open(path, encoding="utf-8").read(), "lxml")
def txt(el, sep=" "):
    t = el.get_text(sep, strip=True).replace("\xad", "")
    t = re.sub(r"\s*\[\s*[\w ]+\s*\]", "", t)          # footnote markers like [ 17 ] or [a]
    return re.sub(r"\s+", " ", t).strip(" ,")


def years(el):
    t = el.get_text(" ", strip=True)
    t = re.sub(r"\[[^\]]*\]", "", t)                       # footnotes
    t = re.sub(r"\s*[–-]\s*", "–", t)                       # 2018 – 2019 -> 2018–2019
    t = re.sub(r"(\d{4}|–)\s+(?=\d{4})", r"\1, ", t)        # line breaks between spans -> comma
    return re.sub(r"\s+", " ", t).strip(" ,")

hist = []
for h in b.find_all(["h3"]):
    club = txt(h)
    tbl = h.find_next("table")
    if not tbl or "Years as DP" not in tbl.get_text():
        continue
    for tr in tbl.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        name = txt(cells[0])
        bold = cells[0].find("b") is not None
        hist.append([club, name, txt(cells[1]), years(cells[2]), "current" if bold else ""])

cur = []
for tbl in b.find_all("table"):
    head = [txt(th) for th in tbl.find_all("tr")[0].find_all("th")]
    if head and head[0].startswith("Year signed as DP"):
        for tr in tbl.find_all("tr")[1:]:
            c = [txt(td) for td in tr.find_all(["td", "th"])]
            if len(c) >= 5:
                cur.append(c[:5])
        break

with open(out_hist, "w", newline="") as f:
    w = csv.writer(f, delimiter="|"); w.writerow(["club", "player", "previous club", "years as DP", "marked"]); w.writerows(hist)
with open(out_cur, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year_signed", "player", "nation", "club", "comp_2026"]); w.writerows(cur)
print(len(hist), "history rows,", len({r[0] for r in hist}), "clubs,", sum(r[4] == "current" for r in hist), "marked current |", len(cur), "current DPs")
