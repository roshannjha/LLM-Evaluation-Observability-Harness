from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = os.environ.get(
    "EDGAR_USER_AGENT", "llm-eval-harness research (guptabhumika1d007@gmail.com)"
)
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
REQUEST_PAUSE_SEC = 0.5

WANTED_FORMS = {"10-K", "10-Q"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s


def parse_cik(accession: str) -> int:
    return int(accession.split("-")[0])


def accession_nodashes(accession: str) -> str:
    return accession.replace("-", "")


def select_filings(csv_path: Path, num_10k: int, num_10q: int) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            form = (row.get("Form type") or "").strip()
            if form in WANTED_FORMS:
                rows.append(
                    {
                        "form": form,
                        "filing_date": (row.get("Filing date") or "").strip(),
                        "reporting_date": (row.get("Reporting date") or "").strip(),
                        "accession": (row.get("Accession number") or "").strip(),
                    }
                )
    rows.sort(key=lambda r: r["filing_date"], reverse=True)
    tenk = [r for r in rows if r["form"] == "10-K"][:num_10k]
    tenq = [r for r in rows if r["form"] == "10-Q"][:num_10q]
    selected = tenk + tenq
    if not selected:
        sys.exit("No 10-K/10-Q rows found in the CSV. Is this an EDGAR filing index?")
    return selected


def find_primary_document(session: requests.Session, cik: int, accession: str) -> str:
    folder = f"{ARCHIVES}/{cik}/{accession_nodashes(accession)}"
    resp = session.get(f"{folder}/index.json", timeout=30)
    resp.raise_for_status()
    items = resp.json()["directory"]["item"]

    candidates = []
    for it in items:
        name = it["name"]
        lower = name.lower()
        if not (lower.endswith(".htm") or lower.endswith(".html") or lower.endswith(".txt")):
            continue
        if re.match(r"^r\d+\.htm", lower):
            continue
        if "index" in lower or lower.endswith("-index.htm"):
            continue
        size = int(it.get("size") or 0)
        candidates.append((size, name))

    if not candidates:
        raise RuntimeError(f"No primary document found in {folder}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def download_filing(session: requests.Session, filing: dict, out_dir: Path) -> Path | None:
    cik = parse_cik(filing["accession"])
    acc = filing["accession"]
    try:
        doc = find_primary_document(session, cik, acc)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {acc}: could not locate primary doc ({exc})")
        return None

    url = f"{ARCHIVES}/{cik}/{accession_nodashes(acc)}/{doc}"
    time.sleep(REQUEST_PAUSE_SEC)
    resp = session.get(url, timeout=60)
    resp.raise_for_status()

    text = html_to_text(resp.text) if doc.lower().endswith((".htm", ".html")) else resp.text
    label = f"{filing['form']}_{filing['reporting_date'] or filing['filing_date']}"
    fname = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{label}_{acc}.txt")
    path = out_dir / fname
    path.write_text(text, encoding="utf-8")
    print(f"  + {filing['form']:5s} {filing['reporting_date'] or filing['filing_date']}"
          f"  ->  {path.name}  ({len(text):,} chars)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path, help="EDGAR filing-index CSV")
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--num-10k", type=int, default=3)
    ap.add_argument("--num-10q", type=int, default=3)
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    args.out.mkdir(parents=True, exist_ok=True)

    filings = select_filings(args.csv, args.num_10k, args.num_10q)
    print(f"Selected {len(filings)} filings "
          f"({args.num_10k} x 10-K, {args.num_10q} x 10-Q). Downloading...")

    session = _session()
    saved = [download_filing(session, f, args.out) for f in filings]
    ok = [p for p in saved if p]
    print(f"\nDone: {len(ok)}/{len(filings)} filings saved to {args.out}/")
    if len(ok) < len(filings):
        print("Some filings failed — re-run to retry (SEC occasionally rate-limits).")


if __name__ == "__main__":
    main()
