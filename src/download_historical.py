"""Download completed Premier League seasons from Football-Data.co.uk."""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path


SEASON_CODES = {
    "2019-20": "1920",
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}

BASE_URL = "https://football-data.co.uk/mmz4281/{code}/E0.csv"
CORE_COLUMNS = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}


# downloads and validates one completed premier league season
def download_season(season: str, output_dir: Path, overwrite: bool = False) -> Path:
    """Download and validate one completed season."""
    if season not in SEASON_CODES:
        raise ValueError(f"Unsupported season: {season}")

    destination = output_dir / f"{season}.csv"
    if destination.exists() and not overwrite:
        return destination

    request = urllib.request.Request(
        BASE_URL.format(code=SEASON_CODES[season]),
        headers={"User-Agent": "prem-league-predictor-data-refresh"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8-sig")

    rows = list(csv.DictReader(io.StringIO(raw)))
    if len(rows) != 380:
        raise ValueError(f"{season}: expected 380 fixtures, found {len(rows)}")
    if not rows or not CORE_COLUMNS.issubset(rows[0]):
        raise ValueError(f"{season}: required match columns are missing")
    if any(not all(row.get(column, "") != "" for column in CORE_COLUMNS) for row in rows):
        raise ValueError(f"{season}: one or more fixtures has missing core values")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(raw, encoding="utf-8", newline="")
    return destination


# runs the historical-data download command line workflow
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seasons", nargs="*", default=list(SEASON_CODES))
    parser.add_argument("--output-dir", type=Path, default=Path("data/historical"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for season in args.seasons:
        path = download_season(season, args.output_dir, args.overwrite)
        print(path)


if __name__ == "__main__":
    main()
