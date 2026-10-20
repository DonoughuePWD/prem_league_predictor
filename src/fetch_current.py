"""Fetch current-season Premier League fixtures from football-data.org."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


API_URL = "https://api.football-data.org/v4/competitions/PL/matches"


# fetches the current premier league match payload from the api
def fetch_matches(api_key: str) -> dict:
    request = urllib.request.Request(API_URL, headers={"X-Auth-Token": api_key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"football-data.org returned HTTP {error.code}") from error


# runs the current-season data fetch command line workflow
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/current/matches.csv"))
    args = parser.parse_args()

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise SystemExit("Set FOOTBALL_DATA_API_KEY before fetching current matches")

    payload = fetch_matches(api_key)
    matches = pd.json_normalize(payload.get("matches", []))
    if matches.empty:
        raise RuntimeError("The API returned no Premier League matches")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"Saved {len(matches)} fixtures to {args.output}")


if __name__ == "__main__":
    main()
