"""Normalize historical and API match data into one canonical schema."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TEAM_ALIASES = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Brighton & Hove Albion FC": "Brighton",
    "Leeds United": "Leeds",
    "Leeds United FC": "Leeds",
    "Leicester City": "Leicester",
    "Leicester City FC": "Leicester",
    "Manchester City": "Man City",
    "Manchester City FC": "Man City",
    "Manchester United": "Man United",
    "Manchester United FC": "Man United",
    "Newcastle United": "Newcastle",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sheffield United FC": "Sheffield United",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United": "West Ham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton Wanderers FC": "Wolves",
}

CANONICAL_COLUMNS = [
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "source",
]


# converts alternative club names into one consistent team name
def normalize_team(name: str) -> str:
    cleaned = re.sub(r"\s+FC$", "", str(name).strip())
    return TEAM_ALIASES.get(str(name).strip(), TEAM_ALIASES.get(cleaned, cleaned))


# converts one football-data csv into the canonical match schema
def load_historical_file(path: Path) -> pd.DataFrame:
    season = path.stem
    frame = pd.read_csv(path)
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    result = pd.DataFrame(
        {
            "season": season,
            "date": pd.to_datetime(frame["Date"], format="%d/%m/%Y"),
            "home_team": frame["HomeTeam"].map(normalize_team),
            "away_team": frame["AwayTeam"].map(normalize_team),
            "home_goals": pd.to_numeric(frame["FTHG"], errors="raise").astype(int),
            "away_goals": pd.to_numeric(frame["FTAG"], errors="raise").astype(int),
            "result": frame["FTR"],
            "source": "football-data.co.uk",
        }
    )
    return result[CANONICAL_COLUMNS]


# combines every historical season csv in a directory
def load_historical_directory(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No historical CSVs found in {directory}")
    return pd.concat([load_historical_file(path) for path in files], ignore_index=True)


# converts api season dates into the repository's season label format
def season_label(start_date: str, end_date: str) -> str:
    start_year = int(str(start_date)[:4])
    end_year = int(str(end_date)[:4])
    return f"{start_year}-{end_year % 100:02d}"


# converts finished api fixtures into the canonical match schema
def load_current_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    finished = frame.loc[frame["status"].eq("FINISHED")].copy()
    if finished.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    if "season.startDate" in finished and "season.endDate" in finished:
        seasons = [
            season_label(start, end)
            for start, end in zip(finished["season.startDate"], finished["season.endDate"])
        ]
    else:
        years = pd.to_datetime(finished["utcDate"], utc=True).dt.year
        seasons = [f"{year}-{(year + 1) % 100:02d}" for year in years]

    result_map = {"HOME_TEAM": "H", "AWAY_TEAM": "A", "DRAW": "D"}
    result = pd.DataFrame(
        {
            "season": seasons,
            "date": pd.to_datetime(finished["utcDate"], utc=True).dt.tz_localize(None).dt.normalize(),
            "home_team": finished["homeTeam.name"].map(normalize_team),
            "away_team": finished["awayTeam.name"].map(normalize_team),
            "home_goals": pd.to_numeric(finished["score.fullTime.home"]).astype(int),
            "away_goals": pd.to_numeric(finished["score.fullTime.away"]).astype(int),
            "result": finished["score.winner"].map(result_map),
            "source": "football-data.org",
        }
    )
    return result[CANONICAL_COLUMNS]


# checks canonical matches for missing, invalid, or duplicate data
def validate_matches(matches: pd.DataFrame) -> None:
    if matches[CANONICAL_COLUMNS].isna().any().any():
        raise ValueError("Canonical match data contains missing core values")
    if not matches["result"].isin(["H", "D", "A"]).all():
        raise ValueError("Canonical match data contains an invalid result")
    duplicates = matches.duplicated(["season", "date", "home_team", "away_team"])
    if duplicates.any():
        raise ValueError(f"Canonical match data contains {int(duplicates.sum())} duplicate fixtures")
    expected = pd.Series("D", index=matches.index)
    expected.loc[matches["home_goals"] > matches["away_goals"]] = "H"
    expected.loc[matches["home_goals"] < matches["away_goals"]] = "A"
    if not expected.eq(matches["result"]).all():
        raise ValueError("One or more results does not agree with the final score")


# combines, sorts, and validates historical and optional current matches
def prepare_matches(
    historical_dir: Path,
    current_file: Path | None = None,
    include_current: bool = False,
) -> pd.DataFrame:
    historical = load_historical_directory(historical_dir)
    frames = [historical]
    if include_current and current_file and current_file.exists():
        frames.append(load_current_file(current_file))
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.drop_duplicates(["season", "date", "home_team", "away_team"], keep="first")
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    validate_matches(matches)
    return matches


# runs the match-preparation command line workflow
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dir", type=Path, default=Path("data/historical"))
    parser.add_argument("--current-file", type=Path, default=Path("data/current/matches.csv"))
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/matches.csv"))
    args = parser.parse_args()

    matches = prepare_matches(args.historical_dir, args.current_file, args.include_current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(matches)} normalized matches across {matches['season'].nunique()} seasons")


if __name__ == "__main__":
    main()
