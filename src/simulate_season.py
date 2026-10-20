"""Simulate the rest of a Premier League season with the trained baseline model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .build_features import (
        ELO_INITIAL_RATING,
        TeamState,
        elo_rating_change,
        league_positions,
        regress_elo,
        snapshot,
        update_state,
    )
    from .prepare_matches import load_current_file, load_historical_file
    from .train_model import softmax
else:
    from build_features import (  # type: ignore[no-redef]
        ELO_INITIAL_RATING,
        TeamState,
        elo_rating_change,
        league_positions,
        regress_elo,
        snapshot,
        update_state,
    )
    from prepare_matches import load_current_file, load_historical_file  # type: ignore[no-redef]
    from train_model import softmax  # type: ignore[no-redef]


def remaining_fixtures(teams: set[str], played: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return every unplayed directed pairing in a double round-robin season."""
    return [
        (home, away)
        for home in sorted(teams)
        for away in sorted(teams)
        if home != away and (home, away) not in played
    ]


def load_model(path: Path) -> dict[str, object]:
    artifact = np.load(path, allow_pickle=False)
    return {
        "weights": artifact["weights"],
        "medians": artifact["medians"],
        "means": artifact["means"],
        "scales": artifact["scales"],
        "feature_columns": artifact["feature_columns"].astype(str).tolist(),
        "targets": artifact["targets"].astype(str).tolist(),
    }


def load_current_results(path: Path) -> pd.DataFrame:
    """Load either a football-data.org API snapshot or football-data.co.uk CSV."""
    columns = pd.read_csv(path, nrows=0).columns
    if "utcDate" in columns:
        return load_current_file(path)
    return load_historical_file(path)


def prepare_feature_vector(row: dict[str, object], model: dict[str, object]) -> np.ndarray:
    """Standardize one simulated fixture without constructing a DataFrame."""
    values = np.array([float(row.get(column, np.nan)) for column in model["feature_columns"]])
    missing = np.isnan(values)
    values[missing] = model["medians"][missing]
    standardized = (values - model["means"]) / model["scales"]
    return np.concatenate(([1.0], standardized))


def replay_matches(matches: pd.DataFrame, target_season: str) -> tuple[dict[str, TeamState], dict[str, float], set[str]]:
    """Replay history and return states and Elo immediately after known results."""
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "home_team", "away_team"])
    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    target_states: dict[str, TeamState] | None = None
    target_teams: set[str] | None = None

    for index, (season, season_matches) in enumerate(frame.groupby("season", sort=False)):
        if index:
            for team in list(ratings):
                ratings[team] = regress_elo(ratings[team])
        states: defaultdict[str, TeamState] = defaultdict(TeamState)
        teams = set(season_matches["home_team"]) | set(season_matches["away_team"])
        for match in season_matches.itertuples(index=False):
            home_goals, away_goals = int(match.home_goals), int(match.away_goals)
            if match.result == "H":
                home_points, away_points, home_score = 3, 0, 1.0
            elif match.result == "A":
                home_points, away_points, home_score = 0, 3, 0.0
            else:
                home_points, away_points, home_score = 1, 1, 0.5
            change = elo_rating_change(
                ratings[match.home_team], ratings[match.away_team], home_score, abs(home_goals - away_goals)
            )
            ratings[match.home_team] += change
            ratings[match.away_team] -= change
            update_state(states[match.home_team], home_goals, away_goals, home_points, "home", match.date)
            update_state(states[match.away_team], away_goals, home_goals, away_points, "away", match.date)
        if str(season) == target_season:
            target_states, target_teams = dict(states), teams
            break

    if target_states is None or target_teams is None:
        raise ValueError(f"Season {target_season!r} is not present in the match data")
    return target_states, dict(ratings), target_teams


def sample_score(
    result: str,
    home_state: TeamState,
    away_state: TeamState,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample a plausible score constrained to the model's sampled result."""
    home_rate = 1.5 if not home_state.games else max(0.25, home_state.goals_for / home_state.games)
    away_rate = 1.2 if not away_state.games else max(0.25, away_state.goals_for / away_state.games)
    for _ in range(100):
        home_goals = int(rng.poisson(home_rate))
        away_goals = int(rng.poisson(away_rate))
        sampled = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"
        if sampled == result:
            return home_goals, away_goals
    return (1, 0) if result == "H" else (0, 1) if result == "A" else (1, 1)


def simulate(
    matches: pd.DataFrame,
    season: str,
    model: dict[str, object],
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    """Run season simulations and aggregate expected positions and points."""
    season_matches = matches.loc[matches["season"].astype(str).eq(season)]
    played = set(zip(season_matches["home_team"], season_matches["away_team"]))
    base_states, base_ratings, teams = replay_matches(matches, season)
    current_positions = league_positions(base_states, teams)
    fixtures = remaining_fixtures(teams, played)
    expected_count = len(teams) * (len(teams) - 1)
    if len(played) + len(fixtures) != expected_count:
        raise ValueError("Known and remaining fixtures do not form a complete double round-robin")

    targets = model["targets"]
    rng = np.random.default_rng(seed)
    totals = {
        team: {"points": 0.0, "position": 0.0, "elo": 0.0, "title": 0, "top_4": 0, "relegated": 0}
        for team in teams
    }

    for _ in range(simulations):
        states = deepcopy(base_states)
        ratings = base_ratings.copy()
        ordered_fixtures = [fixtures[index] for index in rng.permutation(len(fixtures))]
        simulated_date = pd.Timestamp(f"{season[:4]}-10-20")
        for home_team, away_team in ordered_fixtures:
            positions = league_positions(states, teams)
            row = {}
            row.update(snapshot("home", states[home_team], positions[home_team], ratings[home_team], simulated_date, "home"))
            row.update(snapshot("away", states[away_team], positions[away_team], ratings[away_team], simulated_date, "away"))
            # Exact future dates are not required for the round-robin simulation;
            # use neutral one-week rest rather than inventing a schedule.
            row["home_rest_days"] = 7
            row["away_rest_days"] = 7
            row["elo_difference"] = ratings[home_team] - ratings[away_team]
            row["season_progress"] = (states[home_team].games + states[away_team].games) / 76
            vector = prepare_feature_vector(row, model)
            probabilities = softmax((vector @ model["weights"])[None, :])[0]
            result = str(rng.choice(targets, p=probabilities))
            home_goals, away_goals = sample_score(result, states[home_team], states[away_team], rng)
            home_points, away_points, home_score = (
                (3, 0, 1.0) if result == "H" else (0, 3, 0.0) if result == "A" else (1, 1, 0.5)
            )
            change = elo_rating_change(ratings[home_team], ratings[away_team], home_score, abs(home_goals - away_goals))
            ratings[home_team] += change
            ratings[away_team] -= change
            update_state(states[home_team], home_goals, away_goals, home_points, "home", simulated_date)
            update_state(states[away_team], away_goals, home_goals, away_points, "away", simulated_date)

        ordered_teams = sorted(
            teams,
            key=lambda team: (
                states[team].points,
                states[team].goals_for - states[team].goals_against,
                states[team].goals_for,
                team,
            ),
            reverse=True,
        )
        if any(states[team].games != 38 for team in teams):
            raise RuntimeError("Simulation did not produce 38 matches for every team")
        for position, team in enumerate(ordered_teams, start=1):
            totals[team]["points"] += states[team].points
            totals[team]["position"] += position
            totals[team]["elo"] += ratings[team]
            totals[team]["title"] += position == 1
            totals[team]["top_4"] += position <= 4
            totals[team]["relegated"] += position >= len(teams) - 2

    rows = []
    for team, values in totals.items():
        rows.append(
            {
                "team": team,
                "current_position": current_positions[team],
                "played": base_states[team].games,
                "current_points": base_states[team].points,
                "current_elo": base_ratings[team],
                "expected_position": values["position"] / simulations,
                "expected_points": values["points"] / simulations,
                "expected_final_elo": values["elo"] / simulations,
                "title_probability": values["title"] / simulations,
                "top_4_probability": values["top_4"] / simulations,
                "relegation_probability": values["relegated"] / simulations,
            }
        )
    result = pd.DataFrame(rows).sort_values(["expected_position", "expected_points"], ascending=[True, False])
    result.insert(0, "projected_position", range(1, len(result) + 1))
    result.insert(0, "season", season)
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, default=Path("data/processed/matches.csv"))
    parser.add_argument("--current-results", type=Path, default=Path("data/current/2026-27.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/softmax_baseline.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/2026-27-projection.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("data/processed/forecast.json"))
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202627)
    args = parser.parse_args()

    historical = pd.read_csv(args.historical)
    current = load_current_results(args.current_results)
    matches = pd.concat([historical, current], ignore_index=True)
    projection = simulate(matches, args.season, load_model(args.model), args.simulations, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    projection.to_csv(args.output, index=False, float_format="%.4f")
    metrics_path = args.model.with_suffix(".json")
    model_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    forecast = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "data_cutoff": pd.to_datetime(current["date"]).max().date().isoformat(),
        "completed_matches": int(len(current)),
        "remaining_matches": int(380 - len(current)),
        "simulations": args.simulations,
        "seed": args.seed,
        "model": {
            key: model_metrics.get(key)
            for key in ["test_season", "accuracy", "log_loss", "saved_model_rows", "refit_all"]
        },
        "teams": json.loads(projection.to_json(orient="records")),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(forecast, indent=2), encoding="utf-8")
    display = projection.copy()
    for column in ["title_probability", "top_4_probability", "relegation_probability"]:
        display[column] = display[column].map(lambda value: f"{value:.1%}")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSaved {len(projection)} projected teams to {args.output} and {args.json_output}")


if __name__ == "__main__":
    main()
