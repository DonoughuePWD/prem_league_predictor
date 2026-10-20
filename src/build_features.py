"""Create leakage-safe pre-match features from normalized fixtures."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


ELO_INITIAL_RATING = 1500.0
ELO_HOME_ADVANTAGE = 65.0
ELO_K_FACTOR = 20.0
ELO_SEASON_CARRYOVER = 2 / 3


# stores a team's running league, form, venue, and date statistics
@dataclass
class TeamState:
    games: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    recent_points: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    recent_goals_for: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    recent_goals_against: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    home_points: int = 0
    home_games: int = 0
    away_points: int = 0
    away_games: int = 0
    last_date: pd.Timestamp | None = None


# divides safely and returns zero when no observations exist
def safe_rate(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


# calculates normalized points form over a recent match window
def form_points(state: TeamState, window: int) -> float:
    values = list(state.recent_points)[-window:]
    return safe_rate(sum(values), 3 * len(values))


# calculates an average from the most recent values in a series
def recent_average(values: deque[int], window: int) -> float:
    selected = list(values)[-window:]
    return safe_rate(sum(selected), len(selected))


# moves an elo rating toward the league average between seasons
def regress_elo(rating: float) -> float:
    """Move a rating toward average between seasons."""
    return ELO_INITIAL_RATING + (rating - ELO_INITIAL_RATING) * ELO_SEASON_CARRYOVER


# estimates the home team's expected result from both elo ratings
def expected_home_score(home_rating: float, away_rating: float) -> float:
    """Return the Elo expected score for the home team."""
    return 1 / (1 + 10 ** ((away_rating - (home_rating + ELO_HOME_ADVANTAGE)) / 400))


# increases the elo effect of wins with larger goal margins
def elo_margin_multiplier(goal_margin: int) -> float:
    """Give decisive results modestly more influence on Elo."""
    return 1.0 if goal_margin <= 1 else 1.5 if goal_margin == 2 else 1.75


# calculates the zero-sum elo adjustment after a match
def elo_rating_change(home_rating: float, away_rating: float, home_score: float, goal_margin: int) -> float:
    """Return the zero-sum rating change from the home team's perspective."""
    return ELO_K_FACTOR * elo_margin_multiplier(goal_margin) * (
        home_score - expected_home_score(home_rating, away_rating)
    )


# ranks teams using points, goal difference, and goals scored
def league_positions(states: dict[str, TeamState], teams: set[str]) -> dict[str, int]:
    keys = {
        team: (states[team].points, states[team].goals_for - states[team].goals_against, states[team].goals_for)
        for team in teams
    }
    ordered = sorted(teams, key=lambda team: (*keys[team], team), reverse=True)
    positions: dict[str, int] = {}
    previous_key: tuple[int, int, int] | None = None
    previous_position = 1
    for index, team in enumerate(ordered, start=1):
        key = keys[team]
        if key != previous_key:
            previous_position = index
            previous_key = key
        positions[team] = previous_position
    return positions


# captures one team's pre-match features without using future information
def snapshot(
    prefix: str,
    state: TeamState,
    position: int,
    rating: float,
    match_date: pd.Timestamp,
    venue: str,
) -> dict[str, float | int | None]:
    return {
        f"{prefix}_games_played": state.games,
        f"{prefix}_points": state.points,
        f"{prefix}_points_per_game": safe_rate(state.points, state.games),
        f"{prefix}_goal_difference": state.goals_for - state.goals_against,
        f"{prefix}_goals_for_per_game": safe_rate(state.goals_for, state.games),
        f"{prefix}_goals_against_per_game": safe_rate(state.goals_against, state.games),
        f"{prefix}_win_rate": safe_rate(state.wins, state.games),
        f"{prefix}_draw_rate": safe_rate(state.draws, state.games),
        f"{prefix}_loss_rate": safe_rate(state.losses, state.games),
        f"{prefix}_form_last_3": form_points(state, 3),
        f"{prefix}_form_last_5": form_points(state, 5),
        f"{prefix}_goals_for_last_5": recent_average(state.recent_goals_for, 5),
        f"{prefix}_goals_against_last_5": recent_average(state.recent_goals_against, 5),
        f"{prefix}_venue_points_per_game": safe_rate(
            state.home_points if venue == "home" else state.away_points,
            state.home_games if venue == "home" else state.away_games,
        ),
        f"{prefix}_rest_days": None if state.last_date is None else int((match_date - state.last_date).days),
        f"{prefix}_league_position": position,
        f"{prefix}_elo": rating,
    }


# updates a team's running state after a completed match
def update_state(state: TeamState, goals_for: int, goals_against: int, points: int, venue: str, date: pd.Timestamp) -> None:
    state.games += 1
    state.points += points
    state.goals_for += goals_for
    state.goals_against += goals_against
    state.wins += int(points == 3)
    state.draws += int(points == 1)
    state.losses += int(points == 0)
    state.recent_points.append(points)
    state.recent_goals_for.append(goals_for)
    state.recent_goals_against.append(goals_against)
    if venue == "home":
        state.home_games += 1
        state.home_points += points
    else:
        state.away_games += 1
        state.away_points += points
    state.last_date = date


# builds leakage-safe pre-match features for every historical fixture
def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    rows: list[dict] = []
    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    previous_season: str | None = None

    for season, season_matches in matches.groupby("season", sort=False):
        teams = set(season_matches["home_team"]) | set(season_matches["away_team"])
        if previous_season is not None:
            for team in list(ratings):
                ratings[team] = regress_elo(ratings[team])
        previous_season = season
        states: defaultdict[str, TeamState] = defaultdict(TeamState)

        for match in season_matches.itertuples(index=False):
            positions = league_positions(states, teams)
            home_state = states[match.home_team]
            away_state = states[match.away_team]
            row = match._asdict()
            row.update(snapshot("home", home_state, positions[match.home_team], ratings[match.home_team], match.date, "home"))
            row.update(snapshot("away", away_state, positions[match.away_team], ratings[match.away_team], match.date, "away"))
            row["elo_difference"] = ratings[match.home_team] - ratings[match.away_team]
            row["season_progress"] = safe_rate(home_state.games + away_state.games, 76)
            rows.append(row)

            if match.result == "H":
                home_points, away_points, home_score = 3, 0, 1.0
            elif match.result == "A":
                home_points, away_points, home_score = 0, 3, 0.0
            else:
                home_points, away_points, home_score = 1, 1, 0.5

            goal_margin = abs(int(match.home_goals) - int(match.away_goals))
            change = elo_rating_change(
                ratings[match.home_team], ratings[match.away_team], home_score, goal_margin
            )
            ratings[match.home_team] += change
            ratings[match.away_team] -= change

            update_state(home_state, int(match.home_goals), int(match.away_goals), home_points, "home", match.date)
            update_state(away_state, int(match.away_goals), int(match.home_goals), away_points, "away", match.date)

    return pd.DataFrame(rows)


# runs the feature-building command line workflow
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/matches.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/matches_with_features.csv"))
    args = parser.parse_args()

    features = build_features(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(features)} feature rows with {len(features.columns)} columns")


if __name__ == "__main__":
    main()
