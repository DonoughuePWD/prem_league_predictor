"""Build an actual league table enriched with existing form and Elo features."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

if __package__:
    from .build_features import (
        ELO_INITIAL_RATING,
        TeamState,
        elo_rating_change,
        form_points,
        regress_elo,
        safe_rate,
        update_state,
    )
else:
    from build_features import (  # type: ignore[no-redef]
        ELO_INITIAL_RATING,
        TeamState,
        elo_rating_change,
        form_points,
        regress_elo,
        safe_rate,
        update_state,
    )


def build_league_table(matches: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Return final or in-progress standings for a season using completed matches."""
    required = {"season", "date", "home_team", "away_team", "home_goals", "away_goals", "result"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Matches are missing required columns: {', '.join(sorted(missing))}")

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["home_goals", "away_goals", "result"])
    frame = frame.loc[frame["result"].isin(["H", "D", "A"])]
    frame = frame.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No completed matches are available")

    seasons = list(dict.fromkeys(frame["season"].astype(str)))
    selected_season = season or seasons[-1]
    if selected_season not in seasons:
        raise ValueError(f"Season {selected_season!r} is not present in the matches")

    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    selected_states: dict[str, TeamState] | None = None
    selected_teams: set[str] | None = None

    for index, (season_name, season_matches) in enumerate(frame.groupby("season", sort=False)):
        if index:
            for team in list(ratings):
                ratings[team] = regress_elo(ratings[team])

        states: defaultdict[str, TeamState] = defaultdict(TeamState)
        teams = set(season_matches["home_team"]) | set(season_matches["away_team"])

        for match in season_matches.itertuples(index=False):
            home_goals = int(match.home_goals)
            away_goals = int(match.away_goals)
            if match.result == "H":
                home_points, away_points, home_score = 3, 0, 1.0
            elif match.result == "A":
                home_points, away_points, home_score = 0, 3, 0.0
            else:
                home_points, away_points, home_score = 1, 1, 0.5

            change = elo_rating_change(
                ratings[match.home_team],
                ratings[match.away_team],
                home_score,
                abs(home_goals - away_goals),
            )
            ratings[match.home_team] += change
            ratings[match.away_team] -= change
            update_state(states[match.home_team], home_goals, away_goals, home_points, "home", match.date)
            update_state(states[match.away_team], away_goals, home_goals, away_points, "away", match.date)

        if str(season_name) == selected_season:
            selected_states = dict(states)
            selected_teams = teams
            break

    if selected_states is None or selected_teams is None:
        raise ValueError(f"Unable to build table for season {selected_season!r}")

    rows = []
    for team in selected_teams:
        state = selected_states[team]
        rows.append(
            {
                "team": team,
                "played": state.games,
                "won": state.wins,
                "drawn": state.draws,
                "lost": state.losses,
                "goals_for": state.goals_for,
                "goals_against": state.goals_against,
                "goal_difference": state.goals_for - state.goals_against,
                "points": state.points,
                "points_per_game": safe_rate(state.points, state.games),
                "form_last_5": form_points(state, 5),
                "home_points_per_game": safe_rate(state.home_points, state.home_games),
                "away_points_per_game": safe_rate(state.away_points, state.away_games),
                "elo": ratings[team],
            }
        )

    table = pd.DataFrame(rows)
    table["elo_rank"] = table["elo"].rank(method="min", ascending=False).astype(int)
    table = table.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table.insert(0, "position", table.index + 1)
    table.insert(0, "season", selected_season)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/matches.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/league_table.csv"))
    parser.add_argument("--season", help="Season to build; defaults to the latest available season")
    args = parser.parse_args()

    table = build_league_table(pd.read_csv(args.input), args.season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.3f")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved {len(table)} teams to {args.output}")


if __name__ == "__main__":
    main()
