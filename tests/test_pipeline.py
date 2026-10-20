from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.build_features import build_features
from src.build_league_table import build_league_table
from src.prepare_matches import load_current_file, normalize_team, prepare_matches


REPO_ROOT = Path(__file__).resolve().parents[1]


class PrepareMatchesTests(unittest.TestCase):
    def test_all_historical_seasons_are_combined(self) -> None:
        matches = prepare_matches(REPO_ROOT / "data" / "historical")
        self.assertEqual(len(matches), 2660)
        self.assertEqual(matches["season"].nunique(), 7)
        self.assertTrue((matches.groupby("season").size() == 380).all())
        self.assertTrue((matches.groupby("season").apply(lambda group: len(set(group["home_team"]) | set(group["away_team"]))) == 20).all())
        self.assertFalse(matches.duplicated(["season", "date", "home_team", "away_team"]).any())
        self.assertTrue(matches["date"].is_monotonic_increasing)

    def test_team_names_are_normalized(self) -> None:
        self.assertEqual(normalize_team("Manchester United FC"), "Man United")
        self.assertEqual(normalize_team("Nottingham Forest"), "Nott'm Forest")
        self.assertEqual(normalize_team("Wolverhampton Wanderers FC"), "Wolves")

    def test_legacy_api_snapshot_is_supported(self) -> None:
        current = load_current_file(REPO_ROOT / "data" / "matches.csv")
        self.assertEqual(len(current), 40)
        self.assertEqual(set(current["season"]), {"2025-26"})
        self.assertTrue(current["result"].isin(["H", "D", "A"]).all())


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matches = pd.DataFrame(
            [
                {"season": "2025-26", "date": "2025-08-01", "home_team": "Alpha", "away_team": "Beta", "home_goals": 4, "away_goals": 0, "result": "H", "source": "test"},
                {"season": "2025-26", "date": "2025-08-08", "home_team": "Beta", "away_team": "Alpha", "home_goals": 1, "away_goals": 1, "result": "D", "source": "test"},
            ]
        )

    def test_first_match_has_no_future_information(self) -> None:
        features = build_features(self.matches)
        first = features.iloc[0]
        self.assertEqual(first["home_games_played"], 0)
        self.assertEqual(first["away_games_played"], 0)
        self.assertEqual(first["home_points"], 0)
        self.assertEqual(first["home_goal_difference"], 0)

    def test_second_match_uses_only_previous_result(self) -> None:
        features = build_features(self.matches)
        second = features.iloc[1]
        self.assertEqual(second["home_points"], 0)
        self.assertEqual(second["away_points"], 3)
        self.assertEqual(second["away_goal_difference"], 4)
        self.assertEqual(second["away_form_last_3"], 1.0)
        self.assertEqual(second["home_rest_days"], 7)


class LeagueTableTests(unittest.TestCase):
    def test_table_uses_standard_order_and_includes_post_match_elo(self) -> None:
        matches = pd.DataFrame(
            [
                {"season": "2025-26", "date": "2025-08-01", "home_team": "Alpha", "away_team": "Beta", "home_goals": 2, "away_goals": 0, "result": "H"},
                {"season": "2025-26", "date": "2025-08-02", "home_team": "Gamma", "away_team": "Delta", "home_goals": 1, "away_goals": 1, "result": "D"},
                {"season": "2025-26", "date": "2025-08-08", "home_team": "Beta", "away_team": "Gamma", "home_goals": 3, "away_goals": 0, "result": "H"},
            ]
        )

        table = build_league_table(matches)

        self.assertEqual(list(table["team"]), ["Alpha", "Beta", "Delta", "Gamma"])
        self.assertEqual(list(table["position"]), [1, 2, 3, 4])
        self.assertEqual(int(table.loc[table["team"] == "Beta", "goal_difference"].iloc[0]), 1)
        self.assertEqual(int(table.loc[table["team"] == "Gamma", "points"].iloc[0]), 1)
        self.assertFalse((table["elo"] == 1500).all())

    def test_requested_season_is_selected(self) -> None:
        matches = pd.DataFrame(
            [
                {"season": "2024-25", "date": "2024-08-01", "home_team": "Alpha", "away_team": "Beta", "home_goals": 1, "away_goals": 0, "result": "H"},
                {"season": "2025-26", "date": "2025-08-01", "home_team": "Alpha", "away_team": "Beta", "home_goals": 0, "away_goals": 2, "result": "A"},
            ]
        )

        table = build_league_table(matches, "2024-25")

        self.assertEqual(set(table["season"]), {"2024-25"})
        self.assertEqual(table.iloc[0]["team"], "Alpha")


if __name__ == "__main__":
    unittest.main()
