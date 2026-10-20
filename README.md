# Premier League predictor

Data preparation and modelling pipeline for predicting the 2026-27 English
Premier League.

## Pipeline

Run commands from the repository root:

```powershell
python src/download_historical.py
$env:FOOTBALL_DATA_API_KEY = "your-key"
python src/fetch_current.py
python src/prepare_matches.py
python src/build_features.py
python src/build_league_table.py
python src/train_model.py
python -m unittest discover -s tests -v
```

Historical raw files are stored separately by season in `data/historical`.
The normalized and feature-engineered datasets are generated in
`data/processed`. Model artefacts are generated in `models`.

Features are calculated using information available before each fixture. The
current baseline includes league points and position, goal difference,
wins/draws/losses, points per game, three- and five-match form, recent scoring
and conceding rates, home/away form, rest days, season progress, and Elo rating.

`build_league_table.py` creates `data/processed/league_table.csv` for the latest
available season. It uses standard points, goal difference, and goals scored
ordering, and adds points-per-game, five-match form, home/away points-per-game,
current Elo, and Elo rank. Pass `--season 2024-25` to build an earlier season.
