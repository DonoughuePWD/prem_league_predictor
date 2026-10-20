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
python src/train_model.py --refit-all
python src/simulate_season.py
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

`train_model.py` uses a scikit-learn pipeline containing median imputation,
standardisation, and multinomial logistic regression. The `--refit-all` option
retains the 2025-26 holdout metrics, then retrains the saved pipeline on all
completed historical seasons. `simulate_season.py` fixes
completed 2026-27 results, predicts and simulates every remaining home/away
fixture, and writes `data/processed/2026-27-projection.csv`. The default 1,000
runs report expected position, points and final Elo plus title, top-four and
relegation probabilities. Use `--simulations 10000` for a more stable run.

## Dashboard

The simulation writes a versioned JSON data contract used by the Streamlit
dashboard. The weekly workflow publishes it to
`dashboard_data/forecast.json`; Azure deployments additionally copy it to
private Blob Storage. Run the dashboard locally with:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Locally and on Streamlit Community Cloud, the app reads
`dashboard_data/forecast.json`. In Azure, it reads the private
`forecasts/latest.json` blob using the App Service managed identity.

## Azure deployment

Azure provides the public App Service dashboard, private Blob Storage for the
latest and historical forecasts, managed identity/RBAC, and Application
Insights monitoring. GitHub Actions performs weekly data refresh, validation,
training, 1,000 season simulations, blob publication, and dashboard deployment.

Prerequisites:

- An Azure subscription and Azure CLI
- Permission to create resource groups, Entra applications, role assignments,
  App Service, Storage, Log Analytics, and Application Insights resources
- The repository name in `owner/repository` form

After `az login`, provision the resources and GitHub OIDC identity once:

```powershell
.\scripts\setup_azure.ps1 `
  -SubscriptionId "your-subscription-id" `
  -GitHubRepository "DonoughuePWD/prem_league_predictor"
```

The script prints the three GitHub Action secrets and two repository variables
to add under **Settings → Secrets and variables → Actions**. Also add
`FOOTBALL_DATA_API_KEY` as a repository secret. No Azure password, storage key,
or publish profile is stored in GitHub.

The default App Service SKU is `F1` for a low-cost demonstration deployment.
Use `-AppServiceSku B1` for an always-on dashboard with fewer platform limits.

Workflows:

- `deploy-dashboard.yml` tests and deploys dashboard changes to App Service.
- `weekly-forecast.yml` runs at 03:17 UTC each Tuesday, commits the refreshed
  public forecast to GitHub, and can also be started manually from Actions.

The Bicep deployment outputs the public dashboard URL in the form
`https://<app-name>.azurewebsites.net`.
