"""Public Streamlit dashboard for the Premier League forecast."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Premier League Forecast", page_icon="⚽", layout="wide")


@st.cache_data(ttl=300)
def load_forecast() -> dict:
    """Read the latest forecast from Azure Blob Storage or a local development file."""
    account_url = os.environ.get("STORAGE_ACCOUNT_URL")
    if account_url:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        service = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        container = os.environ.get("FORECAST_CONTAINER", "forecasts")
        blob = os.environ.get("FORECAST_BLOB", "latest.json")
        payload = service.get_blob_client(container=container, blob=blob).download_blob().readall()
        return json.loads(payload)

    path = Path(os.environ.get("FORECAST_PATH", "dashboard_data/forecast.json"))
    if not path.exists():
        st.error("No forecast is available yet. Run the training and simulation pipeline first.")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


def percentage(value: float) -> str:
    return f"{value:.1%}"


forecast = load_forecast()
table = pd.DataFrame(forecast["teams"]).sort_values("projected_position")

st.markdown(
    """
    <style>
      .stApp { background: #07130f; color: #f4f7f5; }
      [data-testid="stHeader"] { background: rgba(7, 19, 15, 0.92); }
      .hero { padding: 1.4rem 0 1rem; border-bottom: 1px solid #29443a; margin-bottom: 1rem; }
      .eyebrow { color: #79e3aa; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
      .hero h1 { color: #ffffff; font-size: clamp(2rem, 5vw, 4.2rem); line-height: .96; margin: .5rem 0; }
      .hero p { color: #afc3ba; max-width: 760px; font-size: 1.05rem; }
      [data-testid="stMetric"] { background: #10251d; border: 1px solid #29443a; border-radius: 12px; padding: 1rem; }
      [data-testid="stMetricValue"] { color: #ffffff; }
      [data-testid="stMetricLabel"] { color: #9db7aa; }
      .note { color: #8fa99c; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="hero">
      <div class="eyebrow">2026/27 model forecast</div>
      <h1>Premier League<br>season outlook</h1>
      <p>Current results are fixed. Every remaining fixture is predicted from the latest form,
      league state and Elo ratings, then replayed across {forecast['simulations']:,} complete seasons.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

leader = table.iloc[0]
most_likely_relegated = table.sort_values("relegation_probability", ascending=False).iloc[0]
metric_columns = st.columns(4)
metric_columns[0].metric("Projected champion", leader["team"])
metric_columns[1].metric("Title probability", percentage(leader["title_probability"]))
metric_columns[2].metric("Matches included", f"{forecast['completed_matches']} / 380")
metric_columns[3].metric("Highest relegation risk", most_likely_relegated["team"], percentage(most_likely_relegated["relegation_probability"]))

st.subheader("Projected final table")
display = table[
    [
        "projected_position",
        "team",
        "current_position",
        "played",
        "current_points",
        "current_elo",
        "expected_points",
        "expected_position",
        "title_probability",
        "top_4_probability",
        "relegation_probability",
    ]
].rename(
    columns={
        "projected_position": "Pos",
        "team": "Team",
        "current_position": "Now",
        "played": "P",
        "current_points": "Pts",
        "current_elo": "Elo",
        "expected_points": "Expected pts",
        "expected_position": "Avg finish",
        "title_probability": "Title",
        "top_4_probability": "Top four",
        "relegation_probability": "Relegation",
    }
)
st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Elo": st.column_config.NumberColumn(format="%.0f"),
        "Expected pts": st.column_config.NumberColumn(format="%.1f"),
        "Avg finish": st.column_config.NumberColumn(format="%.2f"),
        "Title": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Top four": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Relegation": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    },
)

left, right = st.columns(2)
with left:
    st.subheader("Title race")
    title = table.nlargest(8, "title_probability").set_index("team")[["title_probability"]]
    st.bar_chart(title, color="#79e3aa", horizontal=True, height=360)
with right:
    st.subheader("Relegation risk")
    relegation = table.nlargest(8, "relegation_probability").set_index("team")[["relegation_probability"]]
    st.bar_chart(relegation, color="#ff8b72", horizontal=True, height=360)

model = forecast.get("model", {})
st.caption(
    f"Data through {forecast['data_cutoff']} · Generated {forecast['generated_at']} · "
    f"{forecast['simulations']:,} simulations · Seed {forecast['seed']} · "
    f"Validation accuracy {model.get('accuracy', 0):.1%} · Log loss {model.get('log_loss', 0):.3f}"
)
st.markdown(
    '<p class="note">Probabilities are model estimates, not betting advice. Forecasts update when new completed matches are processed.</p>',
    unsafe_allow_html=True,
)
