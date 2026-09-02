"""Streamlit dashboard: search a Premier League player, see their bio,
season stats, contract, and actual vs model-predicted transfer value.

Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_feature_matrix

PLAYERS_CSV = "data/processed/players.csv"
MODEL_PATH = "models/transfer_value_model.joblib"

STAT_LABELS = {
    "Min_standard": "Minutes played",
    "Gls_standard": "Goals",
    "Ast_standard": "Assists",
    "Cmp_passing": "Passes completed",
    "Tkl_defense": "Tackles",
    "Succ_possession": "Successful dribbles",
    "CrdY_misc": "Yellow cards",
    "CrdR_misc": "Red cards",
}


@st.cache_data
def load_players():
    return pd.read_csv(PLAYERS_CSV, parse_dates=["ContractExpires"])


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def format_euros(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"€{value:,.0f}"


st.set_page_config(page_title="PL Transfer Value Predictor", layout="wide")
st.title("Premier League Transfer Value Predictor")

try:
    players = load_players()
except FileNotFoundError:
    st.error(
        "No processed player data yet. Run `python -m scripts.run_pipeline` first "
        "to scrape, clean and train the model."
    )
    st.stop()

try:
    bundle = load_model()
    model, feature_columns = bundle["model"], bundle["feature_columns"]
except FileNotFoundError:
    model, feature_columns = None, None
    st.warning("No trained model found yet — showing stats only, no predicted value.")

player_name = st.selectbox("Search for a player", sorted(players["Player"].unique()))
player_rows = players[players["Player"] == player_name]
player = player_rows.iloc[0]

col_bio, col_stats, col_value = st.columns(3)

with col_bio:
    st.subheader("Bio")
    st.write(f"**Club:** {player.get('Club', player.get('Squad', 'N/A'))}")
    st.write(f"**Position:** {player.get('Pos', 'N/A')}")
    st.write(f"**Age:** {player.get('Age', 'N/A')}")
    st.write(f"**Height:** {player.get('Height', 'N/A'):.0f} cm" if pd.notna(player.get("Height")) else "**Height:** N/A")
    st.write(f"**Preferred foot:** {player.get('Foot', 'N/A')}")
    contract = player.get("ContractExpires")
    st.write(f"**Contract expires:** {contract.date() if pd.notna(contract) else 'N/A'}")

with col_stats:
    st.subheader("This season's stats")
    for col, label in STAT_LABELS.items():
        if col in player:
            st.write(f"**{label}:** {player[col]:.0f}" if pd.notna(player[col]) else f"**{label}:** N/A")

with col_value:
    st.subheader("Transfer value")
    st.metric("Current market value", format_euros(player.get("MarketValue")))

    if model is not None:
        X, _, _ = build_feature_matrix(player_rows, feature_columns=feature_columns)
        predicted_value = model.predict(X)[0]
        st.metric("Model-predicted value", format_euros(predicted_value))

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.bar(["Actual", "Predicted"], [player.get("MarketValue", 0), predicted_value], color=["#1f77b4", "#ff7f0e"])
        ax.set_ylabel("€")
        st.pyplot(fig)

st.divider()
st.caption(
    "Market values come from Transfermarkt, season stats from FBref. "
    "Predicted value is a linear regression model trained on this dataset — "
    "treat it as an estimate, not gospel."
)
