"""Clean the raw FBref and Transfermarkt CSVs and merge them into one
players table, matched on player name.

The two sites don't spell names identically (accents, nicknames, extra
initials), so exact string matching drops a lot of players. Instead we
fuzzy-match each FBref player name against the Transfermarkt names and
keep the best match above a similarity threshold.
"""
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

NAME_MATCH_THRESHOLD = 85  # 0-100, higher = stricter


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(stripped.lower().split())


def match_names(fbref_names: pd.Series, transfermarkt_names: pd.Series) -> pd.Series:
    """For each FBref name, return the best-matching Transfermarkt name
    (or None if nothing clears the similarity threshold)."""
    candidates = transfermarkt_names.tolist()
    matches = []
    for name in fbref_names:
        result = process.extractOne(name, candidates, scorer=fuzz.WRatio)
        matches.append(result[0] if result and result[1] >= NAME_MATCH_THRESHOLD else None)
    return pd.Series(matches, index=fbref_names.index)


def merge_datasets(fbref_df: pd.DataFrame, transfermarkt_df: pd.DataFrame) -> pd.DataFrame:
    fbref_df = fbref_df.copy()
    transfermarkt_df = transfermarkt_df.copy()

    fbref_df["norm_name"] = fbref_df["Player"].apply(normalize_name)
    transfermarkt_df["norm_name"] = transfermarkt_df["Player"].apply(normalize_name)
    transfermarkt_df = transfermarkt_df.drop_duplicates(subset="norm_name")

    fbref_df["matched_name"] = match_names(fbref_df["norm_name"], transfermarkt_df["norm_name"])

    merged = fbref_df.merge(
        transfermarkt_df,
        left_on="matched_name",
        right_on="norm_name",
        suffixes=("", "_tm"),
        how="inner",
    )
    return merged.drop(columns=["norm_name", "norm_name_tm", "matched_name"], errors="ignore")


def clean_players(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ContractExpires"] = pd.to_datetime(df["ContractExpires"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Market value is our prediction target, so a row without one is useless.
    df = df.dropna(subset=["MarketValue"])

    stat_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in ("Gls", "Ast", "Tkl", "Cmp", "Cards", "Min")
    )]
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Height"] = df["Height"].fillna(df["Height"].median())
    df["Foot"] = df["Foot"].fillna("Unknown")

    return df.reset_index(drop=True)


def build_players_table(fbref_csv: str, transfermarkt_csv: str, output_csv: str) -> pd.DataFrame:
    fbref_df = pd.read_csv(fbref_csv)
    transfermarkt_df = pd.read_csv(transfermarkt_csv)
    merged = merge_datasets(fbref_df, transfermarkt_df)
    cleaned = clean_players(merged)
    cleaned.to_csv(output_csv, index=False)
    return cleaned


if __name__ == "__main__":
    result = build_players_table(
        fbref_csv="data/raw/fbref_2023-2024.csv",
        transfermarkt_csv="data/raw/transfermarkt_squads.csv",
        output_csv="data/processed/players.csv",
    )
    print(f"Saved {len(result)} merged player rows to data/processed/players.csv")
