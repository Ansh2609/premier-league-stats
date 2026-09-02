"""Pull Premier League player stat tables from FBref.

FBref pages are plain HTML tables, but the "extra" stat tables (passing,
defense, possession, misc) are wrapped inside HTML comments so that naive
scrapers skip them. We strip the comment markers before handing the page
to pandas.

FBref asks scrapers to keep requests to roughly one every few seconds, so
every fetch here sleeps briefly after the request.
"""
import io
import re
import time

import pandas as pd
import requests

BASE_URL = "https://fbref.com/en/comps/9/{season}/{kind}/{season}-Premier-League-Stats"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
REQUEST_DELAY_SECONDS = 3

# (url "kind" segment, table id prefix, friendly name)
STAT_PAGES = [
    ("stats", "stats_standard", "standard"),
    ("passing", "stats_passing", "passing"),
    ("defense", "stats_defense", "defense"),
    ("possession", "stats_possession", "possession"),
    ("misc", "stats_misc", "misc"),
]


def _fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return response.text


def _extract_table(html: str, table_id_prefix: str) -> pd.DataFrame:
    """Find a table whose id starts with table_id_prefix, in the page body
    or inside an HTML comment, and return it as a DataFrame."""
    pattern = re.compile(rf'<table[^>]*id="{re.escape(table_id_prefix)}[^"]*"')
    if not pattern.search(html):
        for comment in re.findall(r"<!--(.*?)-->", html, flags=re.DOTALL):
            if pattern.search(comment):
                html = comment
                break

    for table in pd.read_html(io.StringIO(html)):
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [c[-1] for c in table.columns]
        if "Player" in table.columns:
            return table
    raise ValueError(f"Could not find a player table for id prefix '{table_id_prefix}'")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Player"] != "Player"].copy()  # drop repeated header rows
    df = df.reset_index(drop=True)
    return df


def fetch_season_stats(season: str = "2023-2024") -> pd.DataFrame:
    """Download standard, passing, defense, possession and misc stat
    tables for one Premier League season and merge them into one
    DataFrame keyed on Player + Squad.

    season format: "2023-2024"
    """
    merged = None
    for kind, table_id_prefix, label in STAT_PAGES:
        url = BASE_URL.format(season=season, kind=kind)
        html = _fetch_html(url)
        table = _extract_table(html, table_id_prefix)
        table = _flatten_columns(table)
        table = table.add_suffix(f"_{label}")
        table = table.rename(columns={
            f"Player_{label}": "Player",
            f"Squad_{label}": "Squad",
            f"Nation_{label}": "Nation",
            f"Pos_{label}": "Pos",
            f"Age_{label}": "Age",
            f"Born_{label}": "Born",
        })

        if merged is None:
            merged = table
        else:
            shared = [c for c in ("Player", "Squad") if c in table.columns]
            drop_cols = [c for c in ("Nation", "Pos", "Age", "Born") if c in table.columns]
            merged = merged.merge(table.drop(columns=drop_cols, errors="ignore"), on=shared, how="left")

    merged["Season"] = season
    return merged


if __name__ == "__main__":
    df = fetch_season_stats("2023-2024")
    df.to_csv("data/raw/fbref_2023-2024.csv", index=False)
    print(f"Saved {len(df)} player rows to data/raw/fbref_2023-2024.csv")
