"""Run the full pipeline: scrape -> clean/merge -> train -> evaluate.

Usage: python -m scripts.run_pipeline [--season 2023-2024] [--skip-scrape]

--skip-scrape reuses whatever CSVs are already in data/raw/, useful
while iterating on the cleaning/model steps without re-hitting FBref
and Transfermarkt every time.
"""
import argparse

from src.clean_merge import build_players_table
from src.evaluate import plot_actual_vs_predicted, plot_residuals
from src.scrape_fbref import fetch_season_stats
from src.scrape_transfermarkt import fetch_all_clubs
from src.train_model import train

RAW_FBREF_CSV = "data/raw/fbref_{season}.csv"
RAW_TRANSFERMARKT_CSV = "data/raw/transfermarkt_squads.csv"
PROCESSED_CSV = "data/processed/players.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2023-2024")
    parser.add_argument("--skip-scrape", action="store_true")
    args = parser.parse_args()

    fbref_csv = RAW_FBREF_CSV.format(season=args.season)

    if not args.skip_scrape:
        print(f"Scraping FBref stats for {args.season}...")
        fetch_season_stats(args.season).to_csv(fbref_csv, index=False)

        print("Scraping Transfermarkt squads...")
        fetch_all_clubs().to_csv(RAW_TRANSFERMARKT_CSV, index=False)

    print("Cleaning and merging...")
    build_players_table(fbref_csv, RAW_TRANSFERMARKT_CSV, PROCESSED_CSV)

    print("Training model...")
    _, metrics, (_, y_test, predictions) = train(PROCESSED_CSV)
    print(f"MAE: €{metrics['mae']:,.0f}  R^2: {metrics['r2']:.3f}")

    plot_actual_vs_predicted(y_test, predictions)
    plot_residuals(y_test, predictions)
    print("Done. Model in models/, plots in reports/, run `streamlit run dashboard/app.py` to view the dashboard.")


if __name__ == "__main__":
    main()
