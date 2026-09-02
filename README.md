# Premier League Transfer Value Predictor

Look up a Premier League player and see their bio, this season's stats,
contract situation, and a linear-regression-predicted transfer value
next to their real market value.

## Data sources

No official transfer-value API exists, so this scrapes two sites:

- **[FBref](https://fbref.com)** — season stats: minutes, goals, assists,
  passes completed, tackles, successful dribbles, cards.
- **[Transfermarkt](https://www.transfermarkt.com)** — market value,
  contract expiry, age, height, preferred foot.

Both are scraped with `requests`/`BeautifulSoup`/`pandas`, matched to each
other by player name (fuzzy-matched with `rapidfuzz`, since the two sites
don't always spell names the same way).

## Pipeline

```
scrape (FBref + Transfermarkt) -> clean & merge -> feature engineering -> train -> evaluate -> dashboard
```

| Step | File |
|---|---|
| Scrape stats | `src/scrape_fbref.py` |
| Scrape value/contract/bio | `src/scrape_transfermarkt.py` |
| Clean & merge | `src/clean_merge.py` |
| Feature engineering | `src/features.py` |
| Train model | `src/train_model.py` |
| Evaluation plots | `src/evaluate.py` |
| Run everything | `scripts/run_pipeline.py` |
| Dashboard | `dashboard/app.py` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
# Scrape, clean, train, evaluate — takes a few minutes (scraping is rate-limited on purpose)
python -m scripts.run_pipeline --season 2023-2024

# Then launch the dashboard
streamlit run dashboard/app.py
```

Re-running the pipeline without re-scraping (once `data/raw/` is populated):

```bash
python -m scripts.run_pipeline --season 2023-2024 --skip-scrape
```

## Trying it without scraping

`data/sample/` has a small synthetic dataset (20 made-up players, random
stats — not real numbers) shaped exactly like the real scraped output, so
you can exercise the whole pipeline without hitting either website:

```bash
python -c "
from src.clean_merge import build_players_table
from src.train_model import train
from src.evaluate import plot_actual_vs_predicted, plot_residuals

build_players_table('data/sample/fbref_sample.csv', 'data/sample/transfermarkt_sample.csv', 'data/processed/players.csv')
_, metrics, (X_test, y_test, preds) = train('data/processed/players.csv')
print(metrics)
plot_actual_vs_predicted(y_test, preds)
plot_residuals(y_test, preds)
"
streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/
```

These run against `data/sample/`, so they don't need network access.

## Notes / known limitations

- **Scraping is fragile by nature.** Both sites can change their HTML at
  any time, and Transfermarkt in particular may rate-limit or block
  requests that don't look like a real browser. If a scrape fails,
  wait a bit and retry, or check that the CSS selectors in
  `src/scrape_transfermarkt.py` still match the live page.
- **Be a polite scraper.** Both scripts sleep between requests
  (`REQUEST_DELAY_SECONDS`). Don't lower this to hammer either site, and
  check each site's terms of use before scraping at any real scale.
- **Linear regression is a starting point.** Market value isn't actually
  a linear function of stats — treat predictions as a rough estimate. A
  natural next step is swapping `LinearRegression` for
  `RandomForestRegressor` or `GradientBoostingRegressor` in
  `src/train_model.py` once the basic pipeline works.
- **Name matching isn't perfect.** Fuzzy matching (`NAME_MATCH_THRESHOLD`
  in `src/clean_merge.py`) will occasionally miss a player or mismatch
  two similarly-named ones. Worth spot-checking `data/processed/players.csv`.
