"""Pull player market value, contract and bio data from Transfermarkt.

Transfermarkt has no official API, so this scrapes the public squad
pages with requests + BeautifulSoup. Transfermarkt sometimes rate-limits
or blocks requests that don't look like a real browser, so a realistic
User-Agent and a short delay between requests are used. If a request
gets blocked, wait a while and try again, or lower the request rate.
"""
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.transfermarkt.com"
LEAGUE_URL = f"{BASE_URL}/premier-league/startseite/wettbewerb/GB1"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
REQUEST_DELAY_SECONDS = 3


def _get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(response.text, "lxml")


def parse_market_value(text: str) -> float | None:
    """"€45.00m" -> 45_000_000.0, "€800k" -> 800_000.0"""
    if not text:
        return None
    match = re.search(r"€\s*([\d.,]+)\s*([mk])?", text.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    unit = (match.group(2) or "").lower()
    if unit == "m":
        return number * 1_000_000
    if unit == "k":
        return number * 1_000
    return number


def parse_height_cm(text: str) -> float | None:
    """"1,85 m" -> 185.0"""
    if not text:
        return None
    match = re.search(r"([\d]+)[.,](\d+)", text.strip())
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}") * 100


def get_club_links() -> list[tuple[str, str]]:
    """Return [(club_name, club_squad_url), ...] for every Premier League club."""
    soup = _get_soup(LEAGUE_URL)
    links = []
    for anchor in soup.select("td.hauptlink a[href*='/startseite/verein/']"):
        club_name = anchor.get_text(strip=True)
        club_id_match = re.search(r"/verein/(\d+)", anchor["href"])
        if not club_name or not club_id_match:
            continue
        club_slug = anchor["href"].split("/")[1]
        club_id = club_id_match.group(1)
        squad_url = f"{BASE_URL}/{club_slug}/kader/verein/{club_id}/plus/1"
        links.append((club_name, squad_url))
    return list(dict.fromkeys(links))  # de-duplicate, keep order


def get_squad_details(club_name: str, squad_url: str) -> pd.DataFrame:
    """Scrape one club's detailed squad table: name, position, age,
    nationality, height, foot, contract expiry, market value."""
    soup = _get_soup(squad_url)
    table = soup.select_one("table.items")
    if table is None:
        raise ValueError(f"No squad table found at {squad_url}")

    rows = []
    for row in table.select("tbody > tr"):
        name_cell = row.select_one("td.posrela .hauptlink a")
        if not name_cell:
            continue
        position = row.select_one("td.posrela table tr:nth-of-type(2) td")
        market_value_cell = row.select_one("td.rechts.hauptlink a")

        cells = row.select("td")
        rows.append({
            "Player": name_cell.get_text(strip=True),
            "Club": club_name,
            "Position": position.get_text(strip=True) if position else None,
            "DateOfBirth_Age": cells[4].get_text(" ", strip=True) if len(cells) > 4 else None,
            "Nationality": cells[5].get("title") if len(cells) > 5 and cells[5].find("img") else None,
            "Height": parse_height_cm(cells[6].get_text(strip=True)) if len(cells) > 6 else None,
            "Foot": cells[7].get_text(strip=True) if len(cells) > 7 else None,
            "ContractExpires": cells[9].get_text(strip=True) if len(cells) > 9 else None,
            "MarketValue": parse_market_value(market_value_cell.get_text()) if market_value_cell else None,
        })
    return pd.DataFrame(rows)


def fetch_all_clubs() -> pd.DataFrame:
    clubs = get_club_links()
    frames = [get_squad_details(name, url) for name, url in clubs]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = fetch_all_clubs()
    df.to_csv("data/raw/transfermarkt_squads.csv", index=False)
    print(f"Saved {len(df)} player rows to data/raw/transfermarkt_squads.csv")
