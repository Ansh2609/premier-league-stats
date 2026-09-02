"""Generate small synthetic FBref-shaped and Transfermarkt-shaped CSVs
for testing the pipeline without hitting either website. Not real data —
just plausible numbers so clean_merge/features/train/evaluate can be
exercised end-to-end.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

PLAYERS = [
    ("Erling Haaland", "Manchester City", "FW", 24, "Right"),
    ("Kevin De Bruyne", "Manchester City", "MF", 33, "Right"),
    ("Mohamed Salah", "Liverpool", "FW", 32, "Left"),
    ("Virgil van Dijk", "Liverpool", "DF", 33, "Right"),
    ("Bukayo Saka", "Arsenal", "FW", 23, "Left"),
    ("Declan Rice", "Arsenal", "MF", 25, "Right"),
    ("Cole Palmer", "Chelsea", "MF", 22, "Left"),
    ("Bruno Fernandes", "Manchester United", "MF", 30, "Right"),
    ("Son Heung-min", "Tottenham Hotspur", "FW", 32, "Right"),
    ("Alexander Isak", "Newcastle United", "FW", 25, "Right"),
    ("James Maddison", "Tottenham Hotspur", "MF", 27, "Right"),
    ("Ollie Watkins", "Aston Villa", "FW", 28, "Right"),
    ("Martin Odegaard", "Arsenal", "MF", 26, "Left"),
    ("Rodri", "Manchester City", "MF", 28, "Right"),
    ("William Saliba", "Arsenal", "DF", 23, "Right"),
    ("Trent Alexander-Arnold", "Liverpool", "DF", 26, "Right"),
    ("Phil Foden", "Manchester City", "MF", 24, "Left"),
    ("Marcus Rashford", "Manchester United", "FW", 27, "Right"),
    ("Jarrod Bowen", "West Ham United", "FW", 27, "Right"),
    ("Bernardo Silva", "Manchester City", "MF", 30, "Right"),
]


def build_fbref_sample() -> pd.DataFrame:
    rows = []
    for name, squad, pos, age, _ in PLAYERS:
        minutes = int(RNG.integers(900, 3300))
        rows.append({
            "Player": name,
            "Squad": squad,
            "Nation": "n/a",
            "Pos": pos,
            "Age": age,
            "Born": 2024 - age,
            "Min_standard": minutes,
            "Gls_standard": int(RNG.poisson(6 if pos == "FW" else 2)),
            "Ast_standard": int(RNG.poisson(4 if pos == "MF" else 2)),
            "Cmp_passing": int(RNG.integers(300, 2500)),
            "Tkl_defense": int(RNG.integers(10, 90 if pos == "DF" else 50)),
            "Succ_possession": int(RNG.integers(5, 80)),
            "CrdY_misc": int(RNG.integers(0, 10)),
            "CrdR_misc": int(RNG.integers(0, 2)),
            "Season": "2023-2024",
        })
    return pd.DataFrame(rows)


def build_transfermarkt_sample() -> pd.DataFrame:
    rows = []
    for name, squad, pos, age, foot in PLAYERS:
        base_value = RNG.uniform(20_000_000, 180_000_000)
        rows.append({
            "Player": name,
            "Club": squad,
            "Position": pos,
            "DateOfBirth_Age": f"age {age}",
            "Nationality": "n/a",
            "Height": RNG.uniform(170, 195),
            "Foot": foot,
            "ContractExpires": pd.Timestamp("2024-01-01") + pd.DateOffset(years=int(RNG.integers(1, 5))),
            "MarketValue": round(base_value, -5),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build_fbref_sample().to_csv("data/sample/fbref_sample.csv", index=False)
    build_transfermarkt_sample().to_csv("data/sample/transfermarkt_sample.csv", index=False)
    print("Wrote data/sample/fbref_sample.csv and data/sample/transfermarkt_sample.csv")
