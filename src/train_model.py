"""Train a linear regression model that predicts a player's market value
from their stats, bio and contract situation, and save it for the
dashboard to reuse.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import build_feature_matrix

MODEL_PATH = "models/transfer_value_model.joblib"


def train(players_csv: str, model_path: str = MODEL_PATH):
    df = pd.read_csv(players_csv)
    X, y, feature_columns = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    joblib.dump({"model": model, "feature_columns": feature_columns}, model_path)
    return model, metrics, (X_test, y_test, predictions)


if __name__ == "__main__":
    _, metrics, _ = train("data/processed/players.csv")
    print(f"Trained on {metrics['n_train']} players, tested on {metrics['n_test']}")
    print(f"MAE: €{metrics['mae']:,.0f}")
    print(f"R^2: {metrics['r2']:.3f}")
    print(f"Model saved to {MODEL_PATH}")
