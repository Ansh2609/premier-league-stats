"""Plot how well the model's predictions match real market values."""
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_test, predictions, output_path: str = "reports/actual_vs_predicted.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, predictions, alpha=0.6, edgecolor="k")
    limit = max(y_test.max(), predictions.max())
    ax.plot([0, limit], [0, limit], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual market value (€)")
    ax.set_ylabel("Predicted market value (€)")
    ax.set_title("Actual vs Predicted Transfer Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_residuals(y_test, predictions, output_path: str = "reports/residuals.png"):
    residuals = y_test - predictions
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predictions, residuals, alpha=0.6, edgecolor="k")
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted market value (€)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    from src.train_model import train

    _, metrics, (X_test, y_test, predictions) = train("data/processed/players.csv")
    plot_actual_vs_predicted(y_test, predictions)
    plot_residuals(y_test, predictions)
    print(f"MAE: €{metrics['mae']:,.0f}  R^2: {metrics['r2']:.3f}")
    print("Saved plots to reports/actual_vs_predicted.png and reports/residuals.png")
