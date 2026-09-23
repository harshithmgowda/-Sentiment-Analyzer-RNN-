import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def save_training_plot(train_values, validation_values, title, ylabel, filename):
    """
    Saves a line plot comparing training and validation metric values over epochs.
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_values, label="Train", marker="o", linewidth=2)
    plt.plot(epochs, validation_values, label="Validation", marker="s", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved to: {filepath}")


def save_confusion_matrix(matrix, class_names, filename):
    """
    Plots and saves a high-resolution Confusion Matrix heatmap.
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))

    # Calculate percentages for cell annotation
    matrix_normalized = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
    labels = np.asarray([
        [f"{val}\n({pct:.1%})" for val, pct in zip(row_val, row_pct)]
        for row_val, row_pct in zip(matrix, matrix_normalized)
    ])

    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 12}
    )

    plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Sentiment", fontsize=12, labelpad=10)
    plt.ylabel("Actual Sentiment", fontsize=12, labelpad=10)
    plt.tight_layout()

    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {filepath}")