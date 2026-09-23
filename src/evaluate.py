import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.config import (
        DEVICE,
        MODEL_PATH,
        BEST_MODEL_PATH,
        OUTPUTS_DIR,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        get_device_info
    )
    from src.dataset import (
        get_or_create_splits,
        create_dataloaders
    )
    from src.vocabulary import load_vocabulary
    from src.model import SentimentRNN
    from src.utils import save_confusion_matrix
except ImportError:
    from config import (
        DEVICE,
        MODEL_PATH,
        BEST_MODEL_PATH,
        OUTPUTS_DIR,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        get_device_info
    )
    from dataset import (
        get_or_create_splits,
        create_dataloaders
    )
    from vocabulary import load_vocabulary
    from model import SentimentRNN
    from utils import save_confusion_matrix


def evaluate_model(model_filepath=None):
    print("=" * 70)
    print("            RNN SENTIMENT ANALYSIS EVALUATION             ")
    print("=" * 70)
    print(f"Device Selected : {DEVICE}")
    print(f"Hardware Info   : {get_device_info()}")
    print("=" * 70)

    # 1. Determine which model checkpoint to evaluate
    if model_filepath is None:
        if BEST_MODEL_PATH.exists():
            checkpoint_file = BEST_MODEL_PATH
        elif MODEL_PATH.exists():
            checkpoint_file = MODEL_PATH
        else:
            raise FileNotFoundError(
                f"No model checkpoint found at {MODEL_PATH} or {BEST_MODEL_PATH}.\n"
                "Please train the model first by running `python src/train.py`."
            )
    else:
        checkpoint_file = Path(model_filepath)

    print(f"Loading checkpoint from: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)

    # 2. Load Vocabulary
    vocabulary = load_vocabulary()
    vocab_size = checkpoint.get("vocab_size", len(vocabulary))

    # 3. Load Dataset Splits & Test Loader
    train_data, val_data, test_data = get_or_create_splits(force_reprocess=False)
    _, _, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        vocabulary
    )

    # 4. Instantiate Model with Checkpoint Architecture Parameters
    model_rnn_type = checkpoint.get("rnn_type", RNN_TYPE)
    model_bidirectional = checkpoint.get("bidirectional", BIDIRECTIONAL)

    model = SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=checkpoint.get("embedding_dim", EMBEDDING_DIM),
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        num_layers=checkpoint.get("num_layers", NUM_LAYERS),
        dropout=checkpoint.get("dropout", DROPOUT),
        rnn_type=model_rnn_type,
        bidirectional=model_bidirectional
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded Architecture: {model_rnn_type.upper()} (Bidirectional={model_bidirectional})")

    # 5. Inference loop on Test Set
    predictions = []
    actual_labels = []

    print("\nRunning evaluation on test set (4,959 samples)...")
    with torch.no_grad():
        for text, labels in test_loader:
            text = text.to(DEVICE)
            logits = model(text)
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities >= 0.5).long()

            predictions.extend(predicted.cpu().numpy().astype(int))
            actual_labels.extend(labels.cpu().numpy().astype(int))

    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)

    # 6. Calculate Metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions)
    recall = recall_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions)
    class_report = classification_report(
        actual_labels,
        predictions,
        target_names=["Negative", "Positive"]
    )
    conf_matrix = confusion_matrix(actual_labels, predictions)

    # 7. Print Results
    print("\n" + "=" * 50)
    print("           TEST PERFORMANCE METRICS               ")
    print("=" * 50)
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print("=" * 50)

    print("\nDetailed Classification Report:")
    print(class_report)

    print("\nConfusion Matrix:")
    print(conf_matrix)

    # 8. Save Confusion Matrix Plot
    cm_plot_path = OUTPUTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(
        conf_matrix,
        class_names=["Negative", "Positive"],
        filename=cm_plot_path
    )

    # 9. Save Evaluation Metrics Summary to Text File
    summary_path = OUTPUTS_DIR / "evaluation_results.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Sentiment RNN Model Evaluation ===\n")
        f.write(f"Model Checkpoint: {checkpoint_file}\n")
        f.write(f"Architecture: {model_rnn_type.upper()} (Bidirectional={model_bidirectional})\n")
        f.write(f"Test Samples: {len(actual_labels)}\n\n")
        f.write(f"Accuracy  : {accuracy * 100:.2f}%\n")
        f.write(f"Precision : {precision * 100:.2f}%\n")
        f.write(f"Recall    : {recall * 100:.2f}%\n")
        f.write(f"F1 Score  : {f1 * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")

    print(f"\nEvaluation summary saved to: {summary_path}")
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist()
    }


def main():
    evaluate_model()


if __name__ == "__main__":
    main()