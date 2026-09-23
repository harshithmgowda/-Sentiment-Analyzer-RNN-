import sys
from pathlib import Path
import torch

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
        MAX_SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        PAD_TOKEN,
        UNK_TOKEN,
        PAD_IDX,
        UNK_IDX
    )
    from src.dataset import clean_text
    from src.vocabulary import load_vocabulary
    from src.model import SentimentRNN
except ImportError:
    from config import (
        DEVICE,
        MODEL_PATH,
        BEST_MODEL_PATH,
        MAX_SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        PAD_TOKEN,
        UNK_TOKEN,
        PAD_IDX,
        UNK_IDX
    )
    from dataset import clean_text
    from vocabulary import load_vocabulary
    from model import SentimentRNN

_CACHED_MODEL = None
_CACHED_VOCAB = None


def load_predictor(model_path=None):
    """
    Loads and caches the trained model and vocabulary.
    """
    global _CACHED_MODEL, _CACHED_VOCAB

    if _CACHED_MODEL is not None and _CACHED_VOCAB is not None and model_path is None:
        return _CACHED_MODEL, _CACHED_VOCAB

    if model_path is not None:
        target_path = Path(model_path)
    elif BEST_MODEL_PATH.exists():
        target_path = BEST_MODEL_PATH
    elif MODEL_PATH.exists():
        target_path = MODEL_PATH
    else:
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH} or {BEST_MODEL_PATH}.\n"
            "Please train the model first by running `python src/train.py`."
        )

    vocabulary = load_vocabulary()
    checkpoint = torch.load(target_path, map_location=DEVICE)
    vocab_size = checkpoint.get("vocab_size", len(vocabulary))

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

    _CACHED_MODEL = model
    _CACHED_VOCAB = vocabulary
    return model, vocabulary


def predict_sentiment(text, model=None, vocabulary=None, threshold=0.5):
    """
    Predicts sentiment for a single review string using pre-padding.
    """
    if model is None or vocabulary is None:
        model, vocabulary = load_predictor()

    cleaned = clean_text(text)
    tokens = cleaned.split()

    pad_id = vocabulary.get(PAD_TOKEN, PAD_IDX)
    unk_id = vocabulary.get(UNK_TOKEN, UNK_IDX)

    token_ids = [vocabulary.get(token, unk_id) for token in tokens]
    token_ids = token_ids[:MAX_SEQUENCE_LENGTH]

    # Pre-padding: pad at beginning so meaningful words are at the end
    if len(token_ids) < MAX_SEQUENCE_LENGTH:
        token_ids = [pad_id] * (MAX_SEQUENCE_LENGTH - len(token_ids)) + token_ids

    tensor_input = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(tensor_input)
        probability = torch.sigmoid(logits).item()

    is_positive = probability >= threshold
    prediction = "Positive" if is_positive else "Negative"
    confidence = probability if is_positive else (1.0 - probability)

    return {
        "raw_text": text,
        "clean_text": cleaned,
        "prediction": prediction,
        "probability": probability,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "tokens_count": len(tokens)
    }


def main():
    print("=" * 60)
    print("         IMDb REVIEW SENTIMENT PREDICTOR          ")
    print("=" * 60)

    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result = predict_sentiment(input_text)
        print(f"\nReview      : \"{result['raw_text']}\"")
        print(f"Prediction  : {result['prediction']}")
        print(f"Confidence  : {result['confidence_percentage']}")
        print(f"Probability : {result['probability']:.4f}")
        return

    print("Interactive Mode. Type your review below to test sentiment.")
    print("Type 'quit' or 'exit' to terminate.\n")

    samples = [
        "This film is a breathtaking masterpiece! The performance was incredible.",
        "An absolute waste of time. Poor acting, terrible script, and boring plot."
    ]
    print("Sample Inferences:")
    for sample in samples:
        res = predict_sentiment(sample)
        print(f"  [{res['prediction']} ({res['confidence_percentage']})] \"{sample}\"")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nEnter review > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting Sentiment Predictor. Goodbye!")
                break

            result = predict_sentiment(user_input)
            icon = "🟢" if result["prediction"] == "Positive" else "🔴"
            print(f"{icon} Sentiment : {result['prediction']}")
            print(f"📊 Confidence: {result['confidence_percentage']} (prob: {result['probability']:.4f})")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
