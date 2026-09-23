import sys
import pickle
from pathlib import Path
from collections import Counter

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.config import (
        MAX_VOCAB_SIZE,
        PAD_TOKEN,
        UNK_TOKEN,
        PAD_IDX,
        UNK_IDX,
        VOCAB_PATH
    )
except ImportError:
    from config import (
        MAX_VOCAB_SIZE,
        PAD_TOKEN,
        UNK_TOKEN,
        PAD_IDX,
        UNK_IDX,
        VOCAB_PATH
    )


def build_vocabulary(dataframe, max_vocab_size=MAX_VOCAB_SIZE):
    """
    Builds a word-to-index vocabulary from the training dataframe.
    Reserves special tokens <PAD> at index 0 and <UNK> at index 1.
    """
    print("\n" + "=" * 60)
    print("BUILDING VOCABULARY")
    print("=" * 60)

    counter = Counter()

    for text in dataframe["review"]:
        tokens = str(text).split()
        counter.update(tokens)

    # Initialize special tokens
    vocabulary = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX
    }

    # Select the most common words to fit within MAX_VOCAB_SIZE
    most_common_words = counter.most_common(max_vocab_size - 2)

    for word, _ in most_common_words:
        vocabulary[word] = len(vocabulary)

    print(f"Total unique tokens found: {len(counter)}")
    print(f"Vocabulary size capped at:  {len(vocabulary)} (includes <PAD> and <UNK>)")

    return vocabulary


def save_vocabulary(vocabulary, path=VOCAB_PATH):
    """
    Serializes vocabulary dictionary to a pickle file.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as file:
        pickle.dump(vocabulary, file)

    print(f"Vocabulary saved to: {filepath}")


def load_vocabulary(path=VOCAB_PATH):
    """
    Loads serialized vocabulary from disk.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found at {filepath}. "
            "Please train the model first by running `python src/train.py`."
        )

    with open(filepath, "rb") as file:
        vocabulary = pickle.load(file)

    return vocabulary