import re
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.config import (
        DATASET_PATH,
        TRAIN_DATA_PATH,
        VAL_DATA_PATH,
        TEST_DATA_PATH,
        BATCH_SIZE,
        MAX_SEQUENCE_LENGTH,
        NUM_WORKERS,
        RANDOM_SEED,
        PAD_TOKEN,
        UNK_TOKEN,
        PAD_IDX,
        UNK_IDX
    )
except ImportError:
    from config import (
        DATASET_PATH,
        TRAIN_DATA_PATH,
        VAL_DATA_PATH,
        TEST_DATA_PATH,
        BATCH_SIZE,
        MAX_SEQUENCE_LENGTH,
        NUM_WORKERS,
        RANDOM_SEED,
        PAD_TOKEN,
        UNK_TOKEN,
        PAD_IDX,
        UNK_IDX
    )


# ==========================================
# TEXT PREPROCESSING
# ==========================================
def clean_text(text):
    """
    Cleans raw review text by removing HTML tags, URLs, special characters,
    normalizing whitespace, and converting to lowercase while preserving negation words.
    """
    if not isinstance(text, str):
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags (e.g. <br />, <div>)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Standardize contractions that carry strong sentiment
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "cannot", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # Remove non-alphanumeric characters (keep letters, digits, and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Collapse multiple spaces into single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# LOAD RAW DATASET
# ==========================================
def load_data():
    """
    Loads the full IMDb dataset from CSV, cleans texts, drops duplicates and
    missing values, and maps labels to binary integers (0 = negative, 1 = positive).
    """
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATASET")
    print("=" * 60)

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please run `python src/download_dataset.py` first to download it."
        )

    print(f"Reading from: {DATASET_PATH}")
    dataframe = pd.read_csv(DATASET_PATH)
    print(f"Initial raw samples: {len(dataframe)}")

    # Remove missing values
    dataframe = dataframe.dropna(subset=["review", "sentiment"])

    # Remove duplicate reviews
    dataframe = dataframe.drop_duplicates(subset=["review"])

    # Clean text reviews
    print("Cleaning review texts...")
    dataframe["review"] = dataframe["review"].apply(clean_text)

    # Map sentiments to binary numbers
    sentiment_map = {"negative": 0, "positive": 1}
    dataframe["sentiment"] = dataframe["sentiment"].map(sentiment_map)

    # Drop any unmapped rows
    dataframe = dataframe.dropna(subset=["sentiment"])
    dataframe["sentiment"] = dataframe["sentiment"].astype(int)

    # Reset indices
    dataframe = dataframe.reset_index(drop=True)

    print(f"Clean samples after preprocessing: {len(dataframe)}")
    print(f"Class distribution:\n{dataframe['sentiment'].value_counts()}")

    return dataframe


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================
def split_data(dataframe):
    """
    Splits the dataframe into Train (80%), Validation (10%), and Test (10%) sets
    using stratified sampling, and saves each to CSV.
    """
    print("\n" * 1 + "=" * 60)
    print("SPLITTING DATASET (80% Train, 10% Val, 10% Test)")
    print("=" * 60)

    train_data, temp_data = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=dataframe["sentiment"]
    )

    validation_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_data["sentiment"]
    )

    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    validation_data.to_csv(VAL_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    print(f"Saved splits to:")
    print(f"  - Train:      {TRAIN_DATA_PATH} ({len(train_data)} samples)")
    print(f"  - Validation: {VAL_DATA_PATH} ({len(validation_data)} samples)")
    print(f"  - Test:       {TEST_DATA_PATH} ({len(test_data)} samples)")

    return train_data, validation_data, test_data


def get_or_create_splits(force_reprocess=False):
    """
    Returns train, validation, and test dataframes.
    If the split CSVs already exist and force_reprocess is False, loads them instantly.
    Otherwise, processes the full dataset and creates new splits.
    """
    splits_exist = (
        Path(TRAIN_DATA_PATH).exists() and
        Path(VAL_DATA_PATH).exists() and
        Path(TEST_DATA_PATH).exists()
    )

    if splits_exist and not force_reprocess:
        print(f"Found existing split files in {DATASET_PATH.parent}. Loading splits...")
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        validation_data = pd.read_csv(VAL_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)

        train_data["review"] = train_data["review"].fillna("").astype(str)
        validation_data["review"] = validation_data["review"].fillna("").astype(str)
        test_data["review"] = test_data["review"].fillna("").astype(str)

        train_data["sentiment"] = train_data["sentiment"].astype(int)
        validation_data["sentiment"] = validation_data["sentiment"].astype(int)
        test_data["sentiment"] = test_data["sentiment"].astype(int)

        print(f"Loaded existing splits: Train={len(train_data)}, Val={len(validation_data)}, Test={len(test_data)}")
        return train_data, validation_data, test_data

    dataframe = load_data()
    return split_data(dataframe)


# ==========================================
# PYTORCH DATASET CLASS
# ==========================================
class IMDBDataset(Dataset):
    """
    PyTorch Dataset that maps cleaned textual reviews into sequences of integer token IDs,
    applies PRE-PADDING (pad at beginning) so the model's final state captures meaningful words.
    """
    def __init__(self, dataframe, vocabulary):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.pad_id = vocabulary.get(PAD_TOKEN, PAD_IDX)
        self.unk_id = vocabulary.get(UNK_TOKEN, UNK_IDX)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = str(self.dataframe.loc[index, "review"])
        label = float(self.dataframe.loc[index, "sentiment"])

        tokens = text.split()

        # Convert words to vocabulary indices
        token_ids = [self.vocabulary.get(token, self.unk_id) for token in tokens]

        # Truncate sequence if longer than MAX_SEQUENCE_LENGTH (keep the tail of long reviews)
        token_ids = token_ids[:MAX_SEQUENCE_LENGTH]

        # PRE-PADDING: Pad at the beginning of the sequence so meaningful words sit at the end
        if len(token_ids) < MAX_SEQUENCE_LENGTH:
            token_ids = [self.pad_id] * (MAX_SEQUENCE_LENGTH - len(token_ids)) + token_ids

        text_tensor = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return text_tensor, label_tensor


# ==========================================
# DATALOADER FACTORY
# ==========================================
def create_dataloaders(
    train_data,
    validation_data,
    test_data,
    vocabulary,
    batch_size=BATCH_SIZE
):
    train_dataset = IMDBDataset(train_data, vocabulary)
    validation_dataset = IMDBDataset(validation_data, vocabulary)
    test_dataset = IMDBDataset(test_data, vocabulary)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory
    )

    return train_loader, validation_loader, test_loader