import os
import sys
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.config import (
        DATASET_DIR,
        DATASET_PATH
    )
except ImportError:
    from config import (
        DATASET_DIR,
        DATASET_PATH
    )

DATASET_URL = (
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
)

ARCHIVE_PATH = DATASET_DIR / "aclImdb_v1.tar.gz"
EXTRACTED_DIR = DATASET_DIR / "aclImdb"
CSV_PATH = DATASET_PATH


def download_dataset():
    if ARCHIVE_PATH.exists():
        print(f"Archive already exists at: {ARCHIVE_PATH}")
        return

    print("=" * 60)
    print("DOWNLOADING IMDb DATASET (aclImdb_v1.tar.gz)")
    print("=" * 60)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
    print("Download completed! ✅")


def extract_dataset():
    if EXTRACTED_DIR.exists():
        print(f"Extracted dataset folder already exists at: {EXTRACTED_DIR}")
        return

    print("\nExtracting dataset archive...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(DATASET_DIR)
    print("Extraction completed! ✅")


def read_reviews():
    reviews = []
    sentiments = []

    for split in ["train", "test"]:
        for sentiment in ["pos", "neg"]:
            folder = EXTRACTED_DIR / split / sentiment
            print(f"Reading reviews from: {folder}...")

            if not folder.exists():
                print(f"Warning: {folder} does not exist. Skipping.")
                continue

            for filename in os.listdir(folder):
                if not filename.endswith(".txt"):
                    continue

                filepath = folder / filename
                with open(filepath, "r", encoding="utf-8") as file:
                    review = file.read().strip()

                reviews.append(review)
                sentiments.append("positive" if sentiment == "pos" else "negative")

    return reviews, sentiments


def create_csv():
    if CSV_PATH.exists():
        print(f"\nCSV already exists at: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        print(f"Total reviews in existing CSV: {len(df)}")
        return

    reviews, sentiments = read_reviews()
    if not reviews:
        print("Error: No reviews could be read from extracted folder.")
        return

    dataframe = pd.DataFrame({
        "review": reviews,
        "sentiment": sentiments
    })

    dataframe.to_csv(CSV_PATH, index=False)
    print("\nCSV created successfully! ✅")
    print(f"Total reviews: {len(dataframe)}")
    print(dataframe["sentiment"].value_counts())


def main():
    print("=" * 60)
    print("IMDb DATASET SETUP")
    print("=" * 60)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        print(f"Dataset CSV already present at: {CSV_PATH}")
        print("No download needed! Ready for training. ✅")
        return

    download_dataset()
    extract_dataset()
    create_csv()

    print("\nDataset ready! 🎉")
    print(f"Location: {CSV_PATH}")


if __name__ == "__main__":
    main()