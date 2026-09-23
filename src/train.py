import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.config import (
        DEVICE,
        NUM_EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        MODEL_PATH,
        BEST_MODEL_PATH,
        OUTPUTS_DIR,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        MAX_SEQUENCE_LENGTH,
        get_device_info
    )
    from src.dataset import (
        get_or_create_splits,
        create_dataloaders
    )
    from src.vocabulary import (
        build_vocabulary,
        save_vocabulary
    )
    from src.model import SentimentRNN
    from src.utils import save_training_plot
except ImportError:
    from config import (
        DEVICE,
        NUM_EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        MODEL_PATH,
        BEST_MODEL_PATH,
        OUTPUTS_DIR,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        MAX_SEQUENCE_LENGTH,
        get_device_info
    )
    from dataset import (
        get_or_create_splits,
        create_dataloaders
    )
    from vocabulary import (
        build_vocabulary,
        save_vocabulary
    )
    from model import SentimentRNN
    from utils import save_training_plot


# ==========================================
# TRAINING STEP (ONE EPOCH)
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for text, labels in dataloader:
        text = text.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(text)
        loss = criterion(predictions, labels)

        # Backward pass with gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        # Binary accuracy (Sigmoid >= 0.5)
        predicted_classes = (torch.sigmoid(predictions) >= 0.5).float()
        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return average_loss, accuracy


# ==========================================
# VALIDATION EVALUATION
# ==========================================
def evaluate_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for text, labels in dataloader:
            text = text.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(text)
            loss = criterion(predictions, labels)

            total_loss += loss.item()

            predicted_classes = (torch.sigmoid(predictions) >= 0.5).float()
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return average_loss, accuracy


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def main():
    print("=" * 70)
    print("       HIGH-ACCURACY RNN/LSTM SENTIMENT TRAINING (PyTorch)        ")
    print("=" * 70)
    print(f"Device Selected  : {DEVICE}")
    print(f"Hardware Info    : {get_device_info()}")
    print(f"Architecture     : {RNN_TYPE.upper()} (Bidirectional={BIDIRECTIONAL})")
    print("=" * 70)

    start_time = time.time()

    # 1. Load Data
    train_data, val_data, test_data = get_or_create_splits(force_reprocess=False)

    # 2. Vocabulary
    vocabulary = build_vocabulary(train_data)
    save_vocabulary(vocabulary)

    # 3. DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        vocabulary
    )

    # 4. Model Setup
    model = SentimentRNN(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=RNN_TYPE,
        bidirectional=BIDIRECTIONAL
    )
    model = model.to(DEVICE)

    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")

    # 5. Loss, Optimizer, and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    # 6. Training Loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_accuracy = 0.0

    print("\n" + "=" * 70)
    print(f"Starting Training for {NUM_EPOCHS} Epochs on {DEVICE}...")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_acc = evaluate_epoch(
            model,
            val_loader,
            criterion
        )

        # Step the learning rate scheduler based on validation accuracy
        scheduler.step(val_acc)

        epoch_duration = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocabulary),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "rnn_type": RNN_TYPE,
            "bidirectional": BIDIRECTIONAL,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "epoch": epoch + 1,
            "val_accuracy": val_acc
        }

        # Always save latest
        torch.save(checkpoint_data, MODEL_PATH)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(checkpoint_data, BEST_MODEL_PATH)
            print(f"  ⭐ New best validation accuracy: {best_val_accuracy * 100:.2f}% (Saved to {BEST_MODEL_PATH.name})")

    total_training_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training Complete in {total_training_time / 60:.2f} minutes!")
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Best Model Saved To: {BEST_MODEL_PATH}")
    print("=" * 70)

    # 7. Generate & Save Training Plots
    loss_plot_path = OUTPUTS_DIR / "training_loss.png"
    acc_plot_path = OUTPUTS_DIR / "accuracy.png"

    save_training_plot(
        train_losses,
        val_losses,
        "Training vs Validation Loss",
        "Loss",
        loss_plot_path
    )

    save_training_plot(
        train_accuracies,
        val_accuracies,
        "Training vs Validation Accuracy",
        "Accuracy",
        acc_plot_path
    )

    print("\nNext step: Run `python src/evaluate.py` to evaluate your upgraded model on test data!")


if __name__ == "__main__":
    main()