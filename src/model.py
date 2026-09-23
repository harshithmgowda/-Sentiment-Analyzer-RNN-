import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.config import (
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        PAD_IDX
    )
except ImportError:
    from config import (
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT,
        RNN_TYPE,
        BIDIRECTIONAL,
        PAD_IDX
    )


class SentimentRNN(nn.Module):
    """
    Enhanced Recurrent Neural Network for Sentiment Classification.
    
    Supports:
      - LSTM (default), GRU, or Vanilla RNN
      - Bidirectional processing (reads forward & backward)
      - Concatenated forward/backward top-layer hidden representations
      - Multi-layer MLP classifier head with Dropout & ReLU
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=RNN_TYPE,
        bidirectional=BIDIRECTIONAL
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = str(rnn_type).lower()
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        # -----------------------------
        # Embedding Layer
        # -----------------------------
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX
        )

        # -----------------------------
        # Recurrent Core (LSTM / GRU / RNN)
        # -----------------------------
        rnn_kwargs = dict(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(**rnn_kwargs)

        # -----------------------------
        # Regularization & Classification Head
        # -----------------------------
        self.dropout = nn.Dropout(dropout)
        
        # Combined feature size from recurrent layers
        feature_dim = hidden_dim * self.num_directions

        # 2-layer MLP head with non-linearity for rich decision boundaries
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text):
        """
        Forward pass.
        
        Args:
            text: Tensor of shape [batch_size, sequence_length]
        Returns:
            Tensor of raw logits of shape [batch_size]
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(text)

        if self.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
        if self.bidirectional:
            # Concatenate the top layer's forward (hidden[-2]) and backward (hidden[-1]) states
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            last_hidden = hidden[-1, :, :]

        dropped = self.dropout(last_hidden)
        logits = self.classifier(dropped)

        return logits.squeeze(-1)