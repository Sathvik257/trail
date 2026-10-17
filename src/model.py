from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is absent.
    torch = None
    nn = None


if torch is not None:

    class TeluguLSTM(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 2,
            dropout: float = 0.2,
            pad_idx: int = 0,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            last_hidden = output[:, -1, :]
            return self.fc(last_hidden)

else:

    class TeluguLSTM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for TeluguLSTM. Install dependencies with "
                "`pip install -r requirements.txt`."
            )


if torch is not None:
    class TeluguBiLSTM(nn.Module):
        """Encode an observed prefix in both directions; never include its target."""

        def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                     num_layers=2, dropout=0.2, pad_idx=0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        def forward(self, input_ids):
            _, (hidden, _) = self.lstm(self.embedding(input_ids))
            # Both terminal states summarize the whole observed prefix. Taking
            # output[:, -1] would use only one token of the reverse direction.
            return self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
else:
    TeluguBiLSTM = TeluguLSTM


if torch is not None:
    class TeluguCNN(nn.Module):
        """Residual gated causal CNN; three layers cover a 15-token receptive field."""
        def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128,
                     num_layers=3, dropout=0.0, pad_idx=0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.projection = nn.Conv1d(embedding_dim, hidden_dim, 1)
            self.convs = nn.ModuleList([
                nn.Conv1d(hidden_dim, 2 * hidden_dim, 3, dilation=2 ** i)
                for i in range(num_layers)
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def encode(self, input_ids):
            state = self.projection(self.embedding(input_ids).transpose(1, 2))
            for conv in self.convs:
                padded = torch.nn.functional.pad(state, (2 * conv.dilation[0], 0))
                value, gate = conv(padded).chunk(2, dim=1)
                state = (state + self.dropout(value * torch.sigmoid(gate))) * (0.5 ** 0.5)
            return state

        def forward(self, input_ids):
            return self.fc(self.encode(input_ids)[:, :, -1])
else:
    TeluguCNN = TeluguLSTM


def model_from_checkpoint(checkpoint):
    architecture = checkpoint.get("architecture", "lstm")
    if architecture == 'wordchar_cnn':
        from wordchar_model import from_checkpoint
        return from_checkpoint(checkpoint)
    if architecture not in {"lstm", "bilstm", "cnn"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    model_class = {"lstm": TeluguLSTM, "bilstm": TeluguBiLSTM, "cnn": TeluguCNN}[architecture]
    model = model_class(
        len(checkpoint["word_to_idx"]), checkpoint["embedding_dim"],
        checkpoint["hidden_dim"], checkpoint["num_layers"],
        checkpoint["dropout"], checkpoint["word_to_idx"].get("<pad>", 0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
