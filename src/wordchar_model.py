"""Character-aware prefix encoder; surface IDs are non-learned lookup indices."""
import torch
from torch import nn
from model import TeluguCNN


class TeluguWordCharCNN(TeluguCNN):
    def __init__(self, vocab_size, surface_words, surface_chars, char_count,
                 embedding_dim=64, hidden_dim=128, num_layers=3, dropout=0.0):
        super().__init__(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.register_buffer('surface_words', torch.as_tensor(surface_words, dtype=torch.long))
        self.register_buffer('surface_chars', torch.as_tensor(surface_chars, dtype=torch.long))
        self.char_embedding = nn.Embedding(char_count, 16, padding_idx=0)
        self.char_conv = nn.Conv1d(16, embedding_dim, 3, padding=1)
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def fused_embeddings(self, surface_ids):
        # Compute character features once per distinct surface word in the batch.
        unique, inverse = torch.unique(surface_ids, return_inverse=True)
        chars = self.surface_chars[unique]
        mask = chars.ne(0)
        features = torch.relu(self.char_conv(self.char_embedding(chars).transpose(1, 2)))
        features = features.masked_fill(~mask[:, None, :], -torch.inf).amax(-1)
        features = torch.where(mask.any(-1)[:, None], features, torch.zeros_like(features))
        character = features[inverse]
        word = self.embedding(self.surface_words[surface_ids])
        gate = torch.sigmoid(self.fusion(torch.cat([word, character], dim=-1)))
        fused = gate * word + (1 - gate) * character
        return fused.masked_fill(surface_ids.eq(0).unsqueeze(-1), 0)

    def encode(self, input_ids):
        state = self.projection(self.fused_embeddings(input_ids).transpose(1, 2))
        for conv in self.convs:
            padded = torch.nn.functional.pad(state, (2 * conv.dilation[0], 0))
            value, gate = conv(padded).chunk(2, dim=1)
            state = (state + self.dropout(value * torch.sigmoid(gate))) * (0.5 ** 0.5)
        return state


def from_checkpoint(cp):
    state = cp['model_state_dict']
    model = TeluguWordCharCNN(len(cp['word_to_idx']), state['surface_words'],
        state['surface_chars'], state['char_embedding.weight'].shape[0],
        cp['embedding_dim'], cp['hidden_dim'], cp['num_layers'], cp['dropout'])
    model.load_state_dict(state)
    return model
