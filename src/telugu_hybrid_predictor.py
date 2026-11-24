"""Serve the validation-selected Telugu hybrid using its fitted artifacts."""
import pickle
import torch
from telugu_neural_predictor import TeluguNeuralPredictor
from train_telugu_hybrid_long import long_distribution


class TeluguHybridPredictor(TeluguNeuralPredictor):
    def __init__(self, checkpoint_path, counts_path):
        super().__init__(checkpoint_path)
        with counts_path.open("rb") as handle:
            artifact = pickle.load(handle)
        if artifact["data_hashes"] != self.checkpoint["experiment"]["data_hashes"]:
            raise ValueError("Hybrid components use different data")
        self.counts = artifact["counts"]
        self.unigram = artifact["unigram"]

    def word_probabilities(self, ids):
        neural = super().word_probabilities(ids)
        empirical = torch.zeros_like(neural)
        indices, values = zip(*long_distribution(ids, self.counts, self.unigram))
        empirical[list(indices)] = torch.tensor(values, dtype=neural.dtype)
        return 0.5 * neural + 0.5 * empirical
