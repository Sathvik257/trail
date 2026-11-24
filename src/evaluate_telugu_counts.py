"""Evaluate the fitted count component of Telugu hybrid v2 independently."""
import json
import pickle
from functools import lru_cache
import torch

from train_telugu_comparison import audit_data, load_split, save_json
from train_telugu_hybrid_long import ROOT, long_distribution


class TeluguCountPredictor:
    def __init__(self, artifact, vocab_size):
        self.vocab_size = vocab_size
        self.counts = artifact["counts"]
        self.unigram = artifact["unigram"]

    @lru_cache(maxsize=32768)
    def rank(self, suffix):
        entries = long_distribution(suffix, self.counts, self.unigram)
        indices, values = zip(*entries)
        scores = torch.zeros((1, self.vocab_size), dtype=torch.float32)
        scores[0, list(indices)] = torch.tensor(values, dtype=torch.float32)
        scores[:, :2] = -torch.inf
        return tuple(scores.topk(5, dim=1).indices[0].tolist())

    def predict(self, context):
        for size in (4, 3, 2, 1):
            suffix = tuple(context[-size:])
            if len(suffix) == size and suffix in self.counts[size]:
                return self.rank(suffix)
        return self.rank(())


def main():
    base = ROOT / "data_telugu_llm"
    out = ROOT / "outputs/telugu_count_only"
    with (ROOT / "outputs/telugu_hybrid_v2/counts.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    if audit_data(base) != artifact["data_hashes"]:
        raise ValueError("Dataset differs from the fitted count model")
    with (base / "vocab.pkl").open("rb") as handle:
        vocab_size = len(pickle.load(handle)["word_to_idx"])
    torch.set_num_threads(1)
    predictor = TeluguCountPredictor(artifact, vocab_size)
    results = {}
    for split in ("validation", "test"):
        rows = load_split(base / split).numpy()
        first = five = 0
        for row in rows:
            ranked = predictor.predict(row[:-1].tolist())
            target = int(row[-1])
            first += bool(ranked) and ranked[0] == target
            five += target in ranked
        results[split] = dict(examples=len(rows), top1_hits=first, top5_hits=five,
                              top1_accuracy=first / len(rows), top5_accuracy=five / len(rows))
    out.mkdir(exist_ok=True)
    save_json(out / "evaluation.json", dict(results=results, data_hashes=artifact["data_hashes"],
              tie_rule="hybrid v2 CPU float32 torch.topk(5, dim=1), IDs 0 and 1 masked"))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
