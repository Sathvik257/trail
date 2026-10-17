"""Pure neural Telugu next-word predictions, with no rules or corpus lookup."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from model import model_from_checkpoint
from text_utils import tokenize, project_root, configure_utf8_stdio


class TeluguNeuralPredictor:
    def __init__(self, checkpoint_path):
        torch.set_num_threads(2)
        self.checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        self.model = model_from_checkpoint(self.checkpoint).eval()
        self.vocab = self.checkpoint['word_to_idx']

    def predict(self, text, top_k=5):
        words = tokenize(text, language='telugu')[-self.checkpoint['seq_length']:]
        if not words:
            return dict(predictions=[], unknown_words=[], context=[])
        unknown = [word for word in words if word not in self.vocab]
        if len(unknown) == len(words):
            return dict(predictions=[], unknown_words=unknown, context=words)
        ids = [self.vocab.get(word, 1) for word in words]
        ids = [0] * (self.checkpoint['seq_length'] - len(ids)) + ids
        with torch.inference_mode():
            logits = self.model(torch.tensor([ids]))[0]
            # Match the test protocol: remove special tokens for word rankings;
            # show original softmax probabilities, without renormalizing them.
            probabilities = logits.softmax(-1)
            logits[:2] = -torch.inf
            indices = logits.topk(min(max(1, top_k), len(self.vocab) - 2)).indices
        return dict(predictions=[dict(word=self.checkpoint['idx_to_word'][int(i)],
                                      probability=float(probabilities[i])) for i in indices],
                    unknown_words=unknown, context=words)


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['lstm', 'bilstm'], default='lstm')
    parser.add_argument('--text', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    path = project_root() / 'outputs' / 'telugu_comparison' / f'{args.model}_best.pt'
    if not path.exists():
        raise SystemExit(f'No completed epoch checkpoint yet: {path}')
    print(json.dumps(TeluguNeuralPredictor(path).predict(args.text, args.top_k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

