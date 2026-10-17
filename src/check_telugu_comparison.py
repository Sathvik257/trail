"""Regression checks for prefix directionality, OOV scoring and resumable training."""
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from model import TeluguLSTM, TeluguBiLSTM, model_from_checkpoint
from preprocess import make_line_sequences
from train_telugu_comparison import evaluate


class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_prefix_excludes_target(self):
        vocab = {'<pad>': 0, '<unk>': 1, 'a': 2, 'b': 3, 'c': 4}
        self.assertEqual(make_line_sequences(['a', 'b', 'c'], vocab, 3),
                         [([0, 0, 2], 3), ([0, 2, 3], 4)])

    def test_both_terminal_states_and_gradients(self):
        model = TeluguBiLSTM(9, 4, 6, 2, 0)
        inputs = torch.tensor([[0, 0, 2, 3], [2, 3, 4, 5]])
        _, (hidden, _) = model.lstm(model.embedding(inputs))
        expected = model.fc(torch.cat([hidden[-2], hidden[-1]], 1))
        torch.testing.assert_close(model(inputs), expected)
        model(inputs).sum().backward()
        for name in ['weight_ih_l0', 'weight_ih_l0_reverse']:
            gradient = getattr(model.lstm, name).grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_unknown_targets_are_not_correct_words(self):
        class UnknownModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.tensor([[0., 100., 3., 2., 1., 0.]]).repeat(len(inputs), 1)
        metrics = evaluate(UnknownModel(), torch.tensor([[0, 2, 1], [0, 2, 2]]), 2)
        self.assertEqual(metrics['target_oov_rate'], 0.5)
        self.assertEqual(metrics['top1_accuracy'], 0.5)
        self.assertEqual(metrics['known_target_top1'], 1)

    def test_training_resume_and_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = {'<pad>': 0, '<unk>': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
            for split, raw in [('', 'a b c'), ('validation', 'b c d'), ('test', 'c d a')]:
                folder = root / 'data' / split
                folder.mkdir(parents=True)
                (folder / 'telugu_raw.txt').write_text(raw, encoding='utf-8')
                samples = make_line_sequences(raw.split(), vocab, 3) * 8
                with (folder / 'samples.pkl').open('wb') as handle:
                    pickle.dump(samples, handle)
                (folder / 'sequences_manifest.json').write_text(json.dumps(dict(
                    total_sequences=len(samples), seq_length=3,
                    shards=[dict(path='samples.pkl', count=len(samples))])))
            with (root / 'data' / 'vocab.pkl').open('wb') as handle:
                pickle.dump(dict(word_to_idx=vocab, idx_to_word={v:k for k,v in vocab.items()}, seq_length=3), handle)
            (root / 'data' / 'provenance.json').write_text('{}')
            for epochs in [1, 2]:
                subprocess.run([sys.executable, str(Path(__file__).with_name('train_telugu_comparison.py')),
                    '--data-dir', str(root / 'data'), '--output-dir', str(root / 'out'),
                    '--epochs', str(epochs), '--threads', '1', '--batch-size', '8'],
                    check=True, capture_output=True, text=True)
                status = json.loads((root / 'out' / 'status.json').read_text())
                self.assertEqual(status['status'], 'complete')
                for arch in ['lstm', 'bilstm']:
                    checkpoint = torch.load(root / 'out' / f'{arch}_latest.pt', weights_only=True)
                    self.assertEqual(checkpoint['epoch'], epochs)
                    self.assertEqual(len(checkpoint['history']), epochs)
                    self.assertEqual(tuple(model_from_checkpoint(checkpoint)(torch.tensor([[0, 2, 3]])).shape), (1, 6))
            for epochs in [1, 2]:
                subprocess.run([sys.executable, str(Path(__file__).with_name('train_telugu_comparison.py')),
                    '--data-dir', str(root / 'data'), '--output-dir', str(root / 'cnn'),
                    '--cnn-only', '--epochs', str(epochs), '--threads', '1', '--batch-size', '8'],
                    check=True, capture_output=True, text=True)
                checkpoint = torch.load(root / 'cnn' / 'cnn_latest.pt', weights_only=True)
                self.assertEqual(len(checkpoint['history']), epochs)
                self.assertEqual(checkpoint['epoch'], epochs)
                self.assertTrue(torch.isfinite(model_from_checkpoint(checkpoint)(torch.tensor([[0, 2, 3]]))).all())
                result = json.loads((root / 'cnn' / 'cnn_test_metrics.json').read_text())
                self.assertEqual(result['completed_epochs'], epochs)


if __name__ == '__main__':
    unittest.main()
