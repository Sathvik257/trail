import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from preprocess import make_line_sequences
from wordchar_data import prepare
from wordchar_model import TeluguWordCharCNN
from model import model_from_checkpoint


class WordCharChecks(unittest.TestCase):
    def test_encoding_causality_and_resume(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'data'
            words = ['నేను', 'తెలుగు', 'భాష', 'చదువు']
            mapping = {'<pad>': 0, '<unk>': 1, **{w: i+2 for i,w in enumerate(words)}}
            vocab = dict(word_to_idx=mapping, idx_to_word={i:w for w,i in mapping.items()}, seq_length=3)
            texts = ['నేను తెలుగు భాష చదువు', 'తెలుగు నేను భాష చదువు', 'నేను చదువు కొత్త']
            for name, text in zip(['', 'validation', 'test'], texts):
                folder = base / name
                folder.mkdir(parents=True)
                (folder/'telugu_raw.txt').write_text(text+'\n', encoding='utf-8')
                samples = make_line_sequences(text.split(), mapping, 3)
                with (folder/'samples.pkl').open('wb') as f: pickle.dump(samples, f)
                (folder/'sequences_manifest.json').write_text(json.dumps(dict(total_sequences=len(samples),
                    seq_length=3, shards=[dict(path='samples.pkl', count=len(samples))])))
            with (base/'vocab.pkl').open('wb') as f: pickle.dump(vocab, f)
            (base/'provenance.json').write_text('{}')
            splits, sw, sc, meta = prepare(base, vocab)
            for split, text in zip(splits, texts):
                expected = torch.tensor([x+[y] for x,y in make_line_sequences(text.split(), mapping, 3)])
                actual = torch.cat([torch.tensor(sw)[split[:,:-1]], split[:,-1:]], 1)
                torch.testing.assert_close(actual, expected)
            self.assertNotIn('క', meta['char_to_idx'])
            self.assertEqual(sw[meta['surface_to_idx']['కొత్త']], 1)
            model = TeluguWordCharCNN(6, sw, sc, len(meta['char_to_idx'])+2, 8, 16, 3)
            inputs = splits[0][:,:-1]
            changed = inputs.clone(); changed[:,-1] = 1
            torch.testing.assert_close(model.encode(inputs)[:,:,:-1], model.encode(changed)[:,:,:-1])
            loss = torch.nn.functional.cross_entropy(model(inputs), splits[0][:,-1]); loss.backward()
            for layer in [model.char_conv, model.fusion]:
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            torch.testing.assert_close(model.fused_embeddings(torch.zeros((1,3), dtype=torch.long)), torch.zeros(1,3,8))
            out = Path(directory)/'out'
            for epochs in [1,2]:
                subprocess.run([sys.executable, str(Path(__file__).with_name('train_telugu_comparison.py')),
                    '--wordchar-only','--data-dir',str(base),'--output-dir',str(out),
                    '--epochs',str(epochs),'--threads','1','--batch-size','2'], check=True, capture_output=True)
                cp = torch.load(out/'wordchar_cnn_latest.pt', weights_only=True)
                self.assertEqual(cp['epoch'], epochs)
                self.assertEqual(len(cp['history']), epochs)
                self.assertTrue(torch.isfinite(model_from_checkpoint(cp)(inputs)).all())


if __name__ == '__main__': unittest.main()
