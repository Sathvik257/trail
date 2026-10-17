"""Check causal isolation, gradients and checkpoint round-trip for CNN."""
import unittest
import torch
from model import TeluguCNN, model_from_checkpoint


class CNNChecks(unittest.TestCase):
    def test_causality_and_checkpoint(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = TeluguCNN(12, 8, 16, 3, 0)
        x = torch.tensor([[0, 0, 2, 3, 4, 5, 6, 7]])
        changed = x.clone()
        changed[:, 5:] = 10
        torch.testing.assert_close(model.encode(x)[:, :, :5], model.encode(changed)[:, :, :5])
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([8]))
        loss.backward()
        for layer in model.convs:
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        cp = dict(architecture='cnn', word_to_idx={str(i): i for i in range(12)},
                  embedding_dim=8, hidden_dim=16, num_layers=3, dropout=0,
                  model_state_dict=model.state_dict())
        torch.testing.assert_close(model(x), model_from_checkpoint(cp)(x))


if __name__ == '__main__':
    unittest.main()
