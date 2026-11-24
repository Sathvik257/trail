"""Predict with the same language-specific neural checkpoints as the web app."""
import argparse
import json
from pathlib import Path
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import configure_utf8_stdio, project_root


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['telugu', 'hindi', 'english'], default='telugu')
    parser.add_argument('--model', choices=['lstm', 'bilstm'], default='lstm')
    parser.add_argument('--checkpoint', type=Path, help='Override the baseline checkpoint')
    parser.add_argument('--text', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    folder = 'telugu_comparison' if args.language == 'telugu' else f'{args.language}_hf_comparison'
    path = args.checkpoint or project_root() / 'outputs' / folder / f'{args.model}_best.pt'
    if not path.is_file():
        parser.error(f'Checkpoint not found: {path}')
    predictor = TeluguNeuralPredictor(path)
    if predictor.checkpoint.get('language', 'telugu') != args.language:
        parser.error('Checkpoint language does not match --language')
    print(json.dumps(predictor.predict(args.text, args.top_k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
