"""Train and compare Telugu prefix-only LSTM/BiLSTM on the prepared corpus."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from model import TeluguLSTM, TeluguBiLSTM, TeluguCNN, model_from_checkpoint
from text_utils import project_root


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def save_model(path, data):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def load_split(folder):
    manifest = json.loads((folder / 'sequences_manifest.json').read_text())
    rows = np.empty((manifest['total_sequences'], manifest['seq_length'] + 1), dtype=np.int64)
    offset = 0
    for shard in manifest['shards']:
        with (folder / shard['path']).open('rb') as handle:
            samples = pickle.load(handle)
        count = len(samples)
        rows[offset:offset + count, :-1] = [x for x, _ in samples]
        rows[offset:offset + count, -1] = [y for _, y in samples]
        offset += count
    if offset != len(rows):
        raise ValueError('Sequence count mismatch')
    return torch.from_numpy(rows)


def evaluate(model, data, batch_size, unk_id=1):
    model.eval()
    loss = top1 = top3 = top5 = known1 = known5 = known_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in data.split(batch_size):
            logits = model(batch[:, :-1])
            targets = batch[:, -1]
            loss += torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            # OOV targets cannot be predicted as actual words. Never credit an
            # <unk> prediction as a correct word; report coverage separately.
            logits[:, :2] = -torch.inf
            predictions = logits.topk(min(5, logits.shape[1]), dim=1).indices
            matches = predictions.eq(targets[:, None])
            known = targets.ne(unk_id)
            top1 += matches[:, :1].any(1).sum().item()
            top3 += matches[:, :3].any(1).sum().item()
            top5 += matches.any(1).sum().item()
            known1 += (matches[:, 0] & known).sum().item()
            known5 += (matches.any(1) & known).sum().item()
            known_count += known.sum().item()
    average = loss / len(data)
    return dict(examples=len(data), cross_entropy=average,
                vocabulary_perplexity=math.exp(average), top1_accuracy=top1 / len(data),
                top3_accuracy=top3 / len(data), top5_accuracy=top5 / len(data),
                target_oov_rate=1 - known_count / len(data),
                known_target_top1=known1 / max(known_count, 1),
                known_target_top5=known5 / max(known_count, 1),
                evaluation_seconds=time.perf_counter() - started)


def audit_data(base):
    paths = {name: base / ('' if name == 'train' else name) / 'telugu_raw.txt'
             for name in ('train', 'validation', 'test')}
    sentences = {name: set(path.read_text(encoding='utf-8').splitlines())
                 for name, path in paths.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if sentences[a] & sentences[b]:
            raise ValueError(f'Exact sentence leakage between {a} and {b}')
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    hashes['vocab'] = hashlib.sha256((base / 'vocab.pkl').read_bytes()).hexdigest()
    for name in paths:
        folder = paths[name].parent
        manifest = json.loads((folder / 'sequences_manifest.json').read_text())
        digest = hashlib.sha256()
        for shard in manifest['shards']:
            digest.update((folder / shard['path']).read_bytes())
        hashes[name + '_sequences'] = digest.hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=project_root() / 'data_telugu_llm')
    parser.add_argument('--output-dir', type=Path, default=project_root() / 'outputs' / 'telugu_comparison')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--first-model', choices=['lstm', 'bilstm'], default='lstm')
    parser.add_argument('--cnn-only', action='store_true', help='Train CNN in a separate output directory')
    parser.add_argument('--wordchar-only', action='store_true', help='Train gated word-character CNN separately')
    args = parser.parse_args()
    if args.wordchar_only and (args.cnn_only or args.output_dir.resolve() in {
            (project_root() / 'outputs' / 'telugu_comparison').resolve(),
            (project_root() / 'outputs' / 'telugu_cnn').resolve()}):
        parser.error('Word-character CNN requires its own output directory and cannot use --cnn-only')
    if args.cnn_only and args.output_dir.resolve() == (project_root() / 'outputs' / 'telugu_comparison').resolve():
        parser.error('CNN requires a separate --output-dir to preserve recurrent experiments')
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error('epochs, batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    # Hold an OS lock for the life of this run; stale lock files are harmless.
    lock_handle = (output / '.training.lock').open('a+b')
    lock_handle.seek(0, 2)
    if lock_handle.tell() == 0:
        lock_handle.write(b'0')
        lock_handle.flush()
    lock_handle.seek(0)
    try:
        import os
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('Another training process owns this output directory.')
    state_path = output / 'status.json'
    try:
        save_json(state_path, dict(status='auditing_dataset'))
        hashes = audit_data(args.data_dir)
        with (args.data_dir / 'vocab.pkl').open('rb') as handle:
            vocab = pickle.load(handle)
        provenance = json.loads((args.data_dir / 'provenance.json').read_text())
        print('Loading full train/validation splits', flush=True)
        character_metadata = {}
        if args.wordchar_only:
            from wordchar_data import prepare
            from wordchar_model import TeluguWordCharCNN
            splits, surface_words, surface_chars, character_metadata = prepare(args.data_dir, vocab)
            train, validation, character_test = splits
        else:
            train = load_split(args.data_dir)
            validation = load_split(args.data_dir / 'validation')
        config = dict(embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.0)
        if args.cnn_only or args.wordchar_only:
            config['num_layers'] = 3
        experiment = dict(seed=args.seed, batch_size=args.batch_size, threads=args.threads,
                          train_examples=len(train), validation_examples=len(validation),
                          data_hashes=hashes, model_config=config, learning_rate=0.001,
                          device='cpu', torch_version=str(torch.__version__))
        if args.wordchar_only:
            experiment['character_encoding'] = dict(version=1, max_chars=32, char_embedding_dim=16,
                kernel_size=3, fusion='sigmoid_gate', alphabet='training_only')
        save_json(output / 'experiment.json', experiment)
        summary = {}
        model_order = [('lstm', TeluguLSTM), ('bilstm', TeluguBiLSTM)]
        if args.first_model == 'bilstm':
            model_order.reverse()
        if args.cnn_only:
            model_order = [('cnn', TeluguCNN)]
        if args.wordchar_only:
            model_order = [('wordchar_cnn', TeluguWordCharCNN)]
        for architecture, cls in model_order:
            torch.manual_seed(args.seed)
            if args.wordchar_only:
                model = cls(len(vocab['word_to_idx']), surface_words, surface_chars,
                            len(character_metadata['char_to_idx']) + 2, **config)
            else:
                model = cls(len(vocab['word_to_idx']), **config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            latest_path = output / f'{architecture}_latest.pt'
            best_path = output / f'{architecture}_best.pt'
            first_epoch = 1
            first_batch = 0
            best_loss = float('inf')
            history = []
            if latest_path.exists():
                checkpoint = torch.load(latest_path, weights_only=True, map_location='cpu')
                if checkpoint['experiment'] != experiment:
                    raise ValueError('Resume configuration/data differs; use a new output directory')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                first_epoch = checkpoint['epoch'] + (0 if checkpoint['partial_batch'] else 1)
                first_batch = checkpoint['partial_batch']
                best_loss = checkpoint['best_val_loss']
                history = checkpoint['history']
            def bundle(epoch, partial):
                return dict(model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(), architecture=architecture,
                            **config, word_to_idx=vocab['word_to_idx'], idx_to_word=vocab['idx_to_word'],
                            seq_length=vocab['seq_length'], epoch=epoch, partial_batch=partial,
                            best_val_loss=best_loss, history=history, experiment=experiment,
                            dataset_provenance=provenance, language='telugu',
                            character_metadata=character_metadata)
            for epoch in range(first_epoch, args.epochs + 1):
                model.train()
                generator = torch.Generator().manual_seed(args.seed + epoch)
                order = torch.randperm(len(train), generator=generator)
                started = time.perf_counter()
                loss_sum = count = 0
                batches = math.ceil(len(train) / args.batch_size)
                for batch_index, indices in enumerate(order.split(args.batch_size), 1):
                    if epoch == first_epoch and batch_index <= first_batch:
                        continue
                    batch = train[indices]
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(batch[:, :-1]), batch[:, -1])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    loss_sum += loss.item() * len(batch)
                    count += len(batch)
                    if batch_index % 100 == 0 or batch_index == batches:
                        state = dict(status='training', architecture=architecture, epoch=epoch,
                                     requested_epochs=args.epochs, batch=batch_index, batches=batches,
                                     train_loss=loss_sum / count, elapsed_seconds=time.perf_counter() - started)
                        save_json(state_path, state)
                        print(json.dumps(state), flush=True)
                    if batch_index % 500 == 0:
                        save_model(latest_path, bundle(epoch, batch_index))
                metrics = evaluate(model, validation, args.batch_size)
                record = dict(epoch=epoch, train_loss=loss_sum / max(count, 1),
                              training_loss_examples=count, validation=metrics,
                              training_seconds=time.perf_counter() - started)
                history.append(record)
                if metrics['cross_entropy'] < best_loss:
                    best_loss = metrics['cross_entropy']
                    save_model(best_path, bundle(epoch, 0))
                save_model(latest_path, bundle(epoch, 0))
                save_json(output / f'{architecture}_history.json', history)
                print(architecture, json.dumps(record), flush=True)
            save_json(state_path, dict(status='evaluating_test', architecture=architecture))
            checkpoint = torch.load(best_path, weights_only=True, map_location='cpu')
            best_model = model_from_checkpoint(checkpoint)
            test = character_test if args.wordchar_only else load_split(args.data_dir / 'test')
            metrics = evaluate(best_model, test, args.batch_size)
            metrics.update(architecture=architecture, best_epoch=checkpoint['epoch'],
                           completed_epochs=len(history), parameter_count=sum(p.numel() for p in model.parameters()),
                           checkpoint=str(best_path), training_examples_per_epoch=len(train))
            save_json(output / f'{architecture}_test_metrics.json', metrics)
            summary[architecture] = metrics
            del test
        save_json(output / 'comparison.json', summary)
        save_json(state_path, dict(status='complete', requested_epochs=args.epochs, results=summary))
        print('Requested models trained and evaluated.', flush=True)
    except BaseException as exc:
        save_json(state_path, dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()


