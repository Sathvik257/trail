"""Isolated, resumable Bi-LSTM fine-tuning selected by validation top-1."""
import json
import math
import os
import time
from pathlib import Path
import torch
from model import model_from_checkpoint
from text_utils import project_root
from train_telugu_comparison import audit_data, load_split, evaluate, save_json, save_model

ROOT = project_root()
OUT = ROOT / 'outputs' / 'telugu_accuracy_v2'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.training.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    torch.set_num_threads(4)
    state = OUT / 'status.json'
    try:
        save_json(state, dict(status='auditing_dataset', pid=os.getpid()))
        hashes = audit_data(ROOT / 'data_telugu_llm')
        latest = OUT / 'bilstm_latest.pt'
        source = latest if latest.exists() else ROOT / 'outputs/telugu_comparison/bilstm_best.pt'
        cp = torch.load(source, weights_only=True, map_location='cpu')
        if hashes != cp['experiment']['data_hashes']:
            raise ValueError('Dataset changed from original experiment')
        model = model_from_checkpoint(cp)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['lr'] = 0.0003
        run = cp.get('accuracy_v2', dict(best_accuracy=cp['history'][-1]['validation']['top1_accuracy'], stale=0, history=[]))
        if not (OUT / 'bilstm_best.pt').exists():
            save_model(OUT / 'bilstm_best.pt', cp)
        save_json(OUT / 'experiment.json', dict(parent_checkpoint=str(ROOT / 'outputs/telugu_comparison/bilstm_best.pt'), learning_rate=0.0003, maximum_total_epochs=10, patience=2, selection='validation_top1', batch_size=512, data_hashes=hashes))
        train = load_split(ROOT / 'data_telugu_llm')
        val = load_split(ROOT / 'data_telugu_llm/validation')
        first = cp['epoch'] + (0 if cp.get('partial_batch', 0) else 1)
        skip = cp.get('partial_batch', 0)
        def bundle(epoch, partial):
            return dict(cp, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), epoch=epoch, partial_batch=partial, accuracy_v2=run)
        for epoch in range(first, 11):
            if run['stale'] >= 2:
                break
            model.train()
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(42 + epoch))
            started = time.time()
            for i, indices in enumerate(order.split(512), 1):
                if epoch == first and i <= skip:
                    continue
                batch = train[indices]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(batch[:, :-1]), batch[:, -1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if i % 100 == 0:
                    progress = dict(status='training', pid=os.getpid(), epoch=epoch, batch=i, batches=math.ceil(len(train)/512), batch_loss=loss.item(), elapsed_seconds=time.time()-started)
                    save_json(state, progress)
                    print(json.dumps(progress), flush=True)
                if i % 500 == 0:
                    save_model(latest, bundle(epoch, i))
            save_json(state, dict(status='validating', epoch=epoch, pid=os.getpid()))
            metrics = evaluate(model, val, 512)
            run['history'].append(dict(epoch=epoch, validation=metrics))
            if metrics['top1_accuracy'] > run['best_accuracy']:
                run['best_accuracy'] = metrics['top1_accuracy']
                run['stale'] = 0
                save_model(OUT / 'bilstm_best.pt', bundle(epoch, 0))
            else:
                run['stale'] += 1
            save_model(latest, bundle(epoch, 0))
            save_json(OUT / 'history.json', run)
            print(json.dumps(run), flush=True)
        save_json(state, dict(status='evaluating_test', pid=os.getpid()))
        best = torch.load(OUT / 'bilstm_best.pt', weights_only=True, map_location='cpu')
        chosen = model_from_checkpoint(best).eval()
        test = load_split(ROOT / 'data_telugu_llm/test')
        result = evaluate(chosen, test, 512)
        result.update(best_epoch=best['epoch'], selection='validation_top1')
        save_json(OUT / 'bilstm_test_metrics.json', result)
        with torch.inference_mode():
            logits = chosen(test[:2, :-1])
        if logits.shape != (2, len(best['word_to_idx'])) or not torch.isfinite(logits).all():
            raise ValueError('Saved-model inference verification failed')
        save_json(OUT / 'verification.json', dict(passed=True, checkpoint_load=True, finite_logits=True))
        save_json(state, dict(status='complete', metrics=result, verification_passed=True))
    except BaseException as exc:
        save_json(state, dict(status='failed', error=repr(exc)))
        raise

if __name__ == '__main__':
    main()
