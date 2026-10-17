"""Train count models and select neural interpolation on validation only."""
import json
import os
import pickle
from collections import Counter, defaultdict
import torch
from model import model_from_checkpoint
from text_utils import project_root
from train_telugu_comparison import audit_data, load_split, save_json

ROOT = project_root()
OUT = ROOT / 'outputs/telugu_hybrid_v1'
WEIGHTS = [0.0, 0.1, 0.25, 0.5]

def count_distribution(row, counts, unigram):
    for size in (2, 1):
        key = tuple(row[-size:])
        c = counts[size].get(key)
        if c and sum(c.values()) >= 3:
            total = sum(c.values())
            return [(k, v / total) for k, v in c.items()]
    return unigram

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.training.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    torch.set_num_threads(4)
    def status(stage, **kw):
        value = dict(status=stage, pid=os.getpid(), **kw)
        save_json(OUT / 'status.json', value)
        print(json.dumps(value), flush=True)
    try:
        status('auditing_dataset')
        hashes = audit_data(ROOT / 'data_telugu_llm')
        path = ROOT / 'outputs/telugu_accuracy_v2/bilstm_best.pt'
        cp = torch.load(path, weights_only=True, map_location='cpu')
        if hashes != cp['experiment']['data_hashes']:
            raise ValueError('Dataset mismatch')
        model = model_from_checkpoint(cp).eval()
        status('training_counts')
        train = load_split(ROOT / 'data_telugu_llm')
        counts = {1: defaultdict(Counter), 2: defaultdict(Counter)}
        uni = Counter()
        for i, row in enumerate(train.numpy(), 1):
            target = int(row[-1]); uni[target] += 1
            for size in (1, 2):
                counts[size][tuple(map(int, row[-size-1:-1]))][target] += 1
            if i % 500000 == 0:
                status('training_counts', examples=i, total=len(train))
        unigram = [(k, v / len(train)) for k, v in uni.items()]
        del train
        counts = {size: {key: dict(c) for key, c in table.items() if sum(c.values()) >= 3} for size, table in counts.items()}
        artifact = dict(counts=counts, unigram=unigram, data_hashes=hashes, neural_checkpoint=str(path), weights=WEIGHTS, minimum_context_count=3)
        with (OUT / 'counts.pkl.tmp').open('wb') as f:
            pickle.dump(artifact, f)
        (OUT / 'counts.pkl.tmp').replace(OUT / 'counts.pkl')
        def score(split, weights):
            data = load_split(ROOT / 'data_telugu_llm' / split)
            hits = {a: [0, 0] for a in weights}
            with torch.inference_mode():
                for offset in range(0, len(data), 512):
                    batch = data[offset:offset+512]
                    neural = model(batch[:, :-1]).softmax(-1)
                    empirical = torch.zeros_like(neural)
                    for i, row in enumerate(batch[:, :-1].tolist()):
                        entries = count_distribution(row, counts, unigram)
                        indices, values = zip(*entries)
                        empirical[i, list(indices)] = torch.tensor(values)
                    for a in weights:
                        probs = (1-a)*neural + a*empirical
                        probs[:, :2] = -torch.inf
                        rank = probs.topk(5, dim=1).indices
                        matches = rank.eq(batch[:, -1, None])
                        hits[a][0] += matches[:, 0].sum().item()
                        hits[a][1] += matches.any(1).sum().item()
                    if offset % 25600 == 0:
                        status('evaluating_' + split, examples=offset, total=len(data))
            return {a: dict(examples=len(data), top1_accuracy=v[0]/len(data), top5_accuracy=v[1]/len(data)) for a,v in hits.items()}
        validation = score('validation', WEIGHTS)
        chosen = max(WEIGHTS, key=lambda a: validation[a]['top1_accuracy'])
        save_json(OUT / 'selection.json', dict(weight=chosen, validation=validation, selection='validation_top1', data_hashes=hashes))
        test = score('test', [chosen])[chosen]
        save_json(OUT / 'test_metrics.json', test)
        with (OUT / 'counts.pkl').open('rb') as f:
            reloaded = pickle.load(f)
        assert reloaded['data_hashes'] == hashes
        assert abs(sum(v for _,v in count_distribution([2,3], reloaded['counts'], reloaded['unigram']))-1) < 1e-5
        save_json(OUT / 'verification.json', dict(passed=True, count_reload=True, normalized_distribution=True))
        status('complete', selected_weight=chosen, metrics=test, verification_passed=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise

if __name__ == '__main__':
    main()
