"""Post-hoc context coverage and CPU microbenchmark; no tuning or weight updates."""
import gc, hashlib, json, os, pickle, platform, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from model import model_from_checkpoint
from train_language_comparison import load_split
from train_telugu_hybrid_long import long_distribution

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/checks/paper_inference'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
torch.set_num_threads(2)
report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'torch':torch.__version__,'platform':platform.platform(),'processor':platform.processor(),
        'logical_cpu_count':os.cpu_count(),'threads':2,'seed':20261120,
        'protocol':'Full test-prefix support coverage. Latency: 100 seeded test prefixes, five warmups, three rounds, batch one, model forward + softmax + top5, tokenization and model loading excluded. No parameters selected using these diagnostics. Other processes may contend for CPU.', 'languages':{}}
for lang in ('telugu','hindi','english'):
    data=ROOT/('data_telugu_llm' if lang=='telugu' else f'data_{lang}_hf')
    cp_path=ROOT/'outputs'/('telugu_accuracy_v2' if lang=='telugu' else f'{lang}_hf_bilstm_continuation')/'bilstm_best.pt'
    count_path=ROOT/'outputs'/('telugu_hybrid_v2' if lang=='telugu' else f'{lang}_hf_hybrid_v1')/'counts.pkl'
    cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    with count_path.open('rb') as f:artifact=pickle.load(f)
    assert artifact['data_hashes']==cp['experiment']['data_hashes']
    counts,uni=artifact['counts'],artifact['unigram']
    rows=load_split(data/'test')
    coverage=Counter()
    for row in rows.tolist():
        order=next((n for n in (4,3,2,1) if tuple(row[-n-1:-1]) in counts[n]),0)
        coverage[order]+=1
    model=model_from_checkpoint(cp).eval()
    indexes=np.random.default_rng(20261120).choice(len(rows),100,replace=False)
    inputs=[rows[int(i),:-1].unsqueeze(0) for i in indexes]
    elapsed={'neural':[],'hybrid':[]}
    alpha=.35 if lang=='hindi' else .5
    with torch.inference_mode():
        for x in inputs[:5]:model(x).softmax(-1).topk(5)
        for repeat in range(3):
            for x in inputs:
                # Alternate ordering to reduce systematic timing-order effects.
                for mode in (('neural','hybrid') if repeat%2==0 else ('hybrid','neural')):
                    start=time.perf_counter_ns()
                    probs=model(x)[0].softmax(-1)
                    if mode=='hybrid':
                        empirical=torch.zeros_like(probs)
                        ids,values=zip(*long_distribution(x[0].tolist(),counts,uni))
                        empirical[list(ids)]=torch.tensor(values,dtype=probs.dtype)
                        probs=(1-alpha)*probs+alpha*empirical
                    probs[:2]=-torch.inf
                    probs.topk(5)
                    elapsed[mode].append((time.perf_counter_ns()-start)/1e6)
    result={'examples':len(rows),'highest_retained_context_order':dict(sorted(coverage.items())),
            'target_oov_count':int((rows[:,-1]==1).sum()),'parameter_count':sum(p.numel() for p in model.parameters()),
            'checkpoint_bytes':cp_path.stat().st_size,'counts_bytes':count_path.stat().st_size,
            'checkpoint_sha256':sha(cp_path),'counts_sha256':sha(count_path),'count_weight':alpha,
            'latency_ms':{mode:{'n':len(vals),'median':float(np.median(vals)),'p95':float(np.percentile(vals,95))} for mode,vals in elapsed.items()}}
    report['languages'][lang]=result
    (OUT/'inference_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(lang,json.dumps(result),flush=True)
    del model,cp,artifact,counts,uni,rows,inputs
    gc.collect()
