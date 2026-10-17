"""Post-hoc held-out diagnostic by training target frequency; never fits models."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
import torch
from model import model_from_checkpoint
from text_utils import iter_language_lines,tokenize
from train_telugu_comparison import load_split,save_json

def main():
    root=Path(__file__).resolve().parents[1]; base=root/'data_telugu_llm'
    torch.set_num_threads(4)
    counts=Counter()
    for line in iter_language_lines(base/'telugu_raw.txt'):
        counts.update(tokenize(line)[1:])
    baseline=load_split(base/'test')
    target_words=[]; prefixes=[]
    for line in iter_language_lines(base/'test'/'telugu_raw.txt'):
        words=tokenize(line)
        for i in range(1,len(words)):
            target_words.append(words[i]); prefixes.append(words[max(0,i-8):i])
    assert len(target_words)==len(baseline)
    results={}
    for folder,key in [('telugu_cnn','cnn'),('telugu_wordchar','wordchar_cnn')]:
        path=root/'outputs'/folder/f'{key}_best.pt'
        cp=torch.load(path,map_location='cpu',weights_only=True)
        vocab=cp['word_to_idx']; model=model_from_checkpoint(cp).eval()
        assert torch.equal(torch.tensor([vocab.get(w,1) for w in target_words]),baseline[:,-1])
        data=baseline.clone()
        if key=='wordchar_cnn':
            surfaces=cp['character_metadata']['surface_to_idx']
            data[:,:-1]=torch.tensor([[0]*(8-len(ws))+[surfaces[w] for w in ws] for ws in prefixes])
            assert torch.equal(model.surface_words[data[:,:-1]],baseline[:,:-1])
        groups=['oov' if w not in vocab else ('rare_1_10' if counts[w]<=10 else 'medium_11_100' if counts[w]<=100 else 'frequent_101_plus') for w in target_words]
        totals={k:dict(examples=0,top1_hits=0,top5_hits=0) for k in ['all','oov','rare_1_10','medium_11_100','frequent_101_plus','unknown_prefix','known_prefix']}
        with torch.inference_mode():
            for start in range(0,len(data),512):
                batch=data[start:start+512]; logits=model(batch[:,:-1]); logits[:,:2]=-torch.inf
                hits=logits.topk(5,dim=1).indices.eq(batch[:,-1,None])
                unknown=baseline[start:start+len(batch),:-1].eq(1).any(1)
                for j in range(len(batch)):
                    for group in ['all',groups[start+j],'unknown_prefix' if unknown[j] else 'known_prefix']:
                        r=totals[group]; r['examples']+=1; r['top1_hits']+=int(hits[j,0]); r['top5_hits']+=int(hits[j].any())
        for r in totals.values():
            for k in ['top1','top5']: r[k+'_accuracy']=r[k+'_hits']/r['examples'] if r['examples'] else None
        old=json.loads((path.parent/f'{key}_test_metrics.json').read_text())
        for k in ['top1_accuracy','top5_accuracy']: assert abs(totals['all'][k]-old[k])<1e-9
        results[key]=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=totals)
        print(key, json.dumps(totals),flush=True)
    save_json(root/'outputs'/'telugu_frequency_evaluation.json',dict(protocol='Post-hoc diagnostic, seed 42. Frequency counts training next-word targets. Rare: known vocabulary targets with <=10 training occurrences. OOV is separate. Unknown prefix means at least one input word maps to UNK. No tuning on these results.',models=results))
if __name__=='__main__': main()
