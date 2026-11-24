"""Pinned Hugging Face preparation and resumable Hindi/English LSTM/Bi-LSTM training."""
import hashlib, json, random, subprocess, sys, urllib.request
from pathlib import Path
from collections import Counter
from text_utils import project_root, clean_text, split_sentences
from preprocess import preprocess_corpus, write_sequence_shards
from sequence_data import load_pickle
from prepare_telugu_llm_data import split_for_instruction
ROOT=project_root()
SOURCES={
 'hindi': ('iamshnoo/alpaca-cleaned-hindi','60be13ada2d3a039eb1dd8dce3d5c61eb1445d82','data/train-00000-of-00001-9c3cdcdad5ba72d6.parquet','Unspecified in source card; verify upstream translation rights before redistribution'),
 'english': ('yahma/alpaca-cleaned','12567cabf869d7c92e573c7c783905fc160e9639','alpaca_data_cleaned.json','cc-by-4.0')}
SOURCE_SHA256 = {
 'hindi': '8a2cec5d1dfcbe6a7a8e8fc19e386919b632d71b906177b825d98e106db96d85',
 'english': 'bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b',
}

def prepare(language):
 base=ROOT/('data_'+language+'_hf')
 required = [base/'vocab.pkl', base/'sequences_manifest.json', base/'validation/sequences_manifest.json', base/'test/sequences_manifest.json']
 if (base/'provenance.json').exists() and all(p.is_file() for p in required): return base
 base.mkdir(exist_ok=True)
 repo,revision,filename,license=SOURCES[language]
 source=base/Path(filename).name
 if not source.exists():
  temp=source.with_suffix('.download')
  urllib.request.urlretrieve(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}',temp)
  temp.replace(source)
 if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256[language]:
  raise ValueError(f'Source checksum mismatch: {source}')
 if source.suffix=='.parquet':
  import pandas as pd
  rows=pd.read_parquet(source).to_dict('records')
 else: rows=json.loads(source.read_text(encoding='utf-8'))
 groups={k:set() for k in ('train','validation','test')}; rejected=0
 for row in rows:
  instruction=clean_text(row['instruction'] or '',language)
  if not instruction: continue
  group=split_for_instruction(instruction)
  for field in ('instruction','input','output'):
   for sentence in split_sentences(row[field] or '',language):
    words=sentence.split()
    if len(words)<2: continue
    grams=Counter(tuple(words[i:i+4]) for i in range(len(words)-3))
    if grams and max(grams.values())>=4:
     rejected+=1;continue
    groups[group].add(sentence)
 groups['train']-=groups['validation']|groups['test'];groups['validation']-=groups['test']
 for name,sentences in groups.items():
  assert sentences
  folder=base if name=='train' else base/name;folder.mkdir(exist_ok=True)
  lines=sorted(sentences);random.Random(42).shuffle(lines)
  (folder/(language+'_raw.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
 n,q,v=preprocess_corpus(base/(language+'_raw.txt'),base/'sequences.pkl',base/'vocab.pkl',base/'ngram_model.pkl',max_vocab=20000,min_freq=2,seq_length=8,shard_size=50000,max_legacy_sequences=0,max_ngram_lines=15000,language=language)
 splits={'train':dict(sentences=n,sequences=q)};vocab=load_pickle(base/'vocab.pkl')
 for name in ('validation','test'):
  folder=base/name
  n,q,_=write_sequence_shards(folder/(language+'_raw.txt'),vocab['word_to_idx'],8,folder/'sequences_manifest.json',folder/'sequence_shards',50000,folder/'sequences.pkl',0,language=language)
  splits[name]=dict(sentences=n,sequences=q)
 provenance=dict(language=language,dataset=repo,revision=revision,source_file=filename,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),license=license,source_records=len(rows),splits=splits,vocab_size=v,split_method='SHA256 normalized instruction seed42 90/5/5; exact sentence overlap removed',vocab_fit='train only',repetition_filter='reject sentences with a four-token phrase repeated at least four times',rejected_sentences=rejected)
 (base/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
 print(json.dumps(provenance),flush=True)
 return base

def main():
 import msvcrt
 out=ROOT/'outputs/multilingual_hf';out.mkdir(exist_ok=True)
 lock=(out/'.worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:
  # Prepare both sources before long neural training begins.
  for language in SOURCES: prepare(language)
  for language in SOURCES:
   subprocess.run([sys.executable,'-u',str(ROOT/'src/train_language_comparison.py'),'--data-dir',str(ROOT/('data_'+language+'_hf')),'--output-dir',str(ROOT/'outputs'/(language+'_hf_comparison')),'--epochs','5'],check=True,cwd=ROOT)
  (out/'status.json').write_text(json.dumps({'status':'complete'}))
 except BaseException as e:
  (out/'status.json').write_text(json.dumps({'status':'failed','error':repr(e)}));raise
if __name__=='__main__':main()
