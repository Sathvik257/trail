import json
from pathlib import Path
p=Path('outputs/telugu_frequency_evaluation.json')
r=json.loads(p.read_text())['models']
s='# Telugu frequency and unknown-input evaluation\n\nPost-hoc diagnostic on the existing held-out test set, using the unchanged validation-selected checkpoints. Frequencies count target positions in training text only. No models were fitted or tuned.\n\n| Group | Test examples | Word CNN top-1 | Character CNN top-1 | Word CNN top-5 | Character CNN top-5 |\n|---|---:|---:|---:|---:|---:|\n'
for k in r['cnn']['groups']:
 a=r['cnn']['groups'][k];b=r['wordchar_cnn']['groups'][k]
 s+=f"| {k} | {a['examples']:,} | {a['top1_accuracy']:.2%} | {b['top1_accuracy']:.2%} | {a['top5_accuracy']:.2%} | {b['top5_accuracy']:.2%} |\n"
s+='\nRare means a known-vocabulary target with at most 10 training target occurrences. Medium means 11–100, frequent means over 100. OOV targets form a separate group and count as misses. Unknown-prefix examples contain at least one input word outside the word vocabulary. Prefix groups overlap frequency groups and must not be added together.\n\nThe rare group contains only 125 examples, so it cannot support broad claims about Telugu morphology or rare-word performance. Unknown-input performance does not imply the model can generate unseen target words. These are descriptive single-seed results, not significance tests or gate ablations. Overall top-1 and top-5 accuracies reproduce the saved test metrics exactly.\n'
Path('TELUGU_FREQUENCY_RESULTS.md').write_text(s,encoding='utf-8')
