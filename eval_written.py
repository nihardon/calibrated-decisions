"""Method 2 evaluation: the model writes 'answer: X | confidence: p'. Greedy decoding.

Run: python eval_written.py [model or checkpoint dir] [split ...] [--limit N]
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from written import build_chat, score

BATCH = 16
MAX_NEW_TOKENS = 24
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def run(model_name, split, limit=None, tag=None):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    rows = [json.loads(l) for l in open(f'data/{split}.jsonl')][:limit]

    out, start = [], time.time()
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        enc = tok([build_chat(tok, r) for r in batch], return_tensors='pt', padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        for r, text in zip(batch, texts):
            conf, correct, ok = score(text, r)
            out.append({'task': r['task'], 'text': text.strip(), 'conf': conf, 'correct': correct, 'format_ok': ok})
        if (i // BATCH) % 50 == 0:
            print(f'  {split}: {i + len(batch)}/{len(rows)}  ({time.time() - start:.0f}s)', flush=True)

    os.makedirs('results', exist_ok=True)
    tag = tag or model_name.rstrip('/').split('/')[-1]
    path = f'results/written_{tag}_{split}.jsonl'
    with open(path, 'w') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    print(f'wrote {path} ({time.time() - start:.0f}s)')
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    if limit:
        args = [a for a in args if a != str(limit)]
    model_name = args[0] if args else 'Qwen/Qwen3-1.7B'
    for split in args[1:] or ['test']:
        run(model_name, split, limit)
