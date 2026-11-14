"""Method 1, "read the spread": confidence = probability on the chosen option letter.

For each question, one forward pass; read the next-token probabilities of the option letters
(summing spelling variants like "A" and " A"), keep only the options that exist, renormalize.
Saves per-option log-probabilities so temperature scaling can be fit later without re-running.

Run: python eval_spread.py [model] [split ...]     e.g. python eval_spread.py Qwen/Qwen3-0.6B cal test
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import LETTERS, format_prompt

INSTRUCTION = '\nAnswer with only the letter.'
BATCH = 16
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def letter_token_ids(tok):
    """For each letter, every single-token spelling of it ('A', ' A')."""
    ids = []
    for L in LETTERS:
        variants = {tok.encode(v, add_special_tokens=False)[0] for v in (L, ' ' + L)}
        ids.append(sorted(variants))
    return ids


def build_chat(tok, r):
    msgs = [{'role': 'user', 'content': format_prompt(r) + INSTRUCTION}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def run(model_name, split):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = 'left'                      # so every prompt's last token lines up at position -1
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    letter_ids = letter_token_ids(tok)
    rows = [json.loads(l) for l in open(f'data/{split}.jsonl')]

    out, start = [], time.time()
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        enc = tok([build_chat(tok, r) for r in batch], return_tensors='pt', padding=True).to(device)
        probs = model(**enc).logits[:, -1].float().softmax(-1)
        for r, p in zip(batch, probs):
            k = len(r['options'])
            per_option = torch.stack([p[letter_ids[j]].sum() for j in range(k)])
            mass = per_option.sum().item()          # how much probability landed on valid letters at all
            logp = (per_option / per_option.sum()).log().tolist()
            out.append({'task': r['task'], 'answer': r['answer'], 'logp': logp, 'letter_mass': mass})
        if (i // BATCH) % 50 == 0:
            print(f'  {split}: {i + len(batch)}/{len(rows)}  ({time.time() - start:.0f}s)', flush=True)

    os.makedirs('results', exist_ok=True)
    path = f"results/spread_{model_name.split('/')[-1]}_{split}.jsonl"
    with open(path, 'w') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    print(f'wrote {path} ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen3-0.6B'
    for split in sys.argv[2:] or ['cal', 'test']:
        run(model_name, split)
