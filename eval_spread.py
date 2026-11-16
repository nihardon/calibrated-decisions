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

from spread import build_chat, letter_token_ids, option_logprobs

BATCH = 16
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(name):
    """A Hugging Face model name, or a LoRA checkpoint folder (adapter merged into its base model)."""
    if os.path.exists(os.path.join(name, 'adapter_config.json')):
        from peft import PeftModel
        base = json.load(open(os.path.join(name, 'adapter_config.json')))['base_model_name_or_path']
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        return PeftModel.from_pretrained(model, name).merge_and_unload()
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)


@torch.no_grad()
def run(model_name, split):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = 'left'                      # so every prompt's last token lines up at position -1
    model = load_model(model_name).to(device).eval()
    letter_ids = letter_token_ids(tok)
    rows = [json.loads(l) for l in open(f'data/{split}.jsonl')]

    out, start = [], time.time()
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        enc = tok([build_chat(tok, r) for r in batch], return_tensors='pt', padding=True).to(device)
        last = model(**enc, logits_to_keep=1).logits[:, -1]
        for r, logits in zip(batch, last):
            k = len(r['options'])
            probs = logits.float().softmax(-1)
            mass = sum(probs[letter_ids[j]].sum().item() for j in range(k))   # probability on valid letters at all
            logp = option_logprobs(logits, letter_ids, k).tolist()
            out.append({'task': r['task'], 'answer': r['answer'], 'logp': logp, 'letter_mass': mass})
        if (i // BATCH) % 50 == 0:
            print(f'  {split}: {i + len(batch)}/{len(rows)}  ({time.time() - start:.0f}s)', flush=True)

    os.makedirs('results', exist_ok=True)
    path = f"results/spread_{model_name.rstrip('/').split('/')[-1]}_{split}.jsonl"
    with open(path, 'w') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    print(f'wrote {path} ({time.time() - start:.0f}s)')


if __name__ == '__main__':
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'Qwen/Qwen3-0.6B'
    for split in sys.argv[2:] or ['cal', 'test']:
        run(model_name, split)
