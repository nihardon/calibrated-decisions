"""Supervised fine-tuning (LoRA) on the familiar train split.

Loss = cross-entropy on the option spread: -log p(correct option), using exactly the readout
we evaluate with. Checkpoint chosen by calibration-split NLL (never test).

Run: python sft.py [--steps N] [--out checkpoints/sft]
"""
import json
import math
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from spread import build_chat, letter_token_ids, option_logprobs

MODEL = 'Qwen/Qwen3-1.7B'
SEED = 0
BATCH = 8
LR = 1e-4
EVAL_EVERY = 250
CAL_SUBSET = 400
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def batch_loss(model, tok, letter_ids, rows, tau=1.0):
    # pad to a multiple of 64 so the GPU sees few distinct shapes (MPS caches memory per shape)
    enc = tok([build_chat(tok, r) for r in rows], return_tensors='pt', padding=True, pad_to_multiple_of=64).to(device)
    last = model(**enc, logits_to_keep=1).logits[:, -1]   # only score the answer position (saves GBs)
    losses = [-option_logprobs(last[i] / tau, letter_ids, len(r['options']))[r['answer']] for i, r in enumerate(rows)]
    return torch.stack(losses).mean()


@torch.no_grad()
def cal_nll(model, tok, letter_ids, cal, tau=1.0):
    model.eval()
    total = sum(batch_loss(model, tok, letter_ids, cal[i:i + 16], tau).item() * len(cal[i:i + 16]) for i in range(0, len(cal), 16))
    model.train()
    if device == 'mps':
        torch.mps.empty_cache()
    return total / len(cal)


def main():
    steps, out = arg('--steps', 1000), arg('--out', 'checkpoints/sft')
    random.seed(SEED)
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type='CAUSAL_LM',
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {trainable/1e6:.1f}M of {sum(p.numel() for p in model.parameters())/1e6:.0f}M')

    letter_ids = letter_token_ids(tok)
    train = [json.loads(l) for l in open('data/train.jsonl')]
    cal = [json.loads(l) for l in open('data/cal.jsonl')]
    random.shuffle(cal)
    cal = cal[:CAL_SUBSET]          # a fixed subset of cal for checkpoint picking (speed)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best = cal_nll(model, tok, letter_ids, cal)
    print(f'step 0 | cal NLL {best:.3f}')
    start = time.time()
    for step in range(1, steps + 1):
        rows = random.sample(train, BATCH)
        lr = LR * min(1, step / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))   # warmup + cosine
        for g in opt.param_groups:
            g['lr'] = lr
        loss = batch_loss(model, tok, letter_ids, rows)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            if device == 'mps':
                torch.mps.empty_cache()
            mem = torch.mps.driver_allocated_memory() / 1e9 if device == 'mps' else 0
            print(f'step {step} | train loss {loss.item():.3f} | {(time.time() - start) / step:.2f}s/step | GPU mem {mem:.1f} GB', flush=True)
        if step % EVAL_EVERY == 0 or step == steps:
            nll = cal_nll(model, tok, letter_ids, cal)
            print(f'step {step} | cal NLL {nll:.3f}' + ('  <- best, saved' if nll < best else ''), flush=True)
            if nll < best:
                best = nll
                model.save_pretrained(out)
                tok.save_pretrained(out)


if __name__ == '__main__':
    main()
