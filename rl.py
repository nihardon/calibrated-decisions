"""RL (GRPO) on the option spread, LoRA, starting from the base model.

Each step, for each question: one forward pass -> option spread -> sample GROUP answers from it.
Reward per sampled answer:
    correct : 1 if right else 0
    brier   : (1 if right else 0) - (p(sampled answer) - (1 if right else 0))**2     <- RLCD-style
Advantage = (reward - group mean) / group std. Loss = -mean(advantage * log p(sampled answer)).
The policy is the spread at a fixed temperature TAU (scores / TAU): the base model is ~97% sure of
everything, so at TAU=1 nearly every group samples one letter -> zero advantage -> no learning.
TAU ~ the base model's fitted temperature-scaling T. Every method gets post-hoc T afterwards anyway.
Checkpoint chosen by calibration-split NLL of the policy, same rule as sft.py.

Run: python rl.py --reward brier --steps 1000 --out checkpoints/rl_brier
"""
import json
import math
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft import CAL_SUBSET, EVAL_EVERY, MODEL, arg, cal_nll, device
from spread import build_chat, letter_token_ids, option_logprobs

SEED = 0
BATCH = 8          # questions per step
GROUP = 8          # sampled answers per question
LR = 2e-5
TAU = 10.0         # policy temperature (exploration)


def main():
    reward_type, steps, out = arg('--reward', 'brier'), arg('--steps', 1000), arg('--out', 'checkpoints/rl')
    tau = arg('--tau', TAU)
    assert reward_type in ('correct', 'brier')
    random.seed(SEED)
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type='CAUSAL_LM',
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    letter_ids = letter_token_ids(tok)
    train = [json.loads(l) for l in open('data/train.jsonl')]
    cal = [json.loads(l) for l in open('data/cal.jsonl')]
    random.shuffle(cal)
    cal = cal[:CAL_SUBSET]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best = cal_nll(model, tok, letter_ids, cal, tau)
    print(f'reward={reward_type} tau={tau} | step 0 | cal NLL {best:.3f}', flush=True)
    start = time.time()
    log = {'reward': 0.0, 'conf': 0.0, 'no_signal': 0.0}

    for step in range(1, steps + 1):
        rows = random.sample(train, BATCH)
        lr = LR * min(1, step / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g['lr'] = lr

        enc = tok([build_chat(tok, r) for r in rows], return_tensors='pt', padding=True, pad_to_multiple_of=64).to(device)
        last = model(**enc, logits_to_keep=1).logits[:, -1]

        losses = []
        for i, r in enumerate(rows):
            logp = option_logprobs(last[i] / tau, letter_ids, len(r['options']))
            probs = logp.detach().exp()
            samples = torch.multinomial(probs, GROUP, replacement=True)        # roll the weighted die GROUP times
            correct = (samples == r['answer']).float()
            reward = correct
            if reward_type == 'brier':
                reward = correct - (probs[samples] - correct) ** 2             # punish confident mistakes hardest
            if reward.std() < 1e-6:
                log['no_signal'] += 1 / BATCH                                   # all samples scored the same: nothing to learn
                adv = torch.zeros_like(reward)
            else:
                adv = (reward - reward.mean()) / reward.std()
            losses.append(-(adv * logp[samples]).mean())
            log['reward'] += reward.mean().item() / BATCH
            log['conf'] += probs.max().item() / BATCH

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 25 == 0:
            if device == 'mps':
                torch.mps.empty_cache()
            mem = torch.mps.driver_allocated_memory() / 1e9 if device == 'mps' else 0
            print(f"step {step} | reward {log['reward'] / 25:.3f} | top conf {log['conf'] / 25:.3f} | "
                  f"no-signal {log['no_signal'] / 25:.0%} | {(time.time() - start) / step:.2f}s/step | GPU mem {mem:.1f} GB", flush=True)
            log = {k: 0.0 for k in log}
        if step % EVAL_EVERY == 0 or step == steps:
            nll = cal_nll(model, tok, letter_ids, cal, tau)
            print(f'step {step} | cal NLL {nll:.3f}' + ('  <- best, saved' if nll < best else ''), flush=True)
            if nll < best:
                best = nll
                model.save_pretrained(out)
                tok.save_pretrained(out)


if __name__ == '__main__':
    main()
