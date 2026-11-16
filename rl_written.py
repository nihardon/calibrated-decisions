"""RL (GRPO) on WRITTEN confidence — the RLCD/RLCR-style setup where a Brier reward is proper.

Each step, for each question: sample GROUP completions like 'answer: B | confidence: 0.8'.
Reward per completion (parsed by written.score; format failure = wrong at confidence 1.0):
    brier   : correct - (confidence - correct)**2      (the stated number is separate from sampling,
                                                        so honest confidence maximizes expected reward)
    correct : correct
Advantage = (reward - group mean) / group std; loss = -mean(advantage * log p(completion)).
Checkpoint chosen by Brier of greedy outputs on a calibration subset (step 0 included).

Run: python rl_written.py --reward brier --steps 600 --out checkpoints/rlw_brier
"""
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import brier
from sft import MODEL, arg, device
from written import build_chat, score

SEED = 0
BATCH = 8            # questions per step
GROUP = 8            # completions per question
LR = 1e-5
MAX_NEW = 24
CHUNK = 8            # sequences per backward pass (memory)
EVAL_EVERY = 100
CAL_SUBSET = 300


@torch.no_grad()
def cal_brier(model, tok, cal):
    model.eval()
    conf, correct = [], []
    for i in range(0, len(cal), 32):
        rows = cal[i:i + 32]
        enc = tok([build_chat(tok, r) for r in rows], return_tensors='pt', padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, text in zip(rows, tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)):
            c, ok, _ = score(text, r)
            conf.append(c)
            correct.append(ok)
    model.train()
    if device == 'mps':
        torch.mps.empty_cache()
    return brier(conf, correct), sum(correct) / len(correct)


def completion_logprobs(model, seqs, attn, n_new, temp):
    """Sum of log p(token) over the generated part of each sequence (padding after EOS masked out).
    Computed at the sampling temperature, so the policy we train is the one we sampled from."""
    logits = model(input_ids=seqs, attention_mask=attn, logits_to_keep=n_new + 1).logits[:, :-1] / temp
    targets = seqs[:, -n_new:]
    logp = logits.float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (logp * attn[:, -n_new:]).sum(-1)


def main():
    reward_type, steps, out = arg('--reward', 'brier'), arg('--steps', 600), arg('--out', 'checkpoints/rlw')
    temp = arg('--temp', 1.3)
    assert reward_type in ('correct', 'brier')
    random.seed(SEED)
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()     # recompute activations in backward: far less memory
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type='CAUSAL_LM',
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    train = [json.loads(l) for l in open('data/train.jsonl')]
    cal = [json.loads(l) for l in open('data/cal.jsonl')]
    random.shuffle(cal)
    cal = cal[:CAL_SUBSET]

    longest = sorted(train, key=lambda r: len(build_chat(tok, r)))[-BATCH:] if '--longest' in __import__('sys').argv else None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best, acc = cal_brier(model, tok, cal)
    print(f'reward={reward_type} | step 0 | cal Brier {best:.3f} acc {acc:.3f}', flush=True)
    model.save_pretrained(out)          # step 0 counts as a candidate
    tok.save_pretrained(out)
    peak = 0.0
    start, log = time.time(), {'reward': 0.0, 'conf': 0.0, 'fmt_fail': 0.0, 'no_signal': 0.0}

    for step in range(1, steps + 1):
        rows = longest if longest else random.sample(train, BATCH)
        lr = LR * min(1, step / 30) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g['lr'] = lr

        # 1) sample GROUP completions per question (no gradients)
        model.eval()
        with torch.no_grad():
            enc = tok([build_chat(tok, r) for r in rows], return_tensors='pt', padding=True, pad_to_multiple_of=64).to(device)
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=temp, top_p=1.0,
                                 num_return_sequences=GROUP, pad_token_id=tok.pad_token_id)
        model.train()
        n_prompt = enc['input_ids'].shape[1]
        n_new = gen.shape[1] - n_prompt
        texts = tok.batch_decode(gen[:, n_prompt:], skip_special_tokens=True)
        prompt_mask = enc['attention_mask'].repeat_interleave(GROUP, 0)
        new_mask = (gen[:, n_prompt:] != tok.pad_token_id).long()
        new_mask = torch.cat([torch.ones_like(new_mask[:, :1]), new_mask[:, :-1]], 1)   # keep the first EOS/pad token
        attn = torch.cat([prompt_mask, new_mask], 1)

        # 2) rewards -> group-relative advantages
        adv = torch.zeros(len(texts), device=device)
        for q, r in enumerate(rows):
            rewards = []
            for k in range(GROUP):
                conf, correct, ok = score(texts[q * GROUP + k], r)
                rewards.append(correct - (conf - correct) ** 2 if reward_type == 'brier' else float(correct))
                log['conf'] += conf / (BATCH * GROUP)
                log['fmt_fail'] += (not ok) / (BATCH * GROUP)
            rw = torch.tensor(rewards, device=device)
            log['reward'] += rw.mean().item() / BATCH
            if rw.std() < 1e-6:
                log['no_signal'] += 1 / BATCH
            else:
                adv[q * GROUP:(q + 1) * GROUP] = (rw - rw.mean()) / rw.std()

        # 3) policy-gradient update, in chunks to bound memory
        opt.zero_grad()
        for c in range(0, len(texts), CHUNK):
            sl = slice(c, c + CHUNK)
            if adv[sl].abs().sum() == 0:
                continue
            lp = completion_logprobs(model, gen[sl], attn[sl], n_new, temp)
            (-(adv[sl] * lp).sum() / len(texts)).backward()
            if device == 'mps':
                peak = max(peak, torch.mps.driver_allocated_memory() / 1e9)
        opt.step()
        if device == 'mps':
            torch.mps.empty_cache()

        if step % 10 == 0:
            mem = torch.mps.driver_allocated_memory() / 1e9 if device == 'mps' else 0
            print(f"step {step} | reward {log['reward'] / 10:.3f} | avg conf {log['conf'] / 10:.3f} | fmt fail {log['fmt_fail'] / 10:.0%} | "
                  f"no-signal {log['no_signal'] / 10:.0%} | {(time.time() - start) / step:.1f}s/step | GPU mem {mem:.1f} GB (peak {peak:.1f})", flush=True)
            log = {k: 0.0 for k in log}
        if step % EVAL_EVERY == 0 or step == steps:
            b, acc = cal_brier(model, tok, cal)
            print(f'step {step} | cal Brier {b:.3f} acc {acc:.3f}' + ('  <- best, saved' if b < best else ''), flush=True)
            if b < best:
                best = b
                model.save_pretrained(out)
                tok.save_pretrained(out)
    model.save_pretrained(out + '_final')
    tok.save_pretrained(out + '_final')


if __name__ == '__main__':
    main()
