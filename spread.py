"""The option-spread readout, shared by evaluation, fine-tuning, and RL.

Given the next-token logits after the prompt, return log-probabilities over the K options:
sum each letter's spellings ('A', ' A'), keep only this question's letters, renormalize.
"""
import torch

from data import LETTERS, format_prompt

INSTRUCTION = '\nAnswer with only the letter.'


def letter_token_ids(tok):
    """For each letter, every single-token spelling of it ('A', ' A')."""
    return [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, ' ' + L)}) for L in LETTERS]


def build_chat(tok, r):
    msgs = [{'role': 'user', 'content': format_prompt(r) + INSTRUCTION}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def option_logprobs(last_logits, letter_ids, n_options):
    """last_logits: (vocab,) logits at the answer position -> (n_options,) log-probs over the options."""
    logp_vocab = last_logits.float().log_softmax(-1)
    per_option = torch.stack([torch.logsumexp(logp_vocab[letter_ids[j]], 0) for j in range(n_options)])
    return per_option - torch.logsumexp(per_option, 0)
