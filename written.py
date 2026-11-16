"""Method 2's output format: the model writes its answer AND its confidence.

    answer: B | confidence: 0.85

Shared by evaluation and the RL reward, so both agree on what counts as a valid answer.
"""
import re

from data import LETTERS, format_prompt

INSTRUCTION = ('\nRespond in exactly this format: answer: <letter> | confidence: <number between 0 and 1>\n'
               'The confidence is the probability that your answer is correct.')

ANSWER_RE = re.compile(r'answer\s*[:=]?\s*\(?([a-h])\)?\b', re.IGNORECASE)
CONF_RE = re.compile(r'confidence\s*[:=]?\s*(\d*\.?\d+)\s*(%?)', re.IGNORECASE)


def build_chat(tok, r):
    msgs = [{'role': 'user', 'content': format_prompt(r) + INSTRUCTION}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse(text: str, n_options: int):
    """Return (answer_index, confidence) or None if the output is a format failure.

    Lenient on surface (case, spaces, '72%'); strict on content (valid letter, 0 <= conf <= 1).
    """
    a, c = ANSWER_RE.search(text), CONF_RE.search(text)
    if not a or not c:
        return None
    idx = LETTERS.index(a.group(1).upper())
    if idx >= n_options:
        return None                          # e.g. 'E' on a 4-option question
    conf = float(c.group(1))
    if c.group(2) == '%':
        conf /= 100
    if not 0 <= conf <= 1:
        return None
    return idx, conf


def score(text: str, r):
    """(confidence, correct, format_ok). Format failures count as wrong at confidence 1.0 (max Brier penalty)."""
    parsed = parse(text, len(r['options']))
    if parsed is None:
        return 1.0, 0, False
    idx, conf = parsed
    return conf, int(idx == r['answer']), True
