"""Build the decision-task splits.

Every example becomes: an instruction + input, a list of options (shuffled), and the index
of the correct option. Answers are always a single letter (A, B, C, ...).

Familiar tasks  -> train / cal / test   (cal = held out for temperature scaling + checkpoint picking)
Unfamiliar tasks -> test only            (never trained on, never used to fit anything)

Run: python data.py   -> writes data/{train,cal,test}.jsonl
"""
import json
import os
import random
from collections import Counter

from datasets import load_dataset

SEED = 0
CAPS = {'train': 2000, 'cal': 500, 'test': 1000}
MAX_CONTEXT_WORDS = 250
LETTERS = 'ABCDEFGH'

FAMILIAR = ['boolq', 'mnli', 'banking77', 'mmlu']
UNFAMILIAR = ['pubmedqa', 'arc', 'climate_fever']
# tasks whose options are a fixed label set (so label balance / majority baseline make sense)
FIXED_LABELS = ['boolq', 'mnli', 'pubmedqa', 'climate_fever']


def clip(text: str) -> str:
    words = text.split()
    return text if len(words) <= MAX_CONTEXT_WORDS else ' '.join(words[:MAX_CONTEXT_WORDS]) + ' ...'


def example(task, text, options, answer):
    return {'task': task, 'text': text, 'options': options, 'answer': answer}


# ---------- loaders: each returns a dict of {source_split: [examples]} ----------

def load_boolq():
    d = load_dataset('google/boolq')
    def conv(r):
        return example('boolq', f"Task: Answer the question using the passage.\nPassage: {clip(r['passage'])}\nQuestion: {r['question']}?",
                       ['yes', 'no'], 0 if r['answer'] else 1)
    return {'train': [conv(r) for r in d['train']], 'test': [conv(r) for r in d['validation']]}


def load_mnli():
    d = load_dataset('nyu-mll/glue', 'mnli')
    names = ['follows', 'neither', 'contradicts']   # GLUE labels: 0 entailment, 1 neutral, 2 contradiction
    def conv(r):
        return example('mnli', f"Task: Does the second sentence follow from the first, contradict it, or neither?\n"
                               f"Sentence 1: {r['premise']}\nSentence 2: {r['hypothesis']}", names, r['label'])
    return {'train': [conv(r) for r in d['train']], 'test': [conv(r) for r in d['validation_matched']]}


def load_banking77(rng):
    d = load_dataset('PolyAI/banking77')
    intents = [n.replace('_', ' ') for n in d['train'].features['label'].names]
    def conv(r):
        # route to 1 of 5 candidate intents: the correct one + 4 random distractors
        distractors = rng.sample([i for i in range(len(intents)) if i != r['label']], 4)
        options = [intents[r['label']]] + [intents[i] for i in distractors]
        return example('banking77', f"Task: Route this customer message to the correct intent.\nMessage: {r['text']}", options, 0)
    return {'train': [conv(r) for r in d['train']], 'test': [conv(r) for r in d['test']]}


def load_mmlu():
    # NOTE: not using MMLU's `auxiliary_train` — it contains ARC (one of our unfamiliar tasks).
    d = load_dataset('cais/mmlu', 'all')
    def conv(r):
        subject = r['subject'].replace('_', ' ')
        return example('mmlu', f"Task: Answer this {subject} question.\nQuestion: {r['question']}", list(r['choices']), r['answer'])
    rows = [conv(r) for r in d['test']]
    return {'train': rows, 'test': [conv(r) for r in d['validation']]}


def load_pubmedqa():
    d = load_dataset('qiaojin/PubMedQA', 'pqa_labeled')['train']
    names = ['yes', 'no', 'maybe']
    def conv(r):
        context = clip(' '.join(r['context']['contexts']))
        return example('pubmedqa', f"Task: Answer the medical research question using the abstract.\nAbstract: {context}\nQuestion: {r['question']}",
                       names, names.index(r['final_decision']))
    return {'test': [conv(r) for r in d]}


def load_arc():
    d = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
    def conv(r):
        return example('arc', f"Task: Answer this science question.\nQuestion: {r['question']}",
                       list(r['choices']['text']), r['choices']['label'].index(r['answerKey']))
    return {'test': [conv(r) for r in d['test']]}


def load_climate_fever():
    d = load_dataset('tdiggelm/climate_fever')['test']
    names = ['supports', 'refutes', 'not enough information']   # labels 0,1,2; 3 = disputed (dropped)
    out = []
    for r in d:
        if int(r['claim_label']) == 3:
            continue
        evidence = ' '.join(e['evidence'] for e in r['evidences'][:3])
        out.append(example('climate_fever', f"Task: Does the evidence support the claim, refute it, or is there not enough information?\n"
                                            f"Claim: {r['claim']}\nEvidence: {clip(evidence)}", names, int(r['claim_label'])))
    return {'test': out}


# ---------- splitting ----------

def balance(rows, rng):
    """Downsample so every correct-answer label is equally common (used for train only)."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r['options'][r['answer']], []).append(r)
    n = min(len(v) for v in by_label.values())
    out = [r for v in by_label.values() for r in rng.sample(v, n)]
    rng.shuffle(out)
    return out


def shuffle_options(r, rng):
    """Shuffle option order so the correct answer lands evenly on A, B, C, ... (kills position bias)."""
    order = list(range(len(r['options'])))
    rng.shuffle(order)
    return {**r, 'options': [r['options'][i] for i in order], 'answer': order.index(r['answer'])}


def build():
    rng = random.Random(SEED)
    loaders = {'boolq': load_boolq, 'mnli': load_mnli, 'banking77': lambda: load_banking77(rng),
               'mmlu': load_mmlu, 'pubmedqa': load_pubmedqa, 'arc': load_arc, 'climate_fever': load_climate_fever}
    splits = {'train': [], 'cal': [], 'test': []}

    for task, load in loaders.items():
        src = load()
        if task in FAMILIAR:
            pool = src['train'][:]
            rng.shuffle(pool)
            cal, rest = pool[:CAPS['cal']], pool[CAPS['cal']:]      # cal keeps the natural label mix
            splits['cal'] += cal
            if task in FIXED_LABELS:
                rest = balance(rest, rng)                          # train is label-balanced
            splits['train'] += rest[:CAPS['train']]
        test = src['test'][:]
        rng.shuffle(test)
        splits['test'] += test[:CAPS['test']]

    os.makedirs('data', exist_ok=True)
    for name, rows in splits.items():
        rows = [shuffle_options(r, rng) for r in rows]
        rng.shuffle(rows)
        with open(f'data/{name}.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        splits[name] = rows
    return splits


def format_prompt(r) -> str:
    """The task text + lettered options. The answer-format instruction is added per method later."""
    opts = '\n'.join(f'{LETTERS[i]}. {o}' for i, o in enumerate(r['options']))
    return f"{r['text']}\nOptions:\n{opts}"


if __name__ == '__main__':
    splits = build()
    for name, rows in splits.items():
        print(f'\n== {name}: {len(rows)} examples')
        for task, n in Counter(r['task'] for r in rows).items():
            task_rows = [r for r in rows if r['task'] == task]
            if task in FIXED_LABELS:
                labels = Counter(r['options'][r['answer']] for r in task_rows)
                baseline = f'majority-answer {max(labels.values()) / n:.0%}'
            else:
                baseline = f'random guess {sum(1 / len(r["options"]) for r in task_rows) / n:.0%}'
            letters = Counter(LETTERS[r['answer']] for r in task_rows)
            print(f'  {task:14s} n={n:5d} | baseline: {baseline:16s} | correct letter: '
                  + ' '.join(f'{k}:{v/n:.0%}' for k, v in sorted(letters.items())))
    print('\n== example prompt ==\n' + format_prompt(splits['train'][0]))
