# Calibrated Decisions

**Question:** TypeSafe trains its Jev decision model with "RLCD" — reinforcement learning for
calibrated decisions — and presents it as the way to get trustworthy confidence (recipe not
public). Does RL training actually give more trustworthy confidence than cheap fixes, on
**decision tasks the model has never seen**?

## Setup
- Model: Qwen3-1.7B, thinking off, run locally (Apple MPS).
- Decisions are single-letter answers; confidence = probability on the chosen option, read
  from one forward pass (how open Jev-style models work).
- Familiar tasks (train + calibrate): BoolQ, MNLI, Banking77 routing, MMLU.
  Unfamiliar tasks (test only): PubMedQA, ARC-Challenge, Climate-FEVER (+ JevBench, planned).

## Comparison
| method | |
|---|---|
| base model + temperature scaling | one number fit on familiar calibration data |
| fine-tuned (cross-entropy) + temperature scaling | ordinary supervised training |
| RL (GRPO, proper-scoring-rule reward) | the RLCD-style approach |

Side result: having the model *write* its confidence instead (untrained).

**Metrics:** accuracy, ECE, Brier, selective accuracy — reported separately for familiar and
unfamiliar tasks.

## Results so far (test set)
| | ECE familiar | ECE unfamiliar |
|---|---|---|
| base, raw | 0.274 | 0.363 |
| base, written confidence | 0.208 | 0.247 |
| base + temperature scaling (T = 9.89) | 0.064 | 0.111 |

**Related work:** TypeSafe Jev / RLCD (closed); RLCR (arXiv 2507.16806, RL calibration *with*
reasoning); open Jev alternatives (decider, s1-decision-model, lev, opendecider) use
option-probability readout + post-hoc calibration but don't isolate RL vs. cheap fixes on
unseen task types.
