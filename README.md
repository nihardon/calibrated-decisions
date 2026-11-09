# Calibrated Decisions

**Question:** for fast, no-reasoning LLM decision models (typed answers: yes/no, choose-one,
score — no chain of thought), is RL with a proper-scoring-rule reward needed for trustworthy
confidence, or does post-hoc temperature scaling do as well? Does the answer change on
held-out (out-of-distribution) decision tasks?

## Comparison
| method | what it is |
|---|---|
| prompting | base instruct model, confidence read from answer-token probabilities |
| + temperature scaling | one scalar fit on a validation split (classic post-hoc calibration) |
| SFT | fine-tuned on the decision tasks with cross-entropy |
| RL (correctness only) | GRPO, reward = 1 if correct (control) |
| RL (calibration reward) | GRPO, reward = correctness + Brier-style proper scoring rule |

**Metrics:** accuracy, expected calibration error (ECE), Brier score, selective accuracy at
fixed coverage (abstain when unsure). In-distribution and held-out tasks.

**Related work:** RLCR — *Beyond Binary Rewards: Training LMs to Reason About Their
Uncertainty* (arXiv 2507.16806) does calibration-reward RL **with** reasoning; this project
targets the no-reasoning / single-step setting where temperature scaling is a strong baseline.

_Status: scaffolded 2026-11-08._
