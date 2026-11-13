# Calibrated Decisions

**Question:** for fast, no-reasoning LLM decision models (typed answers: yes/no, choose-one,
score — no chain of thought), is RL with a proper-scoring-rule reward needed for trustworthy
confidence, or does post-hoc temperature scaling do as well? Does the answer change on
held-out (out-of-distribution) decision tasks?

## Comparison
Two ways to get confidence from a decision model:

| confidence source | method | trained by |
|---|---|---|
| token probability (implicit) | prompting | — |
| token probability (implicit) | + temperature scaling | one scalar fit on a validation split |
| token probability (implicit) | SFT | cross-entropy (already a proper scoring rule) |
| written-out number (explicit, Jev-style typed output) | prompting | — |
| written-out number (explicit) | RL, correctness-only reward (control) | GRPO |
| written-out number (explicit) | RL, correctness + Brier reward | GRPO |

RL is only needed for the explicit case: a sampled number isn't differentiable, so it can
only be scored after it's written. The core question is whether RL-trained *written*
confidence beats temperature-scaled *token-probability* confidence, especially on held-out tasks.

**Metrics:** accuracy, expected calibration error (ECE), Brier score, selective accuracy at
fixed coverage (abstain when unsure). In-distribution and held-out tasks.

**Related work:** RLCR — *Beyond Binary Rewards: Training LMs to Reason About Their
Uncertainty* (arXiv 2507.16806) does calibration-reward RL **with** reasoning; this project
targets the no-reasoning / single-step setting where temperature scaling is a strong baseline.

_Status: scaffolded 2026-11-08._
