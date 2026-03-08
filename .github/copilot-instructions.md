# Copilot Instructions

## Big picture
- This repo is a modular RLHF pipeline: SFT -> reward/cost value models -> PPO or PPO-Lagrangian (Safe RLHF).
- Core code lives under safe_rlhf/: finetune/ (SFT), values/ (reward + cost), algorithms/ (PPO, PPO-LAG), trainers/ (shared training loops), datasets/ (raw + tokenized abstractions).

## Primary entry points (use these patterns)
- SFT runs via DeepSpeed module safe_rlhf.finetune (see scripts/sft.sh).
- Reward model runs via safe_rlhf.values.reward and cost model via safe_rlhf.values.cost (see scripts/reward-model.sh and scripts/cost-model.sh).
- PPO runs via safe_rlhf.algorithms.ppo and Safe-RLHF via safe_rlhf.algorithms.ppo_lag (see scripts/ppo.sh and scripts/ppo-lag.sh).
- __main__.py wrappers exist for finetune and algorithms/* so python -m safe_rlhf.<module> is also valid.

## Dataset conventions
- Training CLI flags accept DATASET[:PROPORTION[:PATH]] and are parsed by safe_rlhf.datasets.parse_dataset.
- New datasets are registered under safe_rlhf/datasets/raw and wired through RawDataset subclasses.

## Workflow notes
- scripts/*.sh set PYTHONPATH to repo root, manage output dirs, and run deepspeed with common defaults; prefer editing these scripts over ad-hoc commands.
- If WANDB_API_KEY is unset, scripts force WANDB_MODE=offline to avoid network issues.
- Multi-node runs are supported via --hostfile passed into the scripts (DeepSpeed hostfile flow).

## CHTC (HTCondor) flow
- Use chtc/*.sub and chtc/run_*.sh for UW-Madison CHTC; see chtc/README.md for staging, prebuilt env tarballs, and GPU constraints.

## Where to look when changing behavior
- Trainer behavior: safe_rlhf/finetune/trainer.py, safe_rlhf/trainers/, safe_rlhf/algorithms/*/trainer.py.
- Config assembly: safe_rlhf/configs/ (DeepSpeed config helpers and templates).
- Evaluation utilities: safe_rlhf/evaluate/ and scripts/arena-evaluation.sh.
