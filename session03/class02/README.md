Session 03 / class02 — Reward-Model Evaluation & Offline RL Tuning

This lab complements Session 2 tutorial "Reward-Model Evaluation & Offline RL Tuning".

Objectives

Load the trained reward model (data/reward_model.pt)

Score cached generations (cache.csv)

Perform Offline PPO and DPO updates (demo-mode)

Compare reward distributions and record summary in report.md

Folder Layout
session03/
  └─ class02/
      ├─ data/
      │   ├─ synthetic_prompts.csv
      │   ├─ pairs.csv
      │   ├─ rewards.csv
      │   └─ reward_model.pt
      ├─ class02.ipynb
      ├─ utils_plot.py
      ├─ requirements.txt
      └─ report.md

Quickstart (local)
cd session03\class02
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# (Optional) CPU-only wheel for Windows:
# pip install --index-url https://download.pytorch.org/whl/cpu torch


Open class02.ipynb in VS Code → run Blocks 1–8 sequentially.

Instructor Notes

Generation length ≤ 64 tokens (for speed)

Offline PPO/DPO = visualization proxy, not full RLHF

Reward model remains frozen

If data files are missing, the notebook auto-creates small demo stubs so it still runs