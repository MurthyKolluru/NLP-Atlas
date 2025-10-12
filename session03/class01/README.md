# 🧠 Class 01 — From Supervised Fine-Tuning to Reinforcement Learning  
*(Part of the “RL for LLMs” Course)*

This folder contains all materials and outputs for **Class 01**, which demonstrates how to move from **Supervised Fine-Tuning (SFT)** to **Reward-based Optimization** using lightweight, open models.

---

## 📘 Overview

Students learn to:
1. Understand the conceptual shift from **likelihood maximization** to **reward optimization**.  
2. Design **heuristic rewards** for text tasks (relevance, brevity, politeness, toxicity).  
3. Generate **synthetic prompts & responses** using small models (`distilgpt2`, `flan-t5-small`).  
4. Train a minimal **Reward Model** on pairwise preferences.  
5. Evaluate the RM and reflect on alignment vs. over-optimization.

---

## 📂 Folder Structure

| File / Folder | Description |
|----------------|-------------|
| `class01.ipynb` | Full Colab-ready notebook for Session 1 Lab |
| `synthetic_prompts.csv` | Generated prompt–response dataset |
| `rewards.csv` | Heuristic component scores & total reward |
| `pairs.csv` | Preference pairs for RM training |
| `reward_model.pt` | Trained mini reward model |
| `eval_pairs.csv` | Evaluation results comparing heuristic vs. RM |
| `report.md` | Reflection summary (“Why SFT saturates and how RL helps”) |
| `requirements.txt` | Minimal dependency list for reproduction |
| `.gitignore` | Excludes venv, caches, checkpoints |

---

## ⚙️ Environment Setup

```bash
# From repo root
cd session03/class01
python -m venv .venv
.\.venv\Scripts\activate       # (Windows PowerShell)
pip install -r requirements.txt
The lab runs fine on CPU.
GPU (T4/A10/A100) simply speeds up generation and embedding.
________________________________________
🧪 Quick Start
Open class01.ipynb in VS Code or Jupyter Lab:
1.	Run Block 1 → Block 8 in order.
2.	Verify the outputs (*.csv, reward_model.pt, report.md).
3.	Use these artifacts as inputs for Session 2 (Offline RL Tuning) in ../session02/.
________________________________________
🧭 Next
Continue with Session 2 / Class 02 →
session03/class02/ (planned) for Reward Model Evaluation & Offline RL Simulation.
________________________________________
Author: Murthy Kolluru
Repository: NLP-Atlas
License: MIT

Drop that file in the folder, commit, and push:

```powershell
git add session03/class01/README.md
git commit -m "Add README for class01 (SFT → RL lab)"
git push origin main
