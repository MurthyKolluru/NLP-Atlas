Here’s a **ready-to-paste README.md** for your
`C:\Users\murth\Desktop\NLP+RL\code\class04\` folder.

---

```markdown
# 🧪 Lab 04 — Alignment Deployment & Monitoring

This lab belongs to **Session 04** of the *RL for LLMs* course.

It simulates a **post-deployment alignment loop** that evaluates an aligned
policy under live-like conditions — computing reward, KL, entropy and drift,
and triggering rollback alerts when thresholds are breached.

---

## 📂 Folder Layout
```

class04/
├─ notebooks/
│   └─ lab04_monitoring.ipynb     ← main notebook (Blocks 1–7)
├─ data/                          ← eval_prompts.csv, cache.csv
├─ checkpoints/                   ← policy.pt, reward_model.pt
├─ logs/                          ← metrics_step*.json, alerts, FROZEN.flag
├─ figs/                          ← monitoring.png, monitoring_smoke.png
├─ config/
│   └─ lab04.yaml                 ← rollout + threshold settings
├─ scripts/                       ← optional helper scripts
└─ .gitignore

````

---

## 🚀 Quick Start

1. **Activate venv**

   ```powershell
   cd "C:\Users\murth\Desktop\NLP+RL\code"
   .\.venv\Scripts\Activate.ps1
````

2. **Open the notebook**

   `class04/notebooks/lab04_monitoring.ipynb`
   (select the same `.venv` kernel).

3. **Run cells Block 1 → 7**
   Block 4 performs evaluation; Blocks 5–6 generate alerts and the monitoring plot.

---

## 🧩 Configuration (`config/lab04.yaml`)

| Key                      | Meaning                            | Typical Value |
| ------------------------ | ---------------------------------- | ------------- |
| `rollout.batch_size`     | prompts per step                   | 16            |
| `rollout.max_steps`      | number of rollout iterations       | 4             |
| `thresholds.reward_drop` | min acceptable Δreward vs baseline | -0.05         |
| `thresholds.kl_max`      | max average KL                     | 0.20          |
| `thresholds.entropy_min` | min entropy (avoid collapse)       | 3.5           |
| `thresholds.drift_max`   | max drift index                    | 0.30          |

Edit, save, then re-run Blocks 4–6.

---

## 📊 Outputs

* `logs/metrics_step*.json` — per-step metrics
* `logs/last_alert.json` — last triggered alert
* `logs/FROZEN.flag` — created when freeze condition met
* `figs/monitoring.png` — full run
* `figs/monitoring_smoke.png` — quick smoke test

---

## ⚙️ Using Real Artifacts

Drop your real checkpoints and evaluation prompts:

```
class04/checkpoints/policy.pt
class04/checkpoints/reward_model.pt
class04/data/eval_prompts.csv
```

Then rerun Blocks 4 → 6 for live metrics.

---

## 🧭 Troubleshooting

| Issue                                            | Fix                                             |
| ------------------------------------------------ | ----------------------------------------------- |
| **YAML comment strings** cause float cast errors | Use plain numbers or run updated Block 5 parser |
| **pip points to OneDrive**                       | Rebuild venv locally (`python -m venv .venv`)   |
| **Matplotlib plots blank in VS Code**            | Enable “Inline” or “Interactive Window” output  |
| **CUDA unavailable**                             | All code runs CPU-only; GPU optional            |

---

## 🧠 Learning Objective Recap

You now operate the full **alignment monitoring pipeline**:

> *Evaluate → Compare → Detect → Freeze → Rollback → Audit.*

This closes the loop from **pre-training → fine-tuning → reward → PPO/DPO → deployment**.