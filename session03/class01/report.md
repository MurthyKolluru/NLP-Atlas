# Session 1 — Evaluation & Reflection

*Date:* 2025-10-12 09:00

**New pairs evaluated:** 10

**Agreement (RM vs Heuristic):** 90.0%


## Observations

- RM confidence tends to be higher on prompts with clear stylistic cues (politeness/conciseness).

- Disagreements often occur when relevance and brevity conflict (long but on-topic vs short but shallow).


## Examples (first 3)

### Prompt
Summarize this: Large language models assist with drafting emails. (briefly)

- Heuristic winner: 1 (margin=0.002)

- RM winner: 1 (conf=0.50)

- R1: Summarize this: Large language models assist with drafting emails. (briefly)

















































- R2: Summarize this: Large language models assist with drafting emails. (briefly) .


### Prompt
Summarize this: Large language models assist with drafting emails. (briefly)

- Heuristic winner: 1 (margin=0.090)

- RM winner: 1 (conf=0.51)

- R1: Summarize this: Large language models assist with drafting emails. (briefly) The text is in the form of a text.







































- R2: Summarize this: Large language models assist with drafting emails. (briefly) There are a lot of other languages, but to give a more detailed understanding of the language features that language users should look forward to, see a few things that might be useful for future work.



For example, imagine that


### Prompt
Rephrase politely: Your assumption is incorrect. (nicely)

- Heuristic winner: 1 (margin=0.213)

- RM winner: 1 (conf=0.52)

- R1: Rephrase politely: Your assumption is incorrect. (nicely)

















































- R2: Rephrase politely: Your assumption is incorrect. (nicely)




This is not a comment on the whole, since the original piece didn't include the question, "Can I take the next step to bring you a piece that does what I've already agreed to?" I've seen


## Reflection: Why SFT Saturates & How RL Helps

Supervised fine-tuning maximizes likelihood of reference text but cannot directly target human-valued attributes like tone, brevity, or task adherence. A learned reward provides a *directional* preference signal: it can upweight polite, concise, and relevant behavior even when such outputs are less likely under the base model. This closes the gap between *what the model predicts* and *what users prefer*, enabling policy optimization in Session 2.

## Next Steps for Session 2
- Improve heuristic weights and regenerate pairs
- Train a stronger reward head (feature concat, more data)
- Use the RM to run **offline RL** (PPO/DPO variants) on cached generations
