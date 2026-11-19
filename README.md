# GPT-2 small training dynamics

These scripts measure when a GPT-2 small (124M), trained from scratch, learns what, and which attention heads do the copying. They use 190 saved models, which are snapshots after step 0, step 50 and every 100 steps from there, plus the finished model (`last.pt`). They ask every snapshot the same questions, score every attention head for copying, and read what the previous-token head writes and how the induction head uses it.

The write-up is the blog post [The exact moments GPT-2 small learns](https://johnmachado.dev/blog/what-gpt2-learns/).

## What you need first

- The model code (`gpt2.checkpoint`, `gpt2.train.device`, `gpt2.train.shards`) from [the training repo](https://github.com/JohnMachado11/GPT-2), in its `src/gpt2`. Put that `src` folder on `PYTHONPATH`.
- A run folder with `snapshots/` (the `step_*.pt` files), `last.pt` and `best.pt`. The training repo writes these when `save_snapshots = True` is set in `src/run/settings_gpt2_small.py` and you train with `src/run/train_gpt2.py`.
- A data folder written by the training repo's `src/run/prepare_data.py`. `probe_snapshots.py` uses its validation shard, and `copying_wiring.py` uses it as real text.
- Python 3 with `torch`, `numpy` and `transformers`.

## The order to run them

**Step 1. Measure every snapshot.** These three read all 190 models, so they need a GPU (they were run on one in Colab). They don't depend on each other. Each saves one file in the run folder, and each can be re-run after a disconnect because it skips snapshots it already did.

| Script | Command | Saves |
| --- | --- | --- |
| `probe_snapshots.py` | `python probe_snapshots.py <run folder> --data <data folder>` | `probes.pt` |
| `probe_snapshots_3.py` | `python probe_snapshots_3.py <run folder>` | `probes_3.pt` |
| `knowledge_paths.py` | `python knowledge_paths.py <run folder>` | `paths.pt` |

- `probe_snapshots.py` runs the copying test: 8 rows of 256 random tokens (seed 0), each shown twice. A forward hook on every block's attention weights gives each of the 144 heads an induction score (attention from the second copy to the token after the earlier copy of the current token) and a previous-token score (attention to the token just before), in every snapshot.
- `probe_snapshots_3.py` asks every snapshot the 372 prompts in `knowledge_probes.py` and saves the probability and rank of the right next token. `knowledge_probes.py` is only imported, never run on its own.
- `knowledge_paths.py` saves the top 10 guesses and their probabilities for every prompt in every snapshot.
- `subject_order.py` runs on `probes_3.pt` once it exists (`python subject_order.py <run folder>`, no model needed). It prints, for each of the 16 subjects, the arrival step, which is the step by which half of its prompts have had the right token as the top guess at least once.

**Step 2. Look inside the finished model.** These run on one checkpoint.

| Script | Command | Needs | Saves |
| --- | --- | --- | --- |
| `head_score_example.py` | `python head_score_example.py <run folder>` | `last.pt`, `probes.pt` from step 1 | prints a worked example |
| `induction_figure_data.py` | `python induction_figure_data.py <run folder>/best.pt --out induction_figure_data.json` | `best.pt` | the JSON |

- `head_score_example.py` reruns the copying test on the finished model and prints the scores next to the ones in `probes.pt`. They should match, which checks step 1.
- `induction_figure_data.py` takes one sentence ("The code word is velvet lamp. Again, the code word is velvet") and records what block 5, head 9 writes into the residual stream at each word (its "note"), and block 6, head 0's query, keys and attention. It then removes head 5.9's note at "lamp" only and recomputes the attention. It imports helpers from `copying_wiring.py`.
- `copying_wiring.py` can also be run on its own (`python copying_wiring.py <run folder>/best.pt --text <data folder>`). It tests whether head 5.9's note can be read, and whether the copying heads find their target through it.

## What it sends

The scripts call `AutoTokenizer.from_pretrained("gpt2")`. The first time, this downloads the GPT-2 tokenizer files from the Hugging Face hub, and after that it reads them from the local cache. To make sure nothing is contacted, set these.

```
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

`HF_HUB_OFFLINE=1` only works once the tokenizer is cached. Nothing else in these scripts makes a network call.
