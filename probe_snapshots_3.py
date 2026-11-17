# Asks every snapshot the probe set in knowledge_probes.py (a few hundred facts, word meanings, grammar, sequences
# and copying, in categories), read after every block like probe_snapshots.py's probes. Meant for a GPU machine, with
# probe_snapshots.py and knowledge_probes.py in the same folder as this script:
#
#   python probe_snapshots_3.py <run folder>
#
# For each snapshot, and last.pt, it saves probes_3/step_NNNNN.pt in the run folder, then all of them together in
# probes_3.pt. Run again after a disconnect, it skips the snapshots it already did.
#
# Decided before the results: a probe counts as learned at the first snapshot from which the
# right answer beats the wrong one in every later model, and as known at the first snapshot from which the right
# answer is the model's top guess in every later model (which may never happen).

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint, save_atomic
from gpt2.train.device import pick_device
from knowledge_probes import probes
from probe_snapshots import ask


def main(args):
    run = Path(args.run_dir)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    chosen = probes(tokenizer)
    asked = [(torch.tensor([tokenizer.encode(prompt)], device=device), tokenizer.encode(right)[0],
              tokenizer.encode(wrong)[0]) for _, prompt, right, wrong in chosen]

    todo = [(int(path.stem.split("_")[1]), path) for path in sorted((run / "snapshots").glob("step_*.pt"))]
    last = torch.load(run / "last.pt", map_location="cpu", mmap=True, weights_only=False)
    todo.append((last["step"] + 1, run / "last.pt"))

    print(f"{len(todo)} models, {len(chosen)} probes each, on {device}")
    started, done = time.time(), 0
    for i, (steps_done, path) in enumerate(todo):
        out = run / "probes_3" / f"step_{steps_done:05d}.pt"
        if out.exists():
            continue
        model = load_checkpoint(path, device)
        right, wrong, rank = (torch.stack(answer).cpu() for answer in zip(*(ask(model, *probe) for probe in asked)))
        save_atomic({"right": right, "wrong": wrong, "rank": rank.int(), "steps_done": steps_done}, out)
        done += 1
        left = (time.time() - started) / done * (len(todo) - i - 1)
        print(f"after {steps_done:>6,} steps | right over wrong {(right[:, -1] > wrong[:, -1]).float().mean():.0%} | "
              f"top guess {(rank[:, -1] == 0).float().mean():.0%} | ~{left / 60:.0f} min left", flush=True)

    results = [torch.load(run / "probes_3" / f"step_{steps_done:05d}.pt") for steps_done, _ in todo]
    combined = {key: torch.stack([result[key] for result in results]) for key in ("right", "wrong", "rank")}
    combined |= {"steps": [steps_done for steps_done, _ in todo], "probes": chosen}  # (models, probes, blocks)
    save_atomic(combined, run / "probes_3.pt")
    print(f"saved {run / 'probes_3.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asks every snapshot of a run the knowledge probes.")
    parser.add_argument("run_dir", help="the run folder, with snapshots/ and last.pt")
    main(parser.parse_args())
