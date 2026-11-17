# A pass over every snapshot: each knowledge probe's top 10 guesses after the last block (the path to the right
# answer), and the embeddings of the countries, capitals and languages whose names are one token (the map). Meant for
# a GPU machine, with knowledge_probes.py in the same folder as this script:
#
#   python knowledge_paths.py <run folder>
#
# For each snapshot, and last.pt, it saves paths/step_NNNNN.pt in the run folder, then all of them together in
# paths.pt. Run again after a disconnect, it skips the snapshots it already did.

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint, save_atomic
from gpt2.train.device import pick_device
from knowledge_probes import CAPITALS, LANGUAGES, probes

GUESSES = 10


def map_words(tokenizer):
    # the countries, capitals and languages whose names are one token (with a leading space), each word once, and
    # the country -> capital and country -> language pairs between them, as positions in the word list
    one_token = lambda word: len(tokenizer.encode(" " + word)) == 1
    capitals = [(country, capital) for country, capital in CAPITALS if one_token(country) and one_token(capital)]
    languages = [(country, language) for country, language in LANGUAGES if one_token(country) and one_token(language)]
    words, groups = [], []
    for group, names in (("country", [c for c, _ in capitals + languages]), ("capital", [k for _, k in capitals]),
                         ("language", [lang for _, lang in languages])):
        for name in dict.fromkeys(names):
            if name not in words:
                words.append(name)
                groups.append(group)
    pairs = lambda found: [(words.index(a), words.index(b)) for a, b in found]
    return {"words": words, "groups": groups, "ids": [tokenizer.encode(" " + w)[0] for w in words],
            "capital_pairs": pairs(capitals), "language_pairs": pairs(languages)}


@torch.no_grad()
def top_guesses(model, asked):
    # each prompt's top GUESSES next tokens after the last block, and their probabilities: (probes, GUESSES) each
    picked = [torch.softmax(model(prompt)[0][0, -1].float(), dim=-1).topk(GUESSES) for prompt in asked]
    return torch.stack([p.indices for p in picked]).int().cpu(), torch.stack([p.values for p in picked]).cpu()


def main(args):
    run = Path(args.run_dir)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    chosen = probes(tokenizer)
    asked = [torch.tensor([tokenizer.encode(prompt)], device=device) for _, prompt, _, _ in chosen]
    words = map_words(tokenizer)

    todo = [(int(path.stem.split("_")[1]), path) for path in sorted((run / "snapshots").glob("step_*.pt"))]
    last = torch.load(run / "last.pt", map_location="cpu", mmap=True, weights_only=False)
    todo.append((last["step"] + 1, run / "last.pt"))

    print(f"{len(todo)} models, {len(chosen)} probes and {len(words['words'])} map words each, on {device}")
    started, done = time.time(), 0
    for i, (steps_done, path) in enumerate(todo):
        out = run / "paths" / f"step_{steps_done:05d}.pt"
        if out.exists():
            continue
        model = load_checkpoint(path, device)
        top_ids, top_p = top_guesses(model, asked)
        embedding = model.token_embedding.weight[words["ids"]].float().cpu()
        save_atomic({"top_ids": top_ids, "top_p": top_p, "embedding": embedding, "steps_done": steps_done}, out)
        done += 1
        left = (time.time() - started) / done * (len(todo) - i - 1)
        print(f"after {steps_done:>6,} steps | top guess right on "
              f"{sum(int(top_ids[p, 0]) == tokenizer.encode(right)[0] for p, (_, _, right, _) in enumerate(chosen))} "
              f"of {len(chosen)} probes | ~{left / 60:.0f} min left", flush=True)

    results = [torch.load(run / "paths" / f"step_{steps_done:05d}.pt") for steps_done, _ in todo]
    combined = {key: torch.stack([result[key] for result in results]) for key in ("top_ids", "top_p", "embedding")}
    combined |= {"steps": [steps_done for steps_done, _ in todo], "probes": chosen, "map": words}
    save_atomic(combined, run / "paths.pt")
    print(f"saved {run / 'paths.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record each probe's top guesses and the map words' embeddings in "
                                                 "every snapshot.")
    parser.add_argument("run_dir", help="the run folder, with snapshots/ and last.pt")
    main(parser.parse_args())
