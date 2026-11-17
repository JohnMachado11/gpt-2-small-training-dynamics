# Ask every snapshot of a run the same questions and save the answers. Meant for a GPU machine:
#
#   python probe_snapshots.py <run folder> --data <folder of data shards>
#
# For each snapshot, and last.pt (the finished model), it saves probes/step_NNNNN.pt in the run folder, then all of
# them together in probes.pt. Run again after a disconnect, it skips the snapshots it already did.
#
# What it measures on each one:
#   - weights: the size of each part's weights, and how much they changed since the snapshot before
#   - text: the loss on every token of 64 fixed validation windows (the in-context learning score comes from these)
#   - copying: random tokens, repeated. The loss at each position, and how much each head looks at the token right
#     after the earlier copy of the current one (an induction head) or at the token just before (previous-token head)
#   - probes: facts, grammar, patterns and copying, each a right and a wrong next word, read after every block (the
#     logit lens). The last block's reading is the model's own answer
#   - word meaning: how alike related words' embeddings are (cosine similarity)

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint, save_atomic
from gpt2.train.device import pick_device
from gpt2.train.shards import read_shard, shard_paths

# (kind, prompt, right next word, wrong next word) -- each answer is one token
PROBES = [
    ("fact", "The capital of France is", " Paris", " London"),
    ("fact", "The capital of Japan is", " Tokyo", " Beijing"),
    ("fact", "The capital of Italy is", " Rome", " Madrid"),
    ("fact", "The capital of Germany is", " Berlin", " Paris"),
    ("fact", "Paris is the capital of", " France", " Japan"),
    ("fact", "The largest planet in our solar system is", " Jupiter", " Mars"),
    ("fact", "The Earth orbits the", " sun", " moon"),
    ("fact", "The sun rises in the", " east", " west"),
    ("fact", "The Great Wall is a famous landmark in", " China", " India"),
    ("fact", "Water is made of hydrogen and", " oxygen", " carbon"),
    ("fact", "Water freezes at 0 degrees", " Celsius", " Fahrenheit"),
    ("fact", "The chemical symbol for gold is", " Au", " Ag"),
    ("fact", "The play Hamlet was written by William", " Shakespeare", " Dickens"),
    ("fact", "The first president of the United States was George", " Washington", " Lincoln"),
    ("fact", "Albert Einstein is famous for the theory of", " relativity", " evolution"),
    ("fact", "Charles Darwin is famous for the theory of", " evolution", " relativity"),
    ("fact", "The opposite of hot is", " cold", " warm"),
    ("grammar", "The keys to the cabinet", " are", " is"),
    ("grammar", "The dogs in the park", " are", " is"),
    ("grammar", "The children", " are", " is"),
    ("grammar", "The student", " is", " are"),
    ("grammar", "She", " has", " have"),
    ("grammar", "Yesterday, we", " went", " go"),
    ("grammar", "One cat, two", " cats", " cat"),
    ("grammar", "One mouse, two", " mice", " mouse"),
    ("grammar", "I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked", " an", " a"),
    ("pattern", "Monday, Tuesday, Wednesday,", " Thursday", " Friday"),
    ("pattern", "January, February, March,", " April", " May"),
    ("pattern", "1, 2, 3, 4,", " 5", " 6"),
    ("pattern", "one, two, three, four,", " five", " six"),
    ("pattern", "A, B, C, D,", " E", " F"),
    ("pattern", "(for example, see the figure below", ")", "."),
    ("copying", "The secret word is banana. Remember the secret word. The secret word is", " banana", " apple"),
    ("copying", "My favorite color is green. I like it a lot. My favorite color is", " green", " blue"),
]

# word pairs whose embeddings should grow alike, and one unrelated pair to compare with
PAIRS = [(" king", " queen"), (" man", " woman"), (" mother", " father"), (" cat", " dog"), (" big", " large"),
         (" good", " bad"), (" happy", " sad"), (" Paris", " France"), (" apple", " Jupiter")]

WINDOWS = 64        # validation windows, the same ones for every snapshot
REPEATS = 8         # rows of random tokens for the copying test, each up to 256 tokens and then the same again


def parts_of(model):
    # each part's weights as one long vector -- the same 51 parts as the training history's gradient norms
    # (loop.py). named_parameters() lists the tied unembedding once, as the token embedding
    parts = {}
    for name, weight in model.named_parameters():
        part = ".".join(name.split(".")[:3]) if name.startswith("blocks.") else name.split(".")[0]
        parts.setdefault(part, []).append(weight.detach().flatten())
    return {part: torch.cat(weights) for part, weights in parts.items()}


@torch.no_grad()
def token_losses(model, windows):
    # the loss on every token of every window, (windows, context_length), 8 windows at a time
    losses = []
    for chunk in windows.split(8):
        logits, _ = model(chunk[:, :-1])
        losses.append(F.cross_entropy(logits.transpose(1, 2), chunk[:, 1:], reduction="none"))
    return torch.cat(losses)


@torch.no_grad()
def copying(model, tokens):
    # tokens: rows of random ids, each one twice over. Returns the loss at each position (averaged over the rows)
    # and, for every block and head, the induction and previous-token scores: the average attention weight from
    # each position to the token after the earlier copy of its own token, and to the token just before it
    half = tokens.shape[1] // 2
    later = torch.arange(half, 2 * half, device=tokens.device)          # the positions in the second copy
    every = torch.arange(1, 2 * half, device=tokens.device)
    scores = []

    def look(module, inputs, weights):      # the attention weights after the softmax, (rows, heads, query, key)
        scores.append((weights[:, :, later, later - half + 1].mean(dim=(0, 2)),
                       weights[:, :, every, every - 1].mean(dim=(0, 2))))

    hooks = [block.attn.attn_dropout.register_forward_hook(look) for block in model.blocks]
    try:
        logits, _ = model(tokens)
    finally:
        for hook in hooks:
            hook.remove()
    loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none").mean(dim=0)
    induction, previous = (torch.stack(blocks) for blocks in zip(*scores))      # (blocks, heads) each
    return loss, induction, previous


@torch.no_grad()
def ask(model, prompt, right, wrong):
    # the logit lens: the model's next-word guess read after each block, through the final LayerNorm and the
    # unembedding. Returns the right and the wrong word's log-probability and the right one's rank (0 = top
    # guess), each (blocks,). After the last block, that's the model's own answer
    after = []
    hooks = [block.register_forward_hook(lambda module, inputs, out: after.append(out[0, -1]))
             for block in model.blocks]
    try:
        model(prompt)
    finally:
        for hook in hooks:
            hook.remove()
    logprobs = F.log_softmax(model.unembed(model.final_norm(torch.stack(after))), dim=-1)     # (blocks, vocab)
    return logprobs[:, right], logprobs[:, wrong], (logprobs > logprobs[:, [right]]).sum(dim=-1)


def measure(model, before, windows, repeated, probes, pairs):
    now = parts_of(model)
    copying_loss, induction, previous = copying(model, repeated)
    answers = [torch.stack(answer) for answer in zip(*(ask(model, *probe) for probe in probes))]
    embedding = model.token_embedding.weight
    return now, {
        "weight_norm": torch.stack([weights.norm() for weights in now.values()]),
        "weight_change": torch.stack([(now[part] - before[part]).norm() for part in now]) if before
                         else torch.full((len(now),), float("nan")),
        "val_token_loss": token_losses(model, windows).half(),           # half precision: 25 MB for a whole run
        "copying_loss": copying_loss,
        "induction_score": induction,
        "previous_token_score": previous,
        "probe_right": answers[0],                                       # (probes, blocks) each
        "probe_wrong": answers[1],
        "probe_rank": answers[2],
        "embedding_cosine": torch.stack([F.cosine_similarity(embedding[a], embedding[b], dim=0) for a, b in pairs]),
    }


def main(args):
    run = Path(args.run_dir)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def token(word):
        ids = tokenizer.encode(word)
        if len(ids) != 1:
            raise SystemExit(f"{word!r} is {len(ids)} tokens -- a probe's answers must be one token each")
        return ids[0]

    # every snapshot in step order, then the finished model
    todo = [(int(path.stem.split("_")[1]), path) for path in sorted((run / "snapshots").glob("step_*.pt"))]
    last = torch.load(run / "last.pt", map_location="cpu", mmap=True, weights_only=False)
    todo.append((last["step"] + 1, run / "last.pt"))
    config = last["config"]

    # the same inputs for every snapshot: validation windows spread evenly over the first val shard, random tokens
    # repeated (seed 0), the probes' prompts and answers, the word pairs
    val = read_shard(shard_paths(args.data, "val")[0])
    starts = np.linspace(0, len(val) - config["context_length"] - 1, WINDOWS).astype(np.int64)
    windows = np.stack([val[start:start + config["context_length"] + 1] for start in starts])
    windows = torch.tensor(windows.astype(np.int64), device=device)
    half = min(256, config["context_length"] // 2)
    random_ids = torch.randint(0, config["vocab_size"], (REPEATS, half), generator=torch.Generator().manual_seed(0))
    repeated = torch.cat([random_ids, random_ids], dim=1).to(device)
    probes = [(torch.tensor([tokenizer.encode(prompt)], device=device), token(right), token(wrong))
              for _, prompt, right, wrong in PROBES]
    pairs = [(token(a), token(b)) for a, b in PAIRS]

    print(f"{len(todo)} models to probe (the snapshots, then last.pt) on {device}")
    before, started, done = None, time.time(), 0
    for i, (steps_done, path) in enumerate(todo):
        out = run / "probes" / f"step_{steps_done:05d}.pt"
        if out.exists():
            before = None
            continue
        if before is None and i > 0:            # resuming: the model before this one again, for the weight change
            before = parts_of(load_checkpoint(todo[i - 1][1], device))
        model = load_checkpoint(path, device)
        before, result = measure(model, before, windows, repeated, probes, pairs)
        result = {key: value.cpu() for key, value in result.items()}
        save_atomic(result | {"steps_done": steps_done, "parts": list(before)}, out)
        done += 1
        left = (time.time() - started) / done * (len(todo) - i - 1)
        print(f"after {steps_done:>6,} steps | val loss {result['val_token_loss'].float().mean():.3f} | "
              f"copying loss {result['copying_loss'][half:].mean():.3f} | ~{left / 60:.0f} min left", flush=True)

    results = [torch.load(run / "probes" / f"step_{steps_done:05d}.pt") for steps_done, _ in todo]
    combined = {key: torch.stack([result[key] for result in results])
                for key in results[0] if key not in ("steps_done", "parts")}
    combined |= {"steps": [steps_done for steps_done, _ in todo], "parts": results[0]["parts"], "probes": PROBES,
                 "pairs": PAIRS, "window_starts": starts.tolist(), "copying_half": half}
    save_atomic(combined, run / "probes.pt")
    print(f"saved {run / 'probes.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe every snapshot of a run and save the answers.")
    parser.add_argument("run_dir", help="the run folder, with snapshots/ and last.pt")
    parser.add_argument("--data", required=True, help="the folder prepare_data.py wrote (its val shard is used)")
    main(parser.parse_args())
