# A real example of what a head's copying score means:
#
#   HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 python head_score_example.py <run folder>
#
# One forward pass of the finished model (last.pt) on the same copying test as probe_snapshots.py: 8 rows of 256
# random tokens (seed 0), each followed by the same 256 again. It prints:
#   - the copying and previous-token scores recomputed here, next to the ones in probes.pt (they should match)
#   - row 6 (believe multipl plant Nose ...): at the second "plant", where some heads' attention goes
#   - the first positions of row 6's second copy, each head's share on the token to copy, and that head's average
#     over all 2,048 positions (its score)

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint

ROW, SHOWN = 6, 8                                       # the row to show, and how many positions to list
HEADS = [(6, 0), (8, 4), (9, 3), (9, 11), (0, 0)]       # strongest copier, weakest counted, at the line, below it, a plain head


@torch.no_grad()
def main(args):
    run = Path(args.run_dir)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = load_checkpoint(run / "last.pt")
    for block in model.blocks:
        block.attn.fast = False                         # the hand-written attention, so the weights can be read
    vocab = torch.load(run / "last.pt", map_location="cpu", mmap=True, weights_only=False)["config"]["vocab_size"]
    half = 256
    random_ids = torch.randint(0, vocab, (8, half), generator=torch.Generator().manual_seed(0))
    tokens = torch.cat([random_ids, random_ids], dim=1)

    weights = []
    hooks = [block.attn.attn_dropout.register_forward_hook(lambda m, i, out: weights.append(out))
             for block in model.blocks]
    try:
        model(tokens)
    finally:
        for hook in hooks:
            hook.remove()
    weights = torch.stack(weights)                      # (blocks, rows, heads, query, key)

    later = torch.arange(half, 2 * half)
    every = torch.arange(1, 2 * half)
    to_copy = weights[:, :, :, later, later - half + 1]  # (blocks, rows, heads, 256)
    induction = to_copy.mean(dim=(1, 3))
    previous = weights[:, :, :, every, every - 1].mean(dim=(1, 3))
    saved = torch.load(run / "probes.pt", weights_only=False)
    print("== scores recomputed here | in probes.pt")
    for b, h in HEADS + [(5, 9)]:
        print(f"  {b}.{h:<3} copying {induction[b, h]:.4f} | {saved['induction_score'][-1][b, h]:.4f}"
              f"   previous-token {previous[b, h]:.4f} | {saved['previous_token_score'][-1][b, h]:.4f}")

    word = lambda i: repr(tokenizer.decode([tokens[ROW, i].item()]))
    query = half + 2                                    # the second "plant"
    print(f"\n== row {ROW}, reading position {query} {word(query)}; the token to copy is position 3 {word(3)}")
    for b, h in HEADS + [(5, 9)]:
        row = weights[b, ROW, h, query, :query + 1]
        top = row.topk(3)
        where = ", ".join(f"{word(k)} at {k} {v:.3f}" for v, k in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  {b}.{h:<3} on {word(3)}: {row[3]:.3f} | on the token before ({word(query - 1)}): {row[query - 1]:.3f}"
              f" | top 3: {where}")

    print(f"\n== row {ROW}, the first {SHOWN} positions of the second copy: each head's share on the token to copy")
    print("  reading -> to copy        " + "".join(f"{b}.{h:<6}" for b, h in HEADS))
    for j in range(SHOWN):
        name = f"{word(half + j)} -> {word(j + 1)}"
        print(f"  {name:25s}" + "".join(f"{to_copy[b, ROW, h, j]:<8.3f}" for b, h in HEADS))
    print("  " + "score (all 2,048)".ljust(25) + "".join(f"{induction[b, h]:<8.3f}" for b, h in HEADS))
    for b, h in HEADS:
        shares = to_copy[b, :, h].flatten()
        print(f"  {b}.{h}: share on the token to copy is above 0.5 at {(shares > 0.5).sum()} of {shares.numel()} "
              f"positions, below 0.1 at {(shares < 0.1).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a real example of a head's copying score.")
    parser.add_argument("run_dir", help="the run folder, with last.pt and probes.pt")
    main(parser.parse_args())
