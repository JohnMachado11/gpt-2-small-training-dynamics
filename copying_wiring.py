# How copying is wired, tested directly on a finished model: does head 5.9 leave a note saying which token came just
# before, can the note be read, and do the copying heads find their target by that note?
#
#   python copying_wiring.py <run folder>/best.pt --text <folder of text shards>
#
# A head's "note" is what it adds to the residual stream: its output (64 numbers, embed_dim / num_heads), through
# its slice of attention's out_proj, the shared bias left out.
#   1. Read the note, three ways.
#      a. Map it onto the embedding: fit a linear map from head 5.9's note to the embedding of the token just before
#         (on random tokens), then on other random tokens and on real text, see which of the 50,257 embeddings it
#         lands closest to. Compared with decoding the current token or the one two back, and with every other head
#         in block 5.
#      b. Look it up: record the note the head writes after each of the 50,257 tokens (each placed once as the token
#         before, in random context), then read a note as the token whose recorded note points the most the same way
#         (the highest cosine similarity). On the same positions as (a), compared with the other block-5 heads, and
#         on real text with always guessing the most common token.
#      c. Match the fingerprint: in random tokens where the second half repeats every other token of the first, a
#         position whose current token is new but whose token before also came before some first-half position: is
#         that first-half position the one with the most similar note? Compared with the other block-5 heads.
#   2. Do the copying heads look for it? Their copying scores on repeated random tokens (as in probe_snapshots.py)
#      with head 5.9's note removed from what they match against (their keys), or from what they match with (their
#      queries), or replaced in their keys by its average or by the notes of shuffled positions, or with another
#      block-5 head's note removed from their keys instead. Only the direct path changes: what block 5's
#      feed-forward and blocks 6 and 7 wrote from the note is left as it was.
#   3. Swap the notes: swap head 5.9's notes between a copying head's real target and another position, and see
#      where its attention goes. Compared with swapping between two positions that aren't the target.

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from gpt2.checkpoint import load_checkpoint
from gpt2.train.device import pick_device
from gpt2.train.shards import read_shard, shard_paths

NOTE_HEAD = (5, 9)
COPYING_HEADS = [(6, 0), (6, 4), (7, 1), (7, 10), (8, 4), (8, 5)]       # the heads that scored above 0.3 on probe_snapshots.py's copying test in this run


@torch.no_grad()
def look_inside(model, tokens):
    # one ordinary forward pass, keeping the residual stream entering each block and each block's head outputs
    # (before out_proj mixes them)
    entering, heads, hooks = {}, {}, []
    for b, block in enumerate(model.blocks):
        hooks.append(block.register_forward_pre_hook(lambda module, args, b=b: entering.__setitem__(b, args[0])))
        hooks.append(block.attn.out_proj.register_forward_pre_hook(
            lambda module, args, b=b: heads.__setitem__(b, args[0])))
    try:
        model(tokens)
    finally:
        for hook in hooks:
            hook.remove()
    return entering, heads


def note(model, heads, block, head):
    # what one head adds to the residual stream: its output through its own slice of out_proj (the shared bias left
    # out), (rows, positions, embed_dim)
    size = model.blocks[block].attn.head_dim
    part = slice(head * size, (head + 1) * size)
    return heads[block][..., part] @ model.blocks[block].attn.out_proj.weight[:, part].T


@torch.no_grad()
def attention(block, asking, answering):
    # a block's attention weights, every head, with the queries made from `asking` and the keys from `answering`
    attn = block.attn
    rows, positions, size = asking.shape
    query = attn.qkv(block.norm_before_attn(asking))[..., :size]
    key = attn.qkv(block.norm_before_attn(answering))[..., size:2 * size]
    query, key = (t.view(rows, positions, attn.num_heads, attn.head_dim).transpose(1, 2) for t in (query, key))
    scores = (query @ key.transpose(-2, -1)) / attn.head_dim ** 0.5
    scores = scores.masked_fill(~attn.causal_mask[:positions, :positions], float("-inf"))
    return torch.softmax(scores, dim=-1)                         # (rows, heads, query position, key position)


def copying_score(weights, half):
    # from each position in the second copy, the attention on the token right after the same token in the first copy
    later = torch.arange(half, 2 * half, device=weights.device)
    return weights[:, :, later, later - half + 1].mean(dim=(0, 2))  # (heads,)


@torch.no_grad()
def read_the_note(model, device, text):
    # fit on 48 rows of random tokens, test on 16 more and on real text
    embedding = model.token_embedding.weight
    unit = F.normalize(embedding, dim=-1)
    random_ids = torch.randint(0, embedding.shape[0], (64, 256), generator=torch.Generator().manual_seed(1)).to(device)
    _, heads = look_inside(model, random_ids)
    _, text_heads = look_inside(model, text)

    def decode(block, head, back):
        # fit note -> embedding of the token `back` positions earlier; accuracy = how often the closest token is it
        fit = note(model, heads, block, head)[:48, 2:].reshape(-1, embedding.shape[1])
        target = embedding[random_ids[:48, 2 - back:256 - back].reshape(-1)]
        ridge = 1e-3 * fit.pow(2).sum(dim=1).mean()
        mapping = torch.linalg.solve(fit.T @ fit + ridge * torch.eye(fit.shape[1], device=device), fit.T @ target)
        results = []
        for notes, ids in ((note(model, heads, block, head)[48:], random_ids[48:]),
                           (note(model, text_heads, block, head), text)):
            guess = (F.normalize(notes[:, 2:] @ mapping, dim=-1) @ unit.T).argmax(dim=-1)
            results.append((guess == ids[:, 2 - back:ids.shape[1] - back]).float().mean().item())
        return results

    print("\n== 1a. Reading the note by mapping it onto the embedding: how often a linear map from a head's note names "
          "the right token, on 16 rows of random tokens it wasn't fitted on | on real text")
    random_hit, text_hit = decode(*NOTE_HEAD, back=1)
    print(f"  head {NOTE_HEAD[0]}.{NOTE_HEAD[1]}, the token just before: {random_hit:.1%} | {text_hit:.1%}")
    for back, label in ((0, "the current token"), (2, "the token two back")):
        hits = decode(*NOTE_HEAD, back=back)
        print(f"  head {NOTE_HEAD[0]}.{NOTE_HEAD[1]}, {label}: {hits[0]:.1%} | {hits[1]:.1%}")
    others = {h: decode(NOTE_HEAD[0], h, back=1) for h in range(model.config.num_heads) if h != NOTE_HEAD[1]}
    best = max(others, key=lambda h: others[h][0])
    print(f"  the other block-{NOTE_HEAD[0]} heads, the token just before: at most {others[best][0]:.1%} | "
          f"{max(hits[1] for hits in others.values()):.1%} (best on random tokens: head {NOTE_HEAD[0]}.{best})")

    # b. look it up: the notes written after each of the 50,257 tokens, 255 tokens per row of random context
    vocab, width = embedding.shape
    rows = -(-vocab // 255)
    placed = torch.randperm(vocab, generator=torch.Generator().manual_seed(5))
    placed = torch.cat([placed, placed[:rows * 255 - vocab]]).view(rows, 255)
    context = torch.randint(0, vocab, (rows, 256), generator=torch.Generator().manual_seed(6))
    context[:, :255] = placed                        # so position p + 1's token before is placed[:, p]
    recorded = torch.zeros(vocab, width, device=device)       # block 5's head outputs, all heads, after each token
    for chunk in range(0, rows, 32):
        _, chunk_heads = look_inside(model, context[chunk:chunk + 32].to(device))
        recorded[placed[chunk:chunk + 32].reshape(-1).to(device)] = chunk_heads[NOTE_HEAD[0]][:, 1:].reshape(-1, width)

    def look_up(block, head):
        table = F.normalize(note(model, {block: recorded}, block, head), dim=-1)  # (vocab, embed_dim)
        hits = []
        for notes, ids in ((note(model, heads, block, head)[48:], random_ids[48:]),
                           (note(model, text_heads, block, head), text)):
            guess = torch.cat([(F.normalize(row[2:], dim=-1) @ table.T).argmax(dim=-1) for row in notes])
            hits.append((guess == ids[:, 1:ids.shape[1] - 1].reshape(-1)).float().mean().item())
        return hits
    print("\n== 1b. Reading the note by looking it up: how often the token whose recorded note matches best is the "
          "token just before, on the same positions as 1a")
    random_hit, text_hit = look_up(*NOTE_HEAD)
    print(f"  head {NOTE_HEAD[0]}.{NOTE_HEAD[1]}: {random_hit:.1%} | {text_hit:.1%}")
    others = {h: look_up(NOTE_HEAD[0], h) for h in range(model.config.num_heads) if h != NOTE_HEAD[1]}
    best = max(others, key=lambda h: others[h][0])
    print(f"  the other block-{NOTE_HEAD[0]} heads: at most {others[best][0]:.1%} | "
          f"{max(hits[1] for hits in others.values()):.1%} (best on random tokens: head {NOTE_HEAD[0]}.{best})")
    before = text[:, 1:text.shape[1] - 1].reshape(-1)
    print(f"  on the real text, always guessing its most common token before: "
          f"{before.bincount().max().item() / len(before):.1%}")

    # c. fingerprints: the second half keeps the first half's tokens at odd positions and puts new ones at even
    # positions, so an even position 256 + i has a new current token but the same token before as position i
    first = torch.randint(0, embedding.shape[0], (8, 256), generator=torch.Generator().manual_seed(3))
    second = first.clone()
    second[:, 0::2] = torch.randint(0, embedding.shape[0], (8, 128), generator=torch.Generator().manual_seed(4))
    _, fingerprint_heads = look_inside(model, torch.cat([first, second], dim=1).to(device))
    even = torch.arange(2, 256, 2, device=device)

    def match(block, head):
        notes = F.normalize(note(model, fingerprint_heads, block, head), dim=-1)
        similar = notes[:, 256 + even] @ notes[:, 1:256].transpose(1, 2)          # against first-half positions 1-255
        return (similar.argmax(dim=-1) + 1 == even).float().mean().item()
    print(f"\n== 1c. Matching the fingerprint: for {len(even)} positions per row x 8 rows, how often the most similar "
          f"of 255 first-half notes is the one after the same token (chance: 0.4%)")
    print(f"  head {NOTE_HEAD[0]}.{NOTE_HEAD[1]}: {match(*NOTE_HEAD):.1%}")
    others = {h: match(NOTE_HEAD[0], h) for h in range(model.config.num_heads) if h != NOTE_HEAD[1]}
    best = max(others, key=others.get)
    print(f"  the other block-{NOTE_HEAD[0]} heads: at most {others[best]:.1%} (head {NOTE_HEAD[0]}.{best})")

    # an example on real text: each token, and what head 5.9's note at the next position says came before it, read
    # by looking it up
    table = F.normalize(note(model, {NOTE_HEAD[0]: recorded}, *NOTE_HEAD), dim=-1)
    return (F.normalize(note(model, text_heads, *NOTE_HEAD)[0, 1:16], dim=-1) @ table.T).argmax(dim=-1)


@torch.no_grad()
def wiring(model, device):
    random_ids = torch.randint(0, model.config.vocab_size, (8, 256), generator=torch.Generator().manual_seed(0))
    tokens = torch.cat([random_ids, random_ids], dim=1).to(device)
    half = 256
    entering, heads = look_inside(model, tokens)
    the_note = note(model, heads, *NOTE_HEAD)

    # check: attention() reproduces the model's own attention weights
    seen = []
    hook = model.blocks[6].attn.attn_dropout.register_forward_hook(lambda module, inputs, out: seen.append(out))
    try:
        model(tokens)
    finally:
        hook.remove()
    gap = (attention(model.blocks[6], entering[6], entering[6]) - seen[0]).abs().max().item()
    print(f"\ncheck: recomputed attention vs the model's own, block 6: worst gap {gap:.1e}")

    print("\n== 2. The copying heads' copying scores on repeated random tokens, with head "
          f"{NOTE_HEAD[0]}.{NOTE_HEAD[1]}'s note removed")
    average = the_note.mean(dim=(0, 1), keepdim=True)
    shuffled = the_note[:, torch.randperm(2 * half, generator=torch.Generator().manual_seed(7))]
    print("   head  | as is | from the keys | from the queries | keys: its average instead | keys: shuffled positions' "
          "notes | another block-5 head's note from the keys (lowest, which)")
    for block, head in COPYING_HEADS:
        x = entering[block]
        as_is = copying_score(attention(model.blocks[block], x, x), half)[head]
        no_keys = copying_score(attention(model.blocks[block], x, x - the_note), half)[head]
        no_queries = copying_score(attention(model.blocks[block], x - the_note, x), half)[head]
        averaged = copying_score(attention(model.blocks[block], x, x - the_note + average), half)[head]
        mixed = copying_score(attention(model.blocks[block], x, x - the_note + shuffled), half)[head]
        controls = {h: copying_score(attention(model.blocks[block], x, x - note(model, heads, NOTE_HEAD[0], h)),
                                     half)[head] for h in range(model.config.num_heads) if h != NOTE_HEAD[1]}
        lowest = min(controls, key=controls.get)
        print(f"  {block}.{head:<4d} | {as_is:.3f} | {no_keys:13.3f} | {no_queries:16.3f} | {averaged:25.3f} | "
              f"{mixed:30.3f} | {controls[lowest]:.3f} (head {NOTE_HEAD[0]}.{lowest})")

    print("\n== 3. Swapping notes: from a position in the second copy holding token A, a copying head's attention on "
          "its real target\n   (the first-copy position after A) and on another first-copy position, before and after "
          "swapping their head-" + f"{NOTE_HEAD[0]}.{NOTE_HEAD[1]} notes,\n   and after swapping the other "
          "position's note with a third one instead (8 rows x 32 random pairs)")
    pairs = torch.randint(1, half - 1, (8, 32, 2), generator=torch.Generator().manual_seed(2))
    print("   head  | before: real, other | real and other swapped: real, other | other and a third swapped: real, "
          "other")
    for block, head in COPYING_HEADS:
        x = entering[block]
        found = {"before": [], "swapped": [], "control": []}
        for row in range(8):
            normal = attention(model.blocks[block], x[row:row + 1], x[row:row + 1])[0, head]
            for j, k in pairs[row].tolist():
                third = (k + 7) % (half - 2) + 1
                if len({j, k, third}) < 3 or abs(j - k) < 2:
                    continue
                asking, real, other = 256 + j, j + 1, k + 1        # the token at `asking` is A = random_ids[row, j]
                found["before"].append((normal[asking, real].item(), normal[asking, other].item()))
                for label, (a, b) in (("swapped", (real, other)), ("control", (other, third + 1))):
                    answering = x[row:row + 1].clone()
                    answering[0, a] += the_note[row, b] - the_note[row, a]
                    answering[0, b] += the_note[row, a] - the_note[row, b]
                    weights = attention(model.blocks[block], x[row:row + 1], answering)[0, head, asking]
                    found[label].append((weights[real].item(), weights[other].item()))
        means = {label: np.mean(values, axis=0) for label, values in found.items()}
        print(f"  {block}.{head:<4d} | {means['before'][0]:.3f}, {means['before'][1]:.3f}         | "
              f"{means['swapped'][0]:.3f}, {means['swapped'][1]:.3f}                     | "
              f"{means['control'][0]:.3f}, {means['control'][1]:.3f}")
    print(f"  ({len(found['before'])} pairs per head)")


def main(args):
    device = pick_device()
    model = load_checkpoint(args.model, device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    val = read_shard(shard_paths(args.text, "val")[0])
    text = torch.tensor(np.stack([val[i * 1024:(i + 1) * 1024] for i in range(8)]).astype(np.int64), device=device)
    print(f"model: {args.model} | real text: {args.text}, 8 windows of 1,024 tokens | on {device}")

    guess = read_the_note(model, device, text)
    print(f"\n  example, the first tokens of the real text: each token, and what head {NOTE_HEAD[0]}.{NOTE_HEAD[1]}'s "
          f"note at the next position says came before it")
    for actual, read in zip(text[0, :15].tolist(), guess.tolist()):
        print(f"    {tokenizer.decode([actual])!r:16s} -> note reads {tokenizer.decode([read])!r}")
    wiring(model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test how copying is wired in a finished model.")
    parser.add_argument("model", help="a checkpoint, e.g. src/checkpoints/gpt2-small-snapshots/best.pt")
    parser.add_argument("--text", required=True, help="a folder prepare_data.py wrote, for real text")
    main(parser.parse_args())
