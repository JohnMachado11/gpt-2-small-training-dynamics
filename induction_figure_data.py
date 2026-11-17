# The numbers behind the induction-head figure, on one sentence, from a finished model: the stream's size and the
# heads' size, head 5.9's note at every word (and how close it is to the value head 5.9 made at the word before),
# what the note at "lamp" reads back as, and how head 6.0 picks its target (its query, every position's key, the
# match scores and the attention they give), with and without head 5.9's note in the key at "lamp".
#
#   python induction_figure_data.py <run folder>/best.pt --out <file>.json
#
# It prints the numbers and saves them as JSON. The note readout uses the
# same recipe as copying_wiring.py: the note head 5.9 writes after each of the 50,257 tokens, each placed once as the
# token before in random context (seeds 5 and 6), compared by cosine similarity.

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from gpt2.checkpoint import load_checkpoint
from copying_wiring import NOTE_HEAD, look_inside, note

SENTENCE = "The code word is velvet lamp. Again, the code word is velvet"
COPY_HEAD = (6, 0)


@torch.no_grad()
def main(args):
    model = load_checkpoint(args.model)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ids = tokenizer.encode(SENTENCE)
    words = [tokenizer.decode([i]) for i in ids]
    asking, target = len(ids) - 1, words.index(" lamp")
    entering, heads = look_inside(model, torch.tensor([ids]))
    block, head = NOTE_HEAD
    size = model.blocks[block].attn.head_dim
    raw = heads[block][0, :, head * size:(head + 1) * size]              # (positions, 64): head 5.9's output
    added = note(model, heads, block, head)[0]                           # (positions, 768): what it adds to the stream

    # head 5.9's value at every word, 64 numbers: what it passes on from a word it attends to
    note_attn = model.blocks[block].attn
    note_qkv = note_attn.qkv(model.blocks[block].norm_before_attn(entering[block][0]))
    values = note_qkv[:, 2 * model.config.embed_dim:][:, head * size:(head + 1) * size]
    stream_at_target = entering[block][0, target]

    # what the note at "lamp" reads back as: the recorded note after every token, as in copying_wiring.py
    vocab, width = model.token_embedding.weight.shape
    rows = -(-vocab // 255)
    placed = torch.randperm(vocab, generator=torch.Generator().manual_seed(5))
    placed = torch.cat([placed, placed[:rows * 255 - vocab]]).view(rows, 255)
    context = torch.randint(0, vocab, (rows, 256), generator=torch.Generator().manual_seed(6))
    context[:, :255] = placed
    recorded = torch.zeros(vocab, width)
    for chunk in range(0, rows, 32):
        _, chunk_heads = look_inside(model, context[chunk:chunk + 32])
        recorded[placed[chunk:chunk + 32].reshape(-1)] = chunk_heads[block][:, 1:].reshape(-1, width)
    table = F.normalize(note(model, {block: recorded}, block, head), dim=-1)
    similar = table @ F.normalize(added[target], dim=-1)
    top = similar.topk(5)
    before = ids[target - 1]
    readout = {"top": [[tokenizer.decode([t]), round(v, 3)] for t, v in zip(top.indices.tolist(), top.values.tolist())],
               "velvet_rank": int((similar > similar[before]).sum()) + 1,
               "velvet_cosine": round(similar[before].item(), 3)}

    # head 6.0: its query at the second "velvet", every position's key, the scores and the attention
    copy_block, copy_head = COPY_HEAD
    attn = model.blocks[copy_block].attn
    part = slice(copy_head * attn.head_dim, (copy_head + 1) * attn.head_dim)

    def query_and_keys(stream):
        qkv = attn.qkv(model.blocks[copy_block].norm_before_attn(stream))
        return qkv[:, :width][:, part], qkv[:, width:2 * width][:, part]

    stream = entering[copy_block][0]
    query, keys = query_and_keys(stream)
    query = query[asking]
    scores = (keys[:asking + 1] @ query) / attn.head_dim ** 0.5
    attention = scores.softmax(dim=0)
    without = stream.clone()
    without[target] -= added[target]                  # head 5.9's note taken out of the stream at "lamp" only
    _, keys_without = query_and_keys(without)
    scores_without = scores.clone()
    scores_without[target] = keys_without[target] @ query / attn.head_dim ** 0.5
    attention_without = scores_without.softmax(dim=0)

    seen = []
    hook = attn.attn_dropout.register_forward_hook(lambda module, inputs, out: seen.append(out[0, copy_head, asking]))
    try:
        model(torch.tensor([ids]))
    finally:
        hook.remove()
    seen5 = []
    hook = model.blocks[block].attn.attn_dropout.register_forward_hook(
        lambda module, inputs, out: seen5.append(out[0, head]))
    try:
        model(torch.tensor([ids]))
    finally:
        hook.remove()

    print(f"sentence: {SENTENCE!r} -> {len(ids)} tokens; head 6.0 asks at {asking} ({words[asking]!r}), its target is "
          f"{target} ({words[target]!r})")
    print(f"check: recomputed attention vs the model's own, head 6.0 at the last token: worst gap "
          f"{(attention - seen[0][:asking + 1]).abs().max():.1e}")
    print(f"the stream holds {width} numbers per token; each of the {model.config.num_heads} heads in a block works "
          f"with {size} ({width} / {model.config.num_heads}); head {block}.{head}'s out_proj slice is {width} x {size}")
    print(f"the stream at {words[target]!r} entering block {block}: {width} numbers, from "
          f"{stream_at_target.min():.2f} to {stream_at_target.max():.2f}")
    print(f"head 5.9's note at {words[target]!r}: 64 numbers, from {raw[target].min():.2f} to {raw[target].max():.2f}")
    near = F.cosine_similarity(raw[target], values, dim=-1)
    print(f"head 5.9's 64-number output at {words[target]!r}, compared with its value at each word (cosine): "
          f"{words[target - 1]!r} {near[target - 1]:.3f}, {words[target]!r} itself {near[target]:.3f}, highest "
          f"elsewhere {near[[p for p in range(len(ids)) if p not in (target - 1, target)]].max():.3f}; its attention "
          f"at {words[target]!r} on {words[target - 1]!r}: {seen5[0][target, target - 1]:.1%}")
    print(f"the note at {words[target]!r} read back (cosine with the note written after each token): top 5 " +
          ", ".join(f"{w!r} {v:.3f}" for w, v in readout["top"]) +
          f"; {words[target - 1]!r} is #{readout['velvet_rank']} at {readout['velvet_cosine']:.3f}")
    print("head 6.0 at the last token: position, word, match score (query . key / 8), attention")
    for p in range(asking + 1):
        print(f"  {p:2d} {words[p]!r:9s} {scores[p]:+7.2f}  {attention[p]:6.1%}")
    print(f"with head 5.9's note taken out of the stream at {words[target]!r} only: its score {scores[target]:+.2f} -> "
          f"{scores_without[target]:+.2f}, its attention {attention[target]:.1%} -> {attention_without[target]:.1%}, "
          f"and on {words[0]!r} {attention[0]:.1%} -> {attention_without[0]:.1%}")
    energy = added[target].pow(2)
    share = energy[head * size:(head + 1) * size].sum() / energy.sum()
    print(f"the note in {words[target]!r}'s stream: {int((added[target] != 0).sum())} of {width} numbers changed, "
          f"the largest by {added[target].abs().max():.3f}; {energy.topk(size).values.sum() / energy.sum():.1%} of its "
          f"size is in its {size} biggest numbers and {share:.1%} in stream numbers {head * size + 1}-"
          f"{(head + 1) * size} (an even spread puts {size / width:.1%} in any {size}); the stream at "
          f"{words[target]!r} has size {stream_at_target.norm():.1f} entering block {block} and "
          f"{stream[target].norm():.1f} entering block {copy_block}, and the note {added[target].norm():.2f}")
    from_note = (keys[target] - keys_without[target]) @ query / attn.head_dim ** 0.5
    print(f"the part of head 6.0's key at {words[target]!r} that the note makes (key with it minus key without "
          f"it), times the query, / 8: {from_note:+.2f}")
    products = query * keys[target]
    print(f"the score at {words[target]!r}, number by number: {int((products > 0).sum())} of 64 products positive, "
          f"adding to {products.sum():+.2f} (/ 8 = {products.sum() / 8:+.2f})")

    data = {"words": [w.strip() or w for w in words], "asking": asking, "target": target,
            "stream": [round(v, 3) for v in stream_at_target.tolist()],
            "note_768": [round(v, 4) for v in added[target].tolist()], "note_slot": [head * size, (head + 1) * size],
            "note_slot_share": round(share.item(), 3),
            "value_before": [round(v, 3) for v in values[target - 1].tolist()],
            "note_vs_value": round(near[target - 1].item(), 3),
            "back": [None] + [round(seen5[0][p, p - 1].item() * 100) for p in range(1, len(ids))],
            "notes": [[round(v, 3) for v in row] for row in raw.tolist()], "readout": readout,
            "query": [round(v, 3) for v in query.tolist()],
            "keys": [[round(v, 3) for v in row] for row in keys[:asking + 1].tolist()],
            "key_without": [round(v, 3) for v in keys_without[target].tolist()],
            "scores": [round(v, 3) for v in scores.tolist()], "attention": [round(v, 4) for v in attention.tolist()],
            "score_without": round(scores_without[target].item(), 3),
            "attention_without": [round(v, 4) for v in attention_without.tolist()]}
    with open(args.out, "w") as file:
        json.dump(data, file)
    print(f"saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The numbers for the induction-head figure.")
    parser.add_argument("model", help="a checkpoint, e.g. src/checkpoints/gpt2-small-snapshots/best.pt")
    parser.add_argument("--out", required=True, help="where to save the JSON")
    main(parser.parse_args())
