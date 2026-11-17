# The order the 16 subjects arrive in, from the run's probes_3.pt (made by probe_snapshots_3.py):
#
#   python subject_order.py <run folder>
#
# For each subject: the step by which half of its questions have been the top guess at least once (the median of
# each question's first top-guess step, the lower one for an even count, with "never" for a question that's never
# the top guess), how many are ever the top guess, and the share that's the top guess after 1,000 steps and at the end.

import argparse
from pathlib import Path

import numpy as np
import torch

NEVER = 10 ** 9


def main(args):
    r = torch.load(Path(args.run_dir) / "probes_3.pt")
    steps, probes = r["steps"], r["probes"]
    top = (r["rank"][:, :, -1] == 0).numpy()
    rows = []
    for category in dict.fromkeys(c for c, *_ in probes):
        idx = [p for p, (c, *_) in enumerate(probes) if c == category]
        firsts = sorted(steps[int(np.argmax(top[:, p]))] if top[:, p].any() else NEVER for p in idx)
        rows.append((firsts[(len(idx) - 1) // 2], category, len(idx), sum(f < NEVER for f in firsts),
                     top[steps.index(1000), idx].mean(), top[-1, idx].mean()))
    print(f"{'subject':>20} | questions | half first right by | ever right | right after 1,000 | right at the end")
    for half, category, n, ever, early, end in sorted(rows):
        print(f"{category:>20} | {n:9} | {'never' if half == NEVER else f'{half:,}':>18} | {ever:4} of {n:<3} | "
              f"{early:17.0%} | {end:16.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="When each subject arrives.")
    parser.add_argument("run_dir", help="the run folder, with probes_3.pt")
    main(parser.parse_args())
