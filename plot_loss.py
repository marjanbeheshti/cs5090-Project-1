#!/usr/bin/env python3

"""

Plot training/validation loss curves.


One run:

  python plot_loss.py --runs baseline


Baseline against an ablation (same axes, for the report):

  python plot_loss.py --runs baseline abl_no_pos --out figures/ablation.png


Writes a PNG and prints the best validation loss per run.

"""


import argparse

import json

import os

import sys


import matplotlib


matplotlib.use("Agg")  # headless server, no display

import matplotlib.pyplot as plt



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--runs", nargs="+", default=["baseline"])

    ap.add_argument("--out-dir", default="out")

    ap.add_argument("--out", default="figures/loss_curve.png")

    ap.add_argument("--x", choices=["step", "tokens"], default="step")

    ap.add_argument("--skip-warmup", action="store_true",

                    help="start the x-axis after step 1000 so the curve is readable")

    args = ap.parse_args()


    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)


    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    summary = {}


    for i, run in enumerate(args.runs):

        path = os.path.join(args.out_dir, run, "training_log.json")

        if not os.path.exists(path):

            print(f"missing {path}, skipping", file=sys.stderr)

            continue

        with open(path) as f:

            log = json.load(f)


        if args.skip_warmup:

            log = [e for e in log if e["step"] >= 1000]

        if not log:

            continue


        xs = [e["step"] if args.x == "step" else e["tokens_seen"] / 1e6 for e in log]

        c = colors[i % len(colors)]


        ax.plot(xs, [e["train_loss"] for e in log], color=c, alpha=0.35,

                linewidth=1, label=f"{run} train")

        ax.plot(xs, [e["val_loss"] for e in log], color=c, linewidth=2,

                label=f"{run} val")


        best = min(log, key=lambda e: e["val_loss"])

        summary[run] = {"best_val": best["val_loss"], "best_step": best["step"],

                        "final_val": log[-1]["val_loss"]}

        ax.scatter([best["step"] if args.x == "step" else best["tokens_seen"] / 1e6],

                   [best["val_loss"]], color=c, zorder=5, s=30)


    ax.set_xlabel("step" if args.x == "step" else "tokens seen (millions)")

    ax.set_ylabel("cross-entropy loss")

    ax.set_title("Training and validation loss")

    ax.grid(alpha=0.3)

    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(args.out, dpi=160)


    print(json.dumps(summary, indent=2))

    print(f"\nwrote {args.out}")

    return 0



if __name__ == "__main__":

    sys.exit(main())
