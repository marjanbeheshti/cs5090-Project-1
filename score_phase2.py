#!/usr/bin/env python3

"""

Phase 2 scoring: accept / accept-after-editing / reject, one prompt at a time.


Shows the greedy completion for each prompt (deterministic, so the judgement is

reproducible), records a verdict and an optional note, and writes both a CSV for

the repository and a summary for the report.


Progress is saved after every judgement, so the session can be interrupted and

resumed without losing work.


Keys:

  a  accept as-is          -- correct, would use unchanged

  e  accept after editing  -- right approach, small fix needed

  r  reject                -- wrong, or not an implementation at all

  s  skip (decide later)

  q  save and quit


Usage:

  python score_phase2.py

  python score_phase2.py --summary        # print results without re-scoring

"""


import argparse

import csv

import json

import os

import sys

from collections import Counter



def load_generations(path, decoding="greedy"):

    with open(path, encoding="utf-8") as f:

        data = json.load(f)

    recs = [r for r in data["generations"] if r["decoding"] == decoding]

    recs.sort(key=lambda r: r["prompt_index"])

    return data["summary"], recs



def load_progress(path):

    if not os.path.exists(path):

        return {}

    with open(path, encoding="utf-8") as f:

        return {int(k): v for k, v in json.load(f).items()}



def save_progress(path, verdicts):

    with open(path, "w", encoding="utf-8") as f:

        json.dump({str(k): v for k, v in verdicts.items()}, f, indent=2)



VERDICTS = {"a": "accept", "e": "edit", "r": "reject"}



def summarise(recs, verdicts):

    counts = Counter(v["verdict"] for v in verdicts.values())

    n = sum(counts.values())

    by_index = {r["prompt_index"]: r for r in recs}


    print("\n" + "=" * 60)

    print("PHASE 2 RESULTS")

    print("=" * 60)

    for key in ("accept", "edit", "reject"):

        c = counts.get(key, 0)

        pct = f"{100*c/n:.0f}%" if n else "--"

        print(f"  {key:<8} {c:>3}   {pct:>5}")

    print(f"  {'total':<8} {n:>3}")


    if n:

        parse = sum(by_index[i]["parses"] for i in verdicts if i in by_index)

        print(f"\n  parses but rejected: "

              f"{sum(1 for i, v in verdicts.items() if v['verdict'] == 'reject' and by_index.get(i, {}).get('parses'))}")

        print(f"  parse rate on judged: {parse}/{n}")

    print("=" * 60)

    return counts



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--generations", default="out/baseline/phase2_generations.json")

    ap.add_argument("--progress", default="out/baseline/phase2_verdicts.json")

    ap.add_argument("--csv", default="out/baseline/phase2_results.csv")

    ap.add_argument("--chars", type=int, default=900, help="completion characters to show")

    ap.add_argument("--summary", action="store_true", help="print results, do not score")

    args = ap.parse_args()


    summary, recs = load_generations(args.generations)

    verdicts = load_progress(args.progress)


    if args.summary:

        summarise(recs, verdicts)

        return 0


    print(f"{len(recs)} prompts. {len(verdicts)} already judged.")

    print("Keys: [a]ccept  [e]dit  [r]eject  [s]kip  [q]uit\n")


    for rec in recs:

        i = rec["prompt_index"]

        if i in verdicts:

            continue


        print("\n" + "=" * 70)

        print(f"PROMPT {i + 1}/{len(recs)}   parses={rec['parses']}  uses_params={rec['uses_params']}")

        print("=" * 70)

        print(rec["full"][: args.chars])

        if len(rec["full"]) > args.chars:

            print(f"\n... [{len(rec['full']) - args.chars} more characters]")

        print("-" * 70)


        while True:

            choice = input("verdict [a/e/r/s/q]: ").strip().lower()

            if choice == "q":

                save_progress(args.progress, verdicts)

                print(f"\nsaved {len(verdicts)} verdicts to {args.progress}")

                return 0

            if choice == "s":

                break

            if choice in VERDICTS:

                note = input("note (optional, why): ").strip()

                verdicts[i] = {"verdict": VERDICTS[choice], "note": note}

                save_progress(args.progress, verdicts)

                break

            print("  use a, e, r, s or q")


    counts = summarise(recs, verdicts)


    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    with open(args.csv, "w", newline="", encoding="utf-8") as f:

        w = csv.writer(f)

        w.writerow(["prompt_index", "prompt", "verdict", "note", "parses",

                    "uses_params", "completion"])

        for rec in recs:

            i = rec["prompt_index"]

            v = verdicts.get(i, {"verdict": "unjudged", "note": ""})

            w.writerow([

                i + 1,

                rec["prompt"].split("\n")[0],

                v["verdict"],

                v["note"],

                rec["parses"],

                rec["uses_params"],

                rec["completion"].replace("\n", "\\n")[:500],

            ])


    with open(args.csv.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:

        json.dump({

            "counts": dict(counts),

            "total_judged": sum(counts.values()),

            "generation_summary": summary,

        }, f, indent=2)


    print(f"\nwrote {args.csv}")

    return 0



if __name__ == "__main__":

    sys.exit(main())
