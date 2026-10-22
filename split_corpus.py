#!/usr/bin/env python3

"""

Step 4: split the corpus into a frozen train/val set.


Holds out whole REPOSITORIES rather than random files, because files inside one

repo share idioms and are sometimes literal copies of each other. A random

file-level split leaks that similarity into validation and makes val loss look

better than it is.


After the repo split, any train file whose normalized hash also appears in val

is dropped, which catches copies that crossed repo boundaries.


Outputs:

  data/train.jsonl, data/val.jsonl     files + provenance

  data/train.txt,   data/val.txt       plain text for the tokenizer

  data/split_stats.json                counts (for the report)


Usage:

  python split_corpus.py --val-tokens 2000000

"""


import argparse

import json

import os

import random

import sys

from collections import defaultdict


CHARS_PER_TOKEN = 3.5


# Written between files so the model sees a document boundary instead of one

# file running straight into the next.

DOC_SEP = "\n\n<|file|>\n\n"



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--corpus", default="data/corpus.jsonl")

    ap.add_argument("--out-dir", default="data")

    ap.add_argument("--val-tokens", type=int, default=2_000_000)

    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()


    os.makedirs(args.out_dir, exist_ok=True)


    # Group file offsets by repo. Store the rows themselves; 1.2 GB of text is

    # large but this box has 60 GB of RAM.

    by_repo = defaultdict(list)

    total_files = 0

    with open(args.corpus, encoding="utf-8") as f:

        for line in f:

            row = json.loads(line)

            by_repo[row.get("repo_name") or "<unknown>"].append(row)

            total_files += 1


    repos = sorted(by_repo.keys())

    random.Random(args.seed).shuffle(repos)


    val_target_chars = int(args.val_tokens * CHARS_PER_TOKEN)


    val_repos = []

    val_chars = 0

    for repo in repos:

        if val_chars >= val_target_chars:

            break

        val_repos.append(repo)

        val_chars += sum(r["chars"] for r in by_repo[repo])


    val_repo_set = set(val_repos)


    val_rows = [r for repo in val_repos for r in by_repo[repo]]

    val_hashes = {r["norm_hash"] for r in val_rows}


    train_rows = []

    dropped_leak = 0

    for repo in repos:

        if repo in val_repo_set:

            continue

        for r in by_repo[repo]:

            if r["norm_hash"] in val_hashes:

                dropped_leak += 1

                continue

            train_rows.append(r)


    def write(rows, name):

        jsonl_path = os.path.join(args.out_dir, f"{name}.jsonl")

        txt_path = os.path.join(args.out_dir, f"{name}.txt")

        chars = 0

        with open(jsonl_path, "w", encoding="utf-8") as jf, open(

            txt_path, "w", encoding="utf-8"

        ) as tf:

            for r in rows:

                jf.write(json.dumps(r) + "\n")

                tf.write(r["content"])

                tf.write(DOC_SEP)

                chars += r["chars"]

        return chars


    train_chars = write(train_rows, "train")

    val_chars_final = write(val_rows, "val")


    stats = {

        "seed": args.seed,

        "split_unit": "repository",

        "total_files_in": total_files,

        "total_repos": len(repos),

        "train_repos": len(repos) - len(val_repos),

        "val_repos": len(val_repos),

        "train_files": len(train_rows),

        "val_files": len(val_rows),

        "train_chars": train_chars,

        "val_chars": val_chars_final,

        "train_tokens_est": int(train_chars / CHARS_PER_TOKEN),

        "val_tokens_est": int(val_chars_final / CHARS_PER_TOKEN),

        "train_files_dropped_as_val_duplicates": dropped_leak,

        "doc_separator": DOC_SEP,

    }


    with open(os.path.join(args.out_dir, "split_stats.json"), "w", encoding="utf-8") as f:

        json.dump(stats, f, indent=2)


    print(json.dumps(stats, indent=2))


    # Hard guarantee, checked rather than assumed.

    assert not ({r["norm_hash"] for r in train_rows} & val_hashes), "train/val overlap"

    print("\nleakage check passed: no shared normalized hashes")

    return 0



if __name__ == "__main__":

    sys.exit(main())
