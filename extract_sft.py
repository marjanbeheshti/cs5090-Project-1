#!/usr/bin/env python3

"""

Phase 4, step 1: extract (signature + docstring -> body) pairs.



Pulls every function from the TRAINING split that has both a docstring and a

real body, and splits it at the end of the docstring. The prompt half matches

the shape of the Phase 2 and Phase 3 probes exactly; the target half is the

real implementation, which will be followed by an EOS token during fine-tuning.



Training files are used rather than validation files on purpose: validation must

stay untouched so that validation loss remains a clean held-out measurement.

The model has already seen this text during pre-training, which is acceptable

because the intervention teaches a stopping behaviour and an output format, not

new knowledge.



Outputs:

  data/sft.jsonl        prompt/target pairs

  data/sft_stats.json   extraction counts



Usage:

  python extract_sft.py --max-examples 20000

"""



import argparse

import ast

import hashlib

import json

import os

import sys

from collections import Counter





def iter_functions(source: str):

    """Yield (prompt, body) for every documented function with a real body."""

    try:

        tree = ast.parse(source)

    except (SyntaxError, ValueError, MemoryError, RecursionError):

        return



    lines = source.split("\n")



    for node in ast.walk(tree):

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            continue

        if not node.body:

            continue



        first = node.body[0]

        # The docstring must be the first statement.

        if not (

            isinstance(first, ast.Expr)

            and isinstance(first.value, ast.Constant)

            and isinstance(first.value.value, str)

        ):

            continue



        # A docstring alone is not a body; there must be code after it.

        if len(node.body) < 2:

            continue



        start = node.lineno - 1          # def line, 0-indexed

        doc_end = first.end_lineno       # last line of the docstring, 1-indexed

        end = node.end_lineno



        # Skip decorated functions: the decorator sits above node.lineno and

        # would be dropped, producing a prompt that misrepresents the function.

        if node.decorator_list:

            continue

        if node.col_offset != 0:
            continue



        prompt = "\n".join(lines[start:doc_end])

        body = "\n".join(lines[doc_end:end])



        if not body.strip():

            continue



        yield prompt, body





def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--corpus", default="data/train.jsonl")

    ap.add_argument("--out", default="data/sft.jsonl")

    ap.add_argument("--max-examples", type=int, default=20000)

    ap.add_argument("--min-body-chars", type=int, default=30)

    ap.add_argument("--max-body-chars", type=int, default=1200)

    ap.add_argument("--max-prompt-chars", type=int, default=600)

    ap.add_argument("--max-files", type=int, default=0, help="0 = all")

    args = ap.parse_args()



    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)



    seen = set()

    stats = Counter()

    kept = 0



    with open(args.corpus, encoding="utf-8") as fin, open(

        args.out, "w", encoding="utf-8"

    ) as fout:

        for i, line in enumerate(fin):

            if args.max_files and i >= args.max_files:

                break

            if kept >= args.max_examples:

                stats["stopped_on_cap"] = 1

                break



            row = json.loads(line)

            stats["files_scanned"] += 1



            for prompt, body in iter_functions(row["content"]):

                stats["candidates"] += 1



                if len(prompt) > args.max_prompt_chars:

                    stats["drop_prompt_long"] += 1

                    continue

                if len(body) < args.min_body_chars:

                    stats["drop_body_short"] += 1

                    continue

                if len(body) > args.max_body_chars:

                    stats["drop_body_long"] += 1

                    continue



                # The target must itself be valid Python once dedented, or the

                # model is being taught to produce something unparseable.

                h = hashlib.sha256((prompt + body).encode("utf-8")).hexdigest()

                if h in seen:

                    stats["drop_duplicate"] += 1

                    continue

                seen.add(h)



                fout.write(

                    json.dumps(

                        {

                            "prompt": prompt,

                            "target": body,

                            "repo_name": row.get("repo_name"),

                            "path": row.get("path"),

                        }

                    )

                    + "\n"

                )

                kept += 1

                stats["kept"] += 1



                if kept >= args.max_examples:

                    break



    summary = {

        "source": args.corpus,

        "examples": kept,

        "counts": dict(stats),

        "min_body_chars": args.min_body_chars,

        "max_body_chars": args.max_body_chars,

        "max_prompt_chars": args.max_prompt_chars,

    }

    with open("data/sft_stats.json", "w", encoding="utf-8") as f:

        json.dump(summary, f, indent=2)



    print(json.dumps(summary, indent=2))



    # Show one example so the split point can be eyeballed before training.

    if kept:

        with open(args.out, encoding="utf-8") as f:

            ex = json.loads(f.readline())

        print("\n--- example prompt ---")

        print(ex["prompt"])

        print("--- example target ---")

        print(ex["target"])



    return 0





if __name__ == "__main__":

    sys.exit(main())
