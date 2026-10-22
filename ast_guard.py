#!/usr/bin/env python3

"""

Phase 4, third intervention: a software-layer stopping guard.



The model does not know when a function ends, so it generates to the token

budget and fills the remainder with degenerate text. This script does not change

the model or the decoder. It post-processes each generation by walking the

output line by line and cutting at the first point where exactly one complete

top-level function has been produced.



This is the cheapest possible intervention -- no GPU, no training, no change to

the generator -- and it is applied to EXISTING generations, so the comparison

against the unguarded outputs is exactly controlled: identical model, identical

decoding, identical sampled tokens.



Usage:

  python ast_guard.py --runs out/baseline/phase3_generations.json out/sft2/phase3_generations.json

"""



import argparse

import ast

import json

import sys





def first_complete_function(text: str):

    """Return the shortest prefix of text containing one complete function.



    Walks forward adding one line at a time and checks whether the accumulated

    prefix parses into exactly one top-level function definition. Returns

    (prefix, True) on success, or (text, False) if no complete function is ever

    formed within the generation.

    """

    lines = text.split("\n")

    for end in range(1, len(lines) + 1):

        prefix = "\n".join(lines[:end])

        if not prefix.strip():

            continue

        try:

            tree = ast.parse(prefix)

        except (SyntaxError, ValueError):

            continue



        funcs = [

            n for n in tree.body

            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))

        ]

        if len(funcs) != 1:

            continue

        # The function must have a real body, not just a docstring.

        body = funcs[0].body

        if len(body) < 1:

            continue

        if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):

            continue

        # The prefix must end at the function, with nothing trailing after it.

        if tree.body[-1] is not funcs[0]:

            continue

        return prefix, True



    return text, False





def uses_own_params(code: str):

    try:

        tree = ast.parse(code)

    except (SyntaxError, ValueError):

        return None

    results = []

    for node in ast.walk(tree):

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            continue

        params = {

            a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)

            if a.arg not in ("self", "cls")

        }

        if not params:

            continue

        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

        results.append(bool(params & used))

    if not results:

        return None

    return all(results)





def analyse(path, decoding="greedy"):

    with open(path, encoding="utf-8") as f:

        data = json.load(f)

    recs = [r for r in data["generations"] if (r["decoding"] == "greedy") == (decoding == "greedy")]



    n = len(recs)

    guarded_ok = 0

    parse_before = 0

    parse_after = 0

    chars_before = 0

    chars_after = 0

    pu_before, pu_after = [], []



    for r in recs:

        full = r["full"]

        chars_before += len(full)

        if r["parses"]:

            parse_before += 1

        b = uses_own_params(r.get("trimmed") or full)

        if b is not None:

            pu_before.append(b)



        cut, ok = first_complete_function(full)

        if ok:

            guarded_ok += 1

            parse_after += 1

        chars_after += len(cut)

        a = uses_own_params(cut) if ok else None

        if a is not None:

            pu_after.append(a)



    return {

        "file": path,

        "decoding": decoding,

        "generations": n,

        "guard_found_function": f"{guarded_ok}/{n}",

        "parse_before": f"{parse_before}/{n}",

        "parse_after": f"{parse_after}/{n}",

        "mean_chars_before": round(chars_before / n),

        "mean_chars_after": round(chars_after / n),

        "param_use_before": round(sum(pu_before) / len(pu_before), 3) if pu_before else None,

        "param_use_after": round(sum(pu_after) / len(pu_after), 3) if pu_after else None,

    }





def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--runs", nargs="+", required=True)

    ap.add_argument("--out", default="out/ast_guard_results.json")

    ap.add_argument("--show", type=int, default=2, help="example cuts to print")

    args = ap.parse_args()



    results = []

    for path in args.runs:

        for mode in ("greedy", "sampled"):

            results.append(analyse(path, mode))



    for r in results:

        name = r["file"].split("/")[1]

        print(

            f"{name:10} {r['decoding']:8} "

            f"guard {r['guard_found_function']:>6}  "

            f"chars {r['mean_chars_before']:>4} -> {r['mean_chars_after']:<4}  "

            f"param-use {r['param_use_before']} -> {r['param_use_after']}"

        )



    with open(args.out, "w", encoding="utf-8") as f:

        json.dump(results, f, indent=2)

    print(f"\nwrote {args.out}")



    if args.show:

        with open(args.runs[0], encoding="utf-8") as f:

            data = json.load(f)

        g = [r for r in data["generations"] if r["decoding"] == "greedy"]

        for r in g[: args.show]:

            cut, ok = first_complete_function(r["full"])

            print("\n" + "=" * 64)

            print(f"found={ok}  {len(r['full'])} chars -> {len(cut)} chars")

            print("-" * 64)

            print(cut)



    return 0





if __name__ == "__main__":

    sys.exit(main())
