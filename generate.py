#!/usr/bin/env python3

"""

Step 8: generate completions from a trained checkpoint.


Two modes:


  interactive  one prompt from the command line, printed to screen

  batch        a prompts file, every completion written to JSON for scoring


Each generation is scored automatically for:

  parses        does ast.parse accept it (after truncating to the last

                complete-looking line)

  uses_params   does a generated function actually reference its own arguments

                -- the previous project's models routinely ignored them


These two numbers are what Phase 2 and Phase 3 are judged against, so they are

computed here rather than by eye.


Usage:

  python generate.py --prompt "def binary_search(arr, target):"

  python generate.py --prompts-file prompts.txt --out out/baseline/generations.json

"""


import argparse

import ast

import json

import os

import sys


import torch

import sentencepiece as spm


from model import ModelConfig, Transformer



def load_model(ckpt_path, device):

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ModelConfig(**ckpt["config"])

    model = Transformer(cfg).to(device)

    model.load_state_dict(ckpt["model"])

    model.eval()

    return model, cfg, ckpt



def truncate_to_parseable(text: str):

    """Trim trailing lines until the snippet parses, or give up.


    A generation that is cut off mid-statement is not an interesting failure --

    it just ran out of budget. Trimming separates 'ran out of tokens' from

    'produced something genuinely malformed'.

    """

    lines = text.split("\n")

    for end in range(len(lines), 0, -1):

        candidate = "\n".join(lines[:end]).rstrip()

        if not candidate:

            continue

        try:

            ast.parse(candidate)

            return candidate, True

        except (SyntaxError, ValueError):

            continue

    return text, False



def uses_own_params(code: str):

    """True if some function in the snippet references its non-self arguments.


    Returns None when there is no function to judge.

    """

    try:

        tree = ast.parse(code)

    except (SyntaxError, ValueError):

        return None


    results = []

    for node in ast.walk(tree):

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            continue

        params = {

            a.arg

            for a in list(node.args.args) + list(node.args.kwonlyargs)

            if a.arg not in ("self", "cls")

        }

        if not params:

            continue

        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

        results.append(bool(params & used))


    if not results:

        return None

    return all(results)



@torch.no_grad()

def generate_one(model, sp, prompt, device, max_new_tokens, temperature, top_k, seed=None):

    if seed is not None:

        torch.manual_seed(seed)

    ids = [sp.bos_id()] + sp.encode(prompt)

    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    full = sp.decode([int(i) for i in out[0].tolist() if i != sp.bos_id()])

    completion = full[len(prompt) :] if full.startswith(prompt) else full

    return full, completion



def score(full_text):

    trimmed, parses = truncate_to_parseable(full_text)

    return {

        "parses": parses,

        "uses_params": uses_own_params(trimmed) if parses else None,

        "trimmed_chars": len(trimmed),

        "trimmed": trimmed,

    }



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", default="out/baseline/ckpt.pt")

    ap.add_argument("--tokenizer", default="tokenizer/code_bpe.model")

    ap.add_argument("--prompt", default=None)

    ap.add_argument("--prompts-file", default=None)

    ap.add_argument("--out", default=None)

    ap.add_argument("--max-new-tokens", type=int, default=200)

    ap.add_argument("--temperature", type=float, default=0.8)

    ap.add_argument("--top-k", type=int, default=200)

    ap.add_argument(

        "--samples",

        type=int,

        default=1,

        help="stochastic samples per prompt; a greedy sample is always added too",

    )

    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg, ckpt = load_model(args.ckpt, device)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)


    print(

        f"loaded {args.ckpt}  step {ckpt['step']}  val_loss {ckpt['val_loss']:.4f}",

        file=sys.stderr,

    )


    if args.prompt:

        prompts = [args.prompt]

    elif args.prompts_file:

        with open(args.prompts_file, encoding="utf-8") as f:

            # Blank-line separated, so prompts can span several lines.

            prompts = [p.strip("\n") for p in f.read().split("\n\n") if p.strip()]

    else:

        print("need --prompt or --prompts-file", file=sys.stderr)

        return 1


    records = []

    for pi, prompt in enumerate(prompts):

        for si in range(args.samples):

            full, comp = generate_one(

                model, sp, prompt, device, args.max_new_tokens,

                args.temperature, args.top_k, seed=args.seed + pi * 100 + si,

            )

            records.append(

                {

                    "prompt_index": pi,

                    "prompt": prompt,

                    "decoding": f"temp={args.temperature},top_k={args.top_k}",

                    "sample": si,

                    "completion": comp,

                    "full": full,

                    **score(full),

                }

            )

        # Greedy: deterministic, and the fairest single view of what the model

        # considers most likely.

        full, comp = generate_one(

            model, sp, prompt, device, args.max_new_tokens, 0.0, None

        )

        records.append(

            {

                "prompt_index": pi,

                "prompt": prompt,

                "decoding": "greedy",

                "sample": -1,

                "completion": comp,

                "full": full,

                **score(full),

            }

        )


    parse_rate = sum(r["parses"] for r in records) / len(records)

    judged = [r["uses_params"] for r in records if r["uses_params"] is not None]

    summary = {

        "ckpt": args.ckpt,

        "ckpt_step": ckpt["step"],

        "ckpt_val_loss": ckpt["val_loss"],

        "n_prompts": len(prompts),

        "n_generations": len(records),

        "parse_rate": round(parse_rate, 4),

        "param_use_rate": round(sum(judged) / len(judged), 4) if judged else None,

        "param_use_judged_on": len(judged),

        "max_new_tokens": args.max_new_tokens,

    }


    if args.out:

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

        with open(args.out, "w", encoding="utf-8") as f:

            json.dump({"summary": summary, "generations": records}, f, indent=2)

        print(json.dumps(summary, indent=2))

        print(f"\nwrote {args.out}")

    else:

        for r in records:

            print("=" * 70)

            print(f"[{r['decoding']}] parses={r['parses']} uses_params={r['uses_params']}")

            print("-" * 70)

            print(r["full"])

        print("=" * 70)

        print(json.dumps(summary, indent=2))


    return 0



if __name__ == "__main__":

    sys.exit(main())
