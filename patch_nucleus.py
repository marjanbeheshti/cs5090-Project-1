#!/usr/bin/env python3

"""

Add nucleus (top-p) sampling to model.py and expose it in generate.py.



Nucleus sampling keeps the smallest set of tokens whose cumulative probability

reaches p, so the candidate pool resizes with the model's confidence instead of

being fixed at k. Holtzman et al. (ICLR 2020) show a constant k is suboptimal:

small k produces bland repetitive text, large k admits tail candidates whose

probability is inflated by renormalisation.



Run once from the project root:

  python patch_nucleus.py

"""

import re, sys



# ---------- model.py: add top_p to generate() ----------

m = open('model.py').read()

if 'top_p' in m:

    print('model.py already patched')

else:

    old_sig = 'def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):'

    new_sig = 'def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200, top_p=None):'

    if old_sig not in m:

        print('ERROR: could not find generate() signature in model.py'); sys.exit(1)

    m = m.replace(old_sig, new_sig, 1)



    old_blk = """                if top_k is not None:

                    k = min(top_k, logits.size(-1))

                    v, _ = torch.topk(logits, k)

                    logits[logits < v[:, [-1]]] = -float("inf")"""

    new_blk = """                if top_p is not None:

                    # Nucleus: keep the smallest set whose cumulative

                    # probability reaches top_p, then renormalise.

                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)

                    probs_sorted = F.softmax(sorted_logits, dim=-1)

                    cum = torch.cumsum(probs_sorted, dim=-1)

                    # Drop everything after the token that crosses top_p,

                    # but always keep at least one token.

                    remove = cum - probs_sorted > top_p

                    remove[:, 0] = False

                    sorted_logits[remove] = -float("inf")

                    logits = torch.full_like(logits, -float("inf")).scatter(

                        1, sorted_idx, sorted_logits

                    )

                elif top_k is not None:

                    k = min(top_k, logits.size(-1))

                    v, _ = torch.topk(logits, k)

                    logits[logits < v[:, [-1]]] = -float("inf")"""

    if old_blk not in m:

        print('ERROR: could not find top_k block in model.py'); sys.exit(1)

    m = m.replace(old_blk, new_blk, 1)

    open('model.py','w').write(m)

    print('model.py patched')



# ---------- generate.py: plumb --top-p through ----------

g = open('generate.py').read()

if 'top_p' in g:

    print('generate.py already patched')

else:

    g = g.replace(

        'def generate_one(model, sp, prompt, device, max_new_tokens, temperature, top_k, seed=None):',

        'def generate_one(model, sp, prompt, device, max_new_tokens, temperature, top_k, seed=None, top_p=None):',

        1)

    g = g.replace(

        'out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)',

        'out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)',

        1)

    g = g.replace(

        '    ap.add_argument("--top-k", type=int, default=200)',

        '    ap.add_argument("--top-k", type=int, default=200)\n'

        '    ap.add_argument("--top-p", type=float, default=None,\n'

        '                    help="nucleus sampling; overrides --top-k when set")',

        1)

    g = g.replace(

        """                model, sp, prompt, device, args.max_new_tokens,

                args.temperature, args.top_k, seed=args.seed + pi * 100 + si,

            )""",

        """                model, sp, prompt, device, args.max_new_tokens,

                args.temperature, args.top_k, seed=args.seed + pi * 100 + si,

                top_p=args.top_p,

            )""",

        1)

    g = g.replace(

        '"decoding": f"temp={args.temperature},top_k={args.top_k}",',

        '"decoding": (f"temp={args.temperature},top_p={args.top_p}"\n'

        '                                 if args.top_p else f"temp={args.temperature},top_k={args.top_k}"),',

        1)

    open('generate.py','w').write(g)

    print('generate.py patched')



import ast

ast.parse(open('model.py').read())

ast.parse(open('generate.py').read())

print('both files parse OK')
