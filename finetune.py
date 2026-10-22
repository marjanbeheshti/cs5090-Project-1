#!/usr/bin/env python3

"""

Phase 4, step 2: fine-tune the baseline to stop at the end of a function.



Each example is encoded as



    [BOS] signature + docstring + body [EOS] [pad ...]



with the loss masked (-1) over the padding and, by default, over the prompt as

well, so the gradient only rewards producing the body and then emitting EOS.

That EOS token is the entire point: pre-training never presented a

"stop here" signal, because the corpus was a continuous stream cut into

512-token blocks.



Only the data format changes. Architecture, tokenizer, context length and

parameter count are identical to the baseline, so before/after comparison is

controlled.



Outputs:

  out/sft/ckpt.pt, out/sft/training_log.json, out/sft/config.json



Usage:

  python finetune.py --init out/baseline/ckpt.pt --epochs 2

"""



import argparse

import json

import math

import os

import random

import sys

import time



import numpy as np

import sentencepiece as spm

import torch



from model import ModelConfig, Transformer





def build_examples(path, sp, seq_len, mask_prompt, limit=0):

    """Encode pairs into fixed-length id/label arrays."""

    bos, eos = sp.bos_id(), sp.eos_id()

    xs, ys = [], []

    skipped = 0



    with open(path, encoding="utf-8") as f:

        for i, line in enumerate(f):

            if limit and len(xs) >= limit:

                break

            row = json.loads(line)

            p_ids = sp.encode(row["prompt"] + "\n")

            t_ids = sp.encode(row["target"])



            ids = [bos] + p_ids + t_ids + [eos]

            if len(ids) > seq_len + 1:

                skipped += 1

                continue



            # labels[i] is the target for position i, i.e. ids[i+1]

            inp = ids[:-1]

            lab = ids[1:]



            if mask_prompt:

                # Positions whose target still lies inside the prompt teach

                # nothing about generating a body, so they are masked out.

                n_prompt = len(p_ids)

                lab = [-1] * n_prompt + lab[n_prompt:]



            pad = seq_len - len(inp)

            inp = inp + [0] * pad

            lab = lab + [-1] * pad



            xs.append(inp)

            ys.append(lab)



    return (

        np.array(xs, dtype=np.int64),

        np.array(ys, dtype=np.int64),

        skipped,

    )





def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--init", default="out/baseline/ckpt.pt")

    ap.add_argument("--sft", default="data/sft.jsonl")

    ap.add_argument("--tokenizer", default="tokenizer/code_bpe.model")

    ap.add_argument("--run-name", default="sft")

    ap.add_argument("--out-dir", default="out")

    ap.add_argument("--epochs", type=int, default=2)

    ap.add_argument("--batch-size", type=int, default=16)

    ap.add_argument("--grad-accum", type=int, default=2)

    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--min-lr", type=float, default=1e-5)

    ap.add_argument("--warmup", type=int, default=100)

    ap.add_argument("--weight-decay", type=float, default=0.1)

    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--val-frac", type=float, default=0.02)

    ap.add_argument("--eval-every", type=int, default=100)

    ap.add_argument("--log-every", type=int, default=25)

    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--no-mask-prompt", action="store_true")

    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()



    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True



    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    ctx = (

        torch.amp.autocast(device_type="cuda", dtype=dtype)

        if device == "cuda"

        else torch.amp.autocast(device_type="cpu", enabled=False)

    )



    run_dir = os.path.join(args.out_dir, args.run_name)

    os.makedirs(run_dir, exist_ok=True)



    ckpt = torch.load(args.init, map_location=device, weights_only=False)

    cfg = ModelConfig(**ckpt["config"])

    model = Transformer(cfg).to(device)

    model.load_state_dict(ckpt["model"])

    print(f"loaded {args.init}  step {ckpt['step']}  val_loss {ckpt['val_loss']:.4f}")



    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)



    X, Y, skipped = build_examples(

        args.sft, sp, cfg.seq_len, not args.no_mask_prompt, args.limit

    )

    n = len(X)

    idx = np.random.default_rng(args.seed).permutation(n)

    n_val = max(64, int(n * args.val_frac))

    val_idx, train_idx = idx[:n_val], idx[n_val:]



    tokens_per_step = args.batch_size * args.grad_accum

    steps_per_epoch = len(train_idx) // tokens_per_step

    total_steps = steps_per_epoch * args.epochs



    meta = {

        "init": args.init,

        "init_val_loss": ckpt["val_loss"],

        "examples_total": n,

        "examples_train": len(train_idx),

        "examples_val": n_val,

        "examples_skipped_too_long": skipped,

        "mask_prompt": not args.no_mask_prompt,

        "steps_per_epoch": steps_per_epoch,

        "total_steps": total_steps,

        "config": vars(args),

    }

    print(json.dumps(meta, indent=2))

    with open(os.path.join(run_dir, "config.json"), "w") as f:

        json.dump(meta, f, indent=2)



    if total_steps == 0:

        print("not enough examples to make a single step")

        return 1



    opt = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95), device)



    def get_batch(pool, size):

        pick = np.random.choice(pool, size, replace=False)

        x = torch.from_numpy(X[pick]).to(device)

        y = torch.from_numpy(Y[pick]).to(device)

        return x, y



    @torch.no_grad()

    def eval_loss(iters=20):

        model.eval()

        tot = 0.0

        for _ in range(iters):

            x, y = get_batch(val_idx, min(args.batch_size, len(val_idx)))

            with ctx:

                _, loss = model(x, y)

            tot += loss.item()

        model.train()

        return tot / iters



    log = []

    best = float("inf")

    t0 = time.time()

    model.train()



    for step in range(total_steps):

        if step < args.warmup:

            lr = args.lr * (step + 1) / args.warmup

        else:

            r = (step - args.warmup) / max(1, total_steps - args.warmup)

            lr = args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)

        for g in opt.param_groups:

            g["lr"] = lr



        opt.zero_grad(set_to_none=True)

        running = 0.0

        for _ in range(args.grad_accum):

            x, y = get_batch(train_idx, args.batch_size)

            with ctx:

                _, loss = model(x, y)

                loss = loss / args.grad_accum

            loss.backward()

            running += loss.item()



        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()



        if step % args.log_every == 0:

            el = time.time() - t0

            print(

                f"step {step}/{total_steps} loss {running:.4f} lr {lr:.2e} "

                f"{el/(step+1)*1000:.0f}ms/step",

                flush=True,

            )



        if step % args.eval_every == 0 or step == total_steps - 1:

            v = eval_loss()

            log.append({"step": step, "train_loss": running, "val_loss": v,

                        "elapsed_s": round(time.time() - t0, 1)})

            print(f"  EVAL step {step}: train {running:.4f} sft_val {v:.4f}", flush=True)

            with open(os.path.join(run_dir, "training_log.json"), "w") as f:

                json.dump(log, f, indent=2)

            if v < best:

                best = v

                torch.save(

                    {"model": model.state_dict(), "config": cfg.__dict__,

                     "step": step, "val_loss": v, "meta": meta},

                    os.path.join(run_dir, "ckpt.pt"),

                )

                print(f"  saved checkpoint (best sft val {v:.4f})", flush=True)



    print(f"\ndone in {(time.time()-t0)/60:.1f} min, best sft val {best:.4f}")

    print("NOTE: this loss is on the fine-tuning distribution and is NOT")

    print("comparable to the pre-training validation loss of 1.0773.")

    return 0





if __name__ == "__main__":

    sys.exit(main())
