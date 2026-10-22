#!/usr/bin/env python3

"""

Step 7: train the model.


Sampling note: the corpus is split into non-overlapping blocks of seq_len, the

block ORDER is shuffled under a fixed seed, and blocks are consumed in that

order. One pass therefore sees every token exactly once -- no token is ever

repeated, which is why this run cannot memorise its training data the way the

previous small-corpus project did.


Usage:

  python train.py --run-name baseline

  python train.py --run-name abl_no_rope --no-rope

"""


import argparse

import json

import math

import os

import sys

import time


import numpy as np

import torch


from model import ModelConfig, Transformer



def get_lr(step, warmup, total, lr, min_lr):

    """Linear warmup, then cosine decay to min_lr."""

    if step < warmup:

        return lr * (step + 1) / warmup

    if step >= total:

        return min_lr

    ratio = (step - warmup) / max(1, total - warmup)

    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)



class BlockSampler:

    """Yields shuffled, non-overlapping blocks so one epoch covers everything."""


    def __init__(self, path, seq_len, batch_size, seed, device):

        self.data = np.memmap(path, dtype=np.uint16, mode="r")

        self.seq_len = seq_len

        self.batch_size = batch_size

        self.device = device

        # +1 because targets are inputs shifted by one position.

        self.n_blocks = (len(self.data) - 1) // seq_len

        self.order = np.random.default_rng(seed).permutation(self.n_blocks)

        self.cursor = 0

        self.epochs = 0


    def __len__(self):

        return self.n_blocks


    def next_batch(self):

        idx = []

        for _ in range(self.batch_size):

            if self.cursor >= self.n_blocks:

                self.cursor = 0

                self.epochs += 1

            idx.append(self.order[self.cursor])

            self.cursor += 1


        x = np.stack([self.data[i * self.seq_len : i * self.seq_len + self.seq_len] for i in idx])

        y = np.stack(

            [self.data[i * self.seq_len + 1 : i * self.seq_len + self.seq_len + 1] for i in idx]

        )

        x = torch.from_numpy(x.astype(np.int64))

        y = torch.from_numpy(y.astype(np.int64))

        if self.device.startswith("cuda"):

            x = x.pin_memory().to(self.device, non_blocking=True)

            y = y.pin_memory().to(self.device, non_blocking=True)

        else:

            x, y = x.to(self.device), y.to(self.device)

        return x, y



@torch.no_grad()

def estimate_loss(model, sampler, iters, ctx):

    model.eval()

    losses = torch.zeros(iters)

    for i in range(iters):

        x, y = sampler.next_batch()

        with ctx:

            _, loss = model(x, y)

        losses[i] = loss.item()

    model.train()

    return losses.mean().item()



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--run-name", default="baseline")

    ap.add_argument("--data-dir", default="data")

    ap.add_argument("--out-dir", default="out")


    # architecture

    ap.add_argument("--dim", type=int, default=384)

    ap.add_argument("--n-layers", type=int, default=6)

    ap.add_argument("--n-heads", type=int, default=6)

    ap.add_argument("--seq-len", type=int, default=512)

    ap.add_argument("--vocab-size", type=int, default=8192)

    ap.add_argument("--dropout", type=float, default=0.0)


    # ablation switches

    ap.add_argument("--no-rope", action="store_true", help="learned positions instead of RoPE")

    ap.add_argument("--no-rmsnorm", action="store_true", help="LayerNorm instead of RMSNorm")


    # optimisation

    ap.add_argument("--batch-size", type=int, default=16)

    ap.add_argument("--grad-accum", type=int, default=4)

    ap.add_argument("--lr", type=float, default=6e-4)

    ap.add_argument("--min-lr", type=float, default=6e-5)

    ap.add_argument("--warmup", type=int, default=1000)

    ap.add_argument("--max-steps", type=int, default=0, help="0 = exactly one epoch")

    ap.add_argument("--weight-decay", type=float, default=0.1)

    ap.add_argument("--grad-clip", type=float, default=1.0)


    # logging / eval

    ap.add_argument("--eval-every", type=int, default=250)

    ap.add_argument("--eval-iters", type=int, default=50)

    ap.add_argument("--log-every", type=int, default=25)

    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--compile", action="store_true", help="torch.compile (slow first step)")

    args = ap.parse_args()


    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True


    device = "cuda" if torch.cuda.is_available() else "cpu"

    bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported()

    dtype = torch.bfloat16 if bf16_ok else torch.float32

    ctx = (

        torch.amp.autocast(device_type="cuda", dtype=dtype)

        if device == "cuda"

        else torch.amp.autocast(device_type="cpu", enabled=False)

    )


    run_dir = os.path.join(args.out_dir, args.run_name)

    os.makedirs(run_dir, exist_ok=True)


    train_sampler = BlockSampler(

        os.path.join(args.data_dir, "train.bin"), args.seq_len, args.batch_size, args.seed, device

    )

    val_sampler = BlockSampler(

        os.path.join(args.data_dir, "val.bin"), args.seq_len, args.batch_size, args.seed + 1, device

    )


    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len

    one_epoch_steps = len(train_sampler) // (args.batch_size * args.grad_accum)

    total_steps = args.max_steps or one_epoch_steps


    cfg = ModelConfig(

        vocab_size=args.vocab_size,

        dim=args.dim,

        n_layers=args.n_layers,

        n_heads=args.n_heads,

        seq_len=args.seq_len,

        dropout=args.dropout,

        use_rope=not args.no_rope,

        use_rmsnorm=not args.no_rmsnorm,

    )

    model = Transformer(cfg).to(device)


    n_params = model.num_params()

    meta = {

        "run_name": args.run_name,

        "config": vars(args),

        "device": device,

        "dtype": str(dtype),

        "params_total": n_params,

        "params_non_embedding": model.num_params(non_embedding=True),

        "train_tokens": int(len(train_sampler) * args.seq_len),

        "tokens_per_step": tokens_per_step,

        "one_epoch_steps": one_epoch_steps,

        "total_steps": total_steps,

        "tokens_per_param": round(len(train_sampler) * args.seq_len / n_params, 2),

    }

    print(json.dumps(meta, indent=2))

    with open(os.path.join(run_dir, "config.json"), "w") as f:

        json.dump(meta, f, indent=2)


    opt = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95), device)


    if args.compile:

        model = torch.compile(model)


    log = []

    best_val = float("inf")

    t0 = time.time()

    model.train()


    for step in range(total_steps):

        lr = get_lr(step, args.warmup, total_steps, args.lr, args.min_lr)

        for g in opt.param_groups:

            g["lr"] = lr


        opt.zero_grad(set_to_none=True)

        running = 0.0

        for _ in range(args.grad_accum):

            x, y = train_sampler.next_batch()

            with ctx:

                _, loss = model(x, y)

                loss = loss / args.grad_accum

            loss.backward()

            running += loss.item()


        if args.grad_clip > 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()


        if step % args.log_every == 0:

            el = time.time() - t0

            done = step + 1

            eta = el / done * (total_steps - done)

            print(

                f"step {step}/{total_steps} loss {running:.4f} lr {lr:.2e} "

                f"{el/done*1000:.0f}ms/step eta {eta/3600:.2f}h",

                flush=True,

            )


        if step % args.eval_every == 0 or step == total_steps - 1:

            val = estimate_loss(model, val_sampler, args.eval_iters, ctx)

            entry = {

                "step": step,

                "train_loss": running,

                "val_loss": val,

                "lr": lr,

                "tokens_seen": (step + 1) * tokens_per_step,

                "elapsed_s": round(time.time() - t0, 1),

            }

            log.append(entry)

            print(f"  EVAL step {step}: train {running:.4f} val {val:.4f}", flush=True)


            with open(os.path.join(run_dir, "training_log.json"), "w") as f:

                json.dump(log, f, indent=2)


            if val < best_val:

                best_val = val

                torch.save(

                    {

                        "model": (

                            model._orig_mod.state_dict()

                            if hasattr(model, "_orig_mod")

                            else model.state_dict()

                        ),

                        "config": cfg.__dict__,

                        "step": step,

                        "val_loss": val,

                        "meta": meta,

                    },

                    os.path.join(run_dir, "ckpt.pt"),

                )

                print(f"  saved checkpoint (best val {val:.4f})", flush=True)


    total_h = (time.time() - t0) / 3600

    print(f"\ndone in {total_h:.2f}h, best val {best_val:.4f}")

    print(f"epochs completed: {train_sampler.epochs + train_sampler.cursor/len(train_sampler):.3f}")

    return 0



if __name__ == "__main__":

    sys.exit(main())
