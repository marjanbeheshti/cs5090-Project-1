#!/usr/bin/env python3

"""

Step 6: tokenize the corpus into flat uint16 arrays.


Step 5 learned the RULES for turning text into token ids. This step APPLIES

those rules to every file and saves the result, so training never has to

tokenize anything.


Each file is encoded as:  [BOS] <file tokens> [SEP]

BOS marks a fresh document so the model learns where files start, and SEP marks

the boundary so one file does not read as running into the next.


uint16 is enough because the vocabulary is 8,192 (< 65,536), which halves both

disk and memory against int32.


Outputs:

  data/train.bin, data/val.bin

  data/token_stats.json      the TRUE token count and chars/token ratio


Usage:

  python tokenize_data.py

"""


import argparse

import json

import os

import sys

from multiprocessing import Pool


import numpy as np

import sentencepiece as spm

from tqdm import tqdm


_sp = None

_bos = None

_sep = None



def _init(model_path: str, sep_token: str):

    """Each worker process loads its own tokenizer; they cannot be shared."""

    global _sp, _bos, _sep

    _sp = spm.SentencePieceProcessor(model_file=model_path)

    _bos = _sp.bos_id()

    _sep = _sp.piece_to_id(sep_token)



def _encode(content: str):

    ids = _sp.encode(content)

    return np.array([_bos] + ids + [_sep], dtype=np.uint16)



def read_contents(path: str):

    with open(path, encoding="utf-8") as f:

        for line in f:

            yield json.loads(line)["content"]



def tokenize_split(name, jsonl_path, out_path, model_path, sep_token, workers):

    contents = list(read_contents(jsonl_path))

    total_chars = sum(len(c) for c in contents)


    chunks = []

    with Pool(workers, initializer=_init, initargs=(model_path, sep_token)) as pool:

        # imap preserves input order, so the output is reproducible.

        for arr in tqdm(

            pool.imap(_encode, contents, chunksize=64),

            total=len(contents),

            desc=f"tokenize {name}",

            unit="file",

        ):

            chunks.append(arr)


    all_ids = np.concatenate(chunks)

    all_ids.tofile(out_path)


    return {

        "files": len(contents),

        "chars": total_chars,

        "tokens": int(all_ids.size),

        "chars_per_token": round(total_chars / all_ids.size, 3),

        "bytes_on_disk": int(all_ids.size * 2),

        "path": out_path,

    }



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", default="data")

    ap.add_argument("--tokenizer", default="tokenizer/code_bpe.model")

    ap.add_argument("--sep-token", default="<|file|>")

    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))

    args = ap.parse_args()


    stats = {"tokenizer": args.tokenizer, "workers": args.workers, "dtype": "uint16"}


    for name in ("val", "train"):  # val first: it is small and fails fast

        jsonl_path = os.path.join(args.data_dir, f"{name}.jsonl")

        out_path = os.path.join(args.data_dir, f"{name}.bin")

        stats[name] = tokenize_split(

            name, jsonl_path, out_path, args.tokenizer, args.sep_token, args.workers

        )


    # Chinchilla position: the whole point of this rebuild.

    params = 16_913_280

    stats["params_assumed"] = params

    stats["tokens_per_param"] = round(stats["train"]["tokens"] / params, 2)

    stats["chinchilla_target_tokens_per_param"] = 20


    with open(os.path.join(args.data_dir, "token_stats.json"), "w", encoding="utf-8") as f:

        json.dump(stats, f, indent=2)


    print(json.dumps(stats, indent=2))


    # Verify the round trip survived the full pipeline, not just a toy sample.

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    head = np.fromfile(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, count=120)

    print("\nfirst 120 validation tokens decoded:\n")

    print(sp.decode([int(i) for i in head if i not in (sp.bos_id(), sp.eos_id())]))

    return 0



if __name__ == "__main__":

    sys.exit(main())
