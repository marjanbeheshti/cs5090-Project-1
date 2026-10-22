#!/usr/bin/env python3

"""

Step 5: train the BPE tokenizer.


Trains ONLY on data/train.txt. The validation split must never influence the

vocabulary, or val loss stops being an honest measurement.


This is not neural training -- no gradients, no GPU. SentencePiece counts

frequent character pairs and merges them. It takes minutes on CPU.


Outputs:

  tokenizer/code_bpe.model, tokenizer/code_bpe.vocab

  tokenizer/tokenizer_stats.json


Usage:

  python train_tokenizer.py --vocab-size 8192

"""


import argparse

import json

import os

import sys


import sentencepiece as spm


# Written between files by split_corpus.py; kept as one atomic token.

DOC_SEP_TOKEN = "<|file|>"



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", default="data/train.txt")

    ap.add_argument("--out-dir", default="tokenizer")

    ap.add_argument("--prefix", default="code_bpe")

    ap.add_argument("--vocab-size", type=int, default=8192)

    ap.add_argument(

        "--input-sentence-size",

        type=int,

        default=2_000_000,

        help="lines sampled for training; the full file will not fit in memory",

    )

    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()


    os.makedirs(args.out_dir, exist_ok=True)

    model_prefix = os.path.join(args.out_dir, args.prefix)


    spm.SentencePieceTrainer.train(

        input=args.input,

        model_prefix=model_prefix,

        vocab_size=args.vocab_size,

        model_type="bpe",

        # Code cannot afford to drop rare characters, so cover all of them.

        character_coverage=1.0,

        # Sample lines rather than loading 1.2 GB.

        input_sentence_size=args.input_sentence_size,

        shuffle_input_sentence=True,

        # Unknown characters become raw bytes instead of <unk>.

        byte_fallback=True,

        # llama-style special token ids.

        unk_id=0,

        bos_id=1,

        eos_id=2,

        pad_id=-1,

        user_defined_symbols=[DOC_SEP_TOKEN],

        # Indentation is meaningful in Python; do not strip or collapse it.

        remove_extra_whitespaces=False,

        normalization_rule_name="identity",

        num_threads=os.cpu_count() or 8,

        train_extremely_large_corpus=False,

        seed_sentencepiece_size=1_000_000,

    )


    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")


    # Round-trip check on real indented code. If this breaks, everything

    # downstream is built on a broken tokenizer.

    sample = 'def f(x):\n    if x > 0:\n        return x + 1\n    return 0\n'

    ids = sp.encode(sample)

    back = sp.decode(ids)

    ok = back == sample


    stats = {

        "input": args.input,

        "vocab_size": sp.get_piece_size(),

        "model": model_prefix + ".model",

        "unk_id": sp.unk_id(),

        "bos_id": sp.bos_id(),

        "eos_id": sp.eos_id(),

        "doc_sep_token": DOC_SEP_TOKEN,

        "doc_sep_id": sp.piece_to_id(DOC_SEP_TOKEN),

        "roundtrip_exact": ok,

        "sample_tokens": len(ids),

        "sample_chars": len(sample),

        "chars_per_token_on_sample": round(len(sample) / len(ids), 2),

    }


    with open(os.path.join(args.out_dir, "tokenizer_stats.json"), "w", encoding="utf-8") as f:

        json.dump(stats, f, indent=2)


    print(json.dumps(stats, indent=2))


    if not ok:

        print("\nROUND-TRIP FAILED -- indentation or newlines were altered.")

        print("decoded:", repr(back))

        return 1


    print("\nround-trip exact: indentation and newlines preserved")

    return 0



if __name__ == "__main__":

    sys.exit(main())
