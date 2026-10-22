#!/usr/bin/env python3

"""

Phase 3 review: categorise failures and build the findings table.


Walks the probe generations, shows the greedy and sampled output for each probe

side by side, and records a failure row using the assignment's exact category

names. Progress is saved after every entry, so the session can be interrupted.


Comparing greedy against sampled matters: a failure present in both is a model

failure, while one present only under greedy is a decoding artefact. The tool

records which, because that distinction drives the Phase 4 intervention.


Outputs:

  out/baseline/phase3_findings.csv    the findings table for the repository

  out/baseline/phase3_findings.json   progress / resumable state


Usage:

  python review_phase3.py

  python review_phase3.py --summary

"""


import argparse

import csv

import json

import os

import sys

from collections import Counter


CATEGORIES = {

    "1": "syntax",

    "2": "invented-api",

    "3": "wrong-logic",

    "4": "insecure",

    "5": "memorised",

    "6": "prompt-sensitive",

    "7": "context-ignored",

    "8": "other",

}


CATEGORY_HELP = {

    "syntax": "output does not parse",

    "invented-api": "calls an API or function that does not exist",

    "wrong-logic": "parses/runs but produces the wrong behaviour",

    "insecure": "introduces a security weakness",

    "memorised": "reproduces training data nearly verbatim",

    "prompt-sensitive": "a small rewording changes the answer substantially",

    "context-ignored": "ignores information clearly present in the input",

    "other": "none of the above; explain clearly",

}


SEVERITY = {"1": "low", "2": "medium", "3": "high"}



def load_meta(path):

    meta = {}

    if not os.path.exists(path):

        return meta

    with open(path, newline="", encoding="utf-8") as f:

        for row in csv.DictReader(f):

            meta[int(row["probe_index"]) - 1] = row

    return meta



def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--generations", default="out/baseline/phase3_generations.json")

    ap.add_argument("--meta", default="probes_meta.csv")

    ap.add_argument("--progress", default="out/baseline/phase3_findings.json")

    ap.add_argument("--csv", default="out/baseline/phase3_findings.csv")

    ap.add_argument("--chars", type=int, default=700)

    ap.add_argument("--summary", action="store_true")

    args = ap.parse_args()


    with open(args.generations, encoding="utf-8") as f:

        data = json.load(f)


    by_probe = {}

    for r in data["generations"]:

        by_probe.setdefault(r["prompt_index"], {})[

            "greedy" if r["decoding"] == "greedy" else "sampled"

        ] = r


    meta = load_meta(args.meta)

    findings = {}

    if os.path.exists(args.progress):

        with open(args.progress, encoding="utf-8") as f:

            findings = {int(k): v for k, v in json.load(f).items()}


    if args.summary:

        cats = Counter(v["category"] for v in findings.values())

        sev = Counter(v["severity"] for v in findings.values())

        caught = Counter(v["unit_test_catches"] for v in findings.values())

        print("\n" + "=" * 58)

        print(f"PHASE 3 FINDINGS: {len(findings)} failures recorded")

        print("=" * 58)

        print("\nBy category:")

        for c, n in cats.most_common():

            print(f"  {c:<18} {n:>3}")

        print(f"\n  distinct categories: {len(cats)}")

        print("\nBy severity:")

        for s, n in sev.most_common():

            print(f"  {s:<18} {n:>3}")

        print("\nWould a unit test catch it?")

        for k, n in caught.most_common():

            print(f"  {k:<18} {n:>3}")

        print("=" * 58)

        return 0


    print(f"{len(by_probe)} probes. {len(findings)} failures already recorded.\n")

    print("Categories: " + ", ".join(f"[{k}]{v}" for k, v in CATEGORIES.items()))

    print("Press ENTER at the category prompt to skip a probe (no failure).\n")


    for pi in sorted(by_probe):

        if pi in findings:

            continue


        g = by_probe[pi].get("greedy")

        s = by_probe[pi].get("sampled")

        m = meta.get(pi, {})


        print("\n" + "=" * 72)

        print(f"PROBE {pi + 1}/{len(by_probe)}   {m.get('probe_id', '')}  "

              f"type={m.get('probe_type', '?')}  expected={m.get('targeted_category', '?')}")

        print("=" * 72)

        if m.get("design_rationale"):

            print(f"designed to test: {m['design_rationale']}\n")


        print("--- GREEDY " + "-" * 60)

        print((g["full"] if g else "(missing)")[: args.chars])

        print(f"\n[parses={g['parses'] if g else '?'} uses_params={g['uses_params'] if g else '?'}]")


        print("\n--- SAMPLED " + "-" * 59)

        print((s["full"] if s else "(missing)")[: args.chars])

        print(f"\n[parses={s['parses'] if s else '?'} uses_params={s['uses_params'] if s else '?'}]")


        print("-" * 72)

        cat = input("category [1-8, ENTER to skip]: ").strip()

        if not cat:

            continue

        if cat not in CATEGORIES:

            print("  invalid, skipping")

            continue

        category = CATEGORIES[cat]

        print(f"  -> {category}: {CATEGORY_HELP[category]}")


        why = input("why wrong (one specific line): ").strip()

        sev = input("severity [1=low 2=med 3=high]: ").strip()

        severity = SEVERITY.get(sev, "medium")

        both = input("present in BOTH greedy and sampled? [y/n]: ").strip().lower()

        test = input("would a unit test catch it? [y/n/partial]: ").strip().lower()

        test = {"y": "yes", "n": "no"}.get(test, "partial")


        findings[pi] = {

            "probe_id": m.get("probe_id", f"P{pi+1:02d}"),

            "probe_type": m.get("probe_type", ""),

            "targeted_category": m.get("targeted_category", ""),

            "category": category,

            "why_wrong": why,

            "severity": severity,

            "in_both_decodings": "yes" if both == "y" else "no",

            "decoding_artefact": "no" if both == "y" else "likely",

            "unit_test_catches": test,

            "how_found": f"targeted probe ({m.get('probe_type', 'designed')})",

            "prompt": g["prompt"] if g else "",

            "output_greedy": (g["full"] if g else "")[:800],

            "parses": g["parses"] if g else None,

            "uses_params": g["uses_params"] if g else None,

        }


        with open(args.progress, "w", encoding="utf-8") as f:

            json.dump({str(k): v for k, v in findings.items()}, f, indent=2)

        print(f"  recorded ({len(findings)} total)")


    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    cols = ["probe_id", "probe_type", "targeted_category", "category", "why_wrong",

            "severity", "in_both_decodings", "decoding_artefact",

            "unit_test_catches", "how_found", "parses", "uses_params",

            "prompt", "output_greedy"]

    with open(args.csv, "w", newline="", encoding="utf-8") as f:

        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")

        w.writeheader()

        for pi in sorted(findings):

            row = dict(findings[pi])

            row["prompt"] = row["prompt"].replace("\n", "\\n")

            row["output_greedy"] = row["output_greedy"].replace("\n", "\\n")

            w.writerow(row)


    print(f"\nwrote {args.csv} ({len(findings)} failures)")

    return 0



if __name__ == "__main__":

    sys.exit(main())
