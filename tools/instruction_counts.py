#!/usr/bin/env python3
"""Instruction-count snapshots and paired build comparisons (macOS).

usage:
  tools/instruction_counts.py snapshot [--worktree DIR] [--no-examples] OUTDIR
  tools/instruction_counts.py compare BASE_DIR CAND_DIR [--rounds N] [--reps N]
                                      [--filter SUBSTRING] [--no-examples] [--commit]

`snapshot` builds the `instruction_counts` bench and the deterministic release
examples, then copies the binaries and the source revision into OUTDIR.

`compare` alternates base and candidate runs, requires identical output
fingerprints, and prints a Markdown table of minimum retired instructions
(the steadiest statistic on a busy host) and median cycles; `--commit` prints
a 72-column instruction table for commit messages instead. It exits with
status 1 when any fingerprint differs.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

EXAMPLES = ("software_team", "simple_experiment")
TIME_COUNTER = re.compile(r"^\s*(\d+)\s+(instructions retired|cycles elapsed)\s*$", re.M)


def run(command, cwd=None):
    try:
        return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stdout or "")
        sys.stderr.write(error.stderr or "")
        raise


def snapshot(args):
    worktree = Path(args.worktree).resolve()
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = run(
        ["cargo", "bench", "--all-features", "--bench", "instruction_counts", "--no-run",
         "--message-format=json-render-diagnostics"],
        cwd=worktree,
    )
    executables = [
        message["executable"]
        for message in map(json.loads, filter(None, build.stdout.splitlines()))
        if message.get("executable")
    ]
    shutil.copy2(executables[-1], out / "instruction_counts")
    if not args.no_examples:
        command = ["cargo", "build", "--release"]
        for example in EXAMPLES:
            command += ["--example", example]
        run(command, cwd=worktree)
        metadata = json.loads(
            run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=worktree).stdout
        )
        examples = Path(metadata["target_directory"]) / "release" / "examples"
        for example in EXAMPLES:
            shutil.copy2(examples / example, out / example)
    revision = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=worktree).stdout
    (out / "revision.txt").write_text(revision + ("\ndirty:\n" + dirty if dirty else "\n"))
    print(f"snapshot {out} at {revision}{' (dirty)' if dirty else ''}")


def bench_rows(directory, reps, substring):
    command = [str(directory / "instruction_counts"), "--bench", "--json", "--reps", str(reps)]
    if substring:
        command += ["--filter", substring]
    return [json.loads(line) for line in run(command).stdout.splitlines() if line.startswith("{")]


def example_row(directory, example, reps):
    instructions, cycles, outputs = [], [], set()
    for _ in range(reps):
        completed = run(["/usr/bin/time", "-l", str(directory / example)])
        counters = {name: int(value) for value, name in TIME_COUNTER.findall(completed.stderr)}
        instructions.append(counters.get("instructions retired"))
        cycles.append(counters.get("cycles elapsed"))
        outputs.add(hashlib.sha256(completed.stdout.encode()).hexdigest()[:16])
    return {
        "workload": f"process/{example}",
        "instructions_min": min(instructions) if None not in instructions else None,
        "cycles_min": min(cycles) if None not in cycles else None,
        # Nondeterministic output yields a unique mismatch marker.
        "fingerprint": outputs.pop() if len(outputs) == 1 else f"varies:{sorted(outputs)}",
    }


def collect(directory, args):
    rows = bench_rows(directory, args.reps, args.filter)
    if not args.no_examples:
        rows += [
            example_row(directory, example, args.reps)
            for example in EXAMPLES
            if (directory / example).exists()
            and (not args.filter or args.filter in f"process/{example}")
        ]
    return rows


def percent(base, candidate):
    if not base or candidate is None:
        return "n/a"
    return f"{100.0 * (candidate - base) / base:+.2f}%"


def compare(args):
    arms = {"base": Path(args.base).resolve(), "cand": Path(args.cand).resolve()}
    results = {arm: {} for arm in arms}
    order = []
    for round_index in range(args.rounds):
        sequence = list(arms) if round_index % 2 == 0 else list(reversed(list(arms)))
        for arm in sequence:
            for row in collect(arms[arm], args):
                name = row["workload"]
                if name not in order:
                    order.append(name)
                entry = results[arm].setdefault(
                    name, {"instructions": [], "cycles": [], "fingerprints": set()}
                )
                if row.get("instructions_min") is not None:
                    entry["instructions"].append(row["instructions_min"])
                if row.get("cycles_min") is not None:
                    entry["cycles"].append(row["cycles_min"])
                entry["fingerprints"].add(row["fingerprint"])
    rows = []
    for name in order:
        base = results["base"].get(name)
        cand = results["cand"].get(name)
        if not base or not cand or not base["instructions"] or not cand["instructions"]:
            rows.append((name, None, None, None, None, False))
            continue
        rows.append((
            name,
            min(base["instructions"]),
            min(cand["instructions"]),
            statistics.median(base["cycles"]) if base["cycles"] else None,
            statistics.median(cand["cycles"]) if cand["cycles"] else None,
            len(base["fingerprints"] | cand["fingerprints"]) == 1,
        ))
    ratios = [cand / base for _, base, cand, _, _, _ in rows if base and cand]
    geometric = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None
    differs = any(not same for *_, same in rows)
    if args.commit:
        # Fits a 72-column commit body.
        print(f"{'workload':<26}{'base':>14}{'candidate':>14}{'delta':>9}")
        for name, base, cand, _, _, same in rows:
            if base is None:
                print(f"{name:<26}{'missing':>14}")
                continue
            flag = "" if same else "  DIFFERENT"
            print(f"{name:<26}{base:>14,}{cand:>14,}{percent(base, cand):>9}{flag}")
        if geometric is not None:
            print(f"{'geometric mean':<54}{100.0 * (geometric - 1.0):>+8.2f}%")
        print(
            f"min of {args.rounds}x{args.reps} runs; outputs "
            f"{'DIFFER' if differs else 'identical'}"
        )
    else:
        for arm, directory in arms.items():
            revision = directory / "revision.txt"
            label = revision.read_text().splitlines()[0][:12] if revision.exists() else "?"
            print(f"{arm}: {directory} ({label})")
        print(f"rounds {args.rounds} x reps {args.reps}; "
              "instructions = min, cycles = median of round minima\n")
        print("| workload | base instructions | candidate instructions "
              "| Δ instructions | Δ cycles | outputs |")
        print("|---|---:|---:|---:|---:|---|")
        for name, base, cand, base_cycles, cand_cycles, same in rows:
            if base is None:
                print(f"| {name} | missing | | | | |")
                continue
            print(
                f"| {name} | {base:,} | {cand:,} | {percent(base, cand)} | "
                f"{percent(base_cycles, cand_cycles)} | {'same' if same else 'DIFFERENT'} |"
            )
        if geometric is not None:
            print(f"\ngeometric mean instruction ratio: {geometric:.4f} "
                  f"({100.0 * (geometric - 1.0):+.2f}%)")
    if differs:
        print("\nOUTPUT FINGERPRINTS DIFFER", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("outdir")
    snap.add_argument("--worktree", default=".")
    snap.add_argument("--no-examples", action="store_true")
    snap.set_defaults(handler=snapshot)
    comp = commands.add_parser("compare")
    comp.add_argument("base")
    comp.add_argument("cand")
    comp.add_argument("--rounds", type=int, default=5)
    comp.add_argument("--reps", type=int, default=3)
    comp.add_argument("--filter")
    comp.add_argument("--no-examples", action="store_true")
    comp.add_argument("--commit", action="store_true", help="72-column table for commit bodies")
    comp.set_defaults(handler=compare)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
