#!/usr/bin/env python3
"""Fixed-input demonstration for the rna_fold module.

Run with:  python demo.py

Everything is computed live by the exact algorithms (no sleep, no
pre-recorded output).  It prints maximum-score folds and Boltzmann partition
functions, plus hard-constraint conflicts and invalid parameters that are
actually rejected.
"""

from __future__ import annotations

import math
import time

from rna_fold import (
    ConstraintError,
    InvalidParameterError,
    InvalidSequenceError,
    fold,
    partition_fold,
)


def _print_result(title, sequence, constraints=None):
    print("=" * 64)
    print(title)
    print(f"sequence    : {sequence}  (n={len(sequence)})")
    if constraints is not None:
        print(f"constraints : {constraints!r}")
    t0 = time.perf_counter()
    result = fold(sequence, constraints)
    dt = time.perf_counter() - t0
    print(f"structure   : {result['structure']}")
    print(f"score       : {result['score']}")
    print(f"opt count   : {result['count']} (mod 1_000_000_007)")
    pairs = result["pairs"]
    shown = ", ".join(f"({i},{j})" for i, j in pairs[:8])
    more = "" if len(pairs) <= 8 else f"  ...({len(pairs) - 8} more)"
    print(f"optimal pairs ({len(pairs)}): {shown}{more}")
    print(f"elapsed     : {dt * 1000:.1f} ms")
    return result


def main() -> None:
    # --- 1. Normal result: free folding, mix of GC/AU/GU -----------------
    seq1 = "GGCGAUUAGCCGAUU"
    res1 = _print_result(
        "1) Free folding (GC=3, AU=2, GU=1; lex-smallest optimum)", seq1
    )
    assert len(res1["structure"]) == len(seq1)
    assert res1["score"] > 0

    # --- 2. Forced spanning pair changes the optimum --------------------
    forced = {"pairs": [(0, 14)]}  # outer G-C spanning the whole sequence
    res2 = _print_result(
        "2) Hard constraint: force the outer pair (0,14) to span everything",
        seq1,
        forced,
    )
    assert (0, 14) in res2["pairs"]

    # --- 3. Actual rejection: conflicting hard constraints ---------------
    # (0,9)=G-C and (4,13)=A-U are each individually legal, but they cross
    # (0 < 4 < 9 < 13): a pseudoknot, rejected up front with locations.
    print("=" * 64)
    print("3) Conflict boundary: crossing forced pairs are rejected")
    print(f"sequence    : {seq1}  (n={len(seq1)})")
    bad = {"pairs": [(0, 9), (4, 13)]}
    print(f"constraints : {bad!r}")
    try:
        fold(seq1, bad)
    except ConstraintError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - demo must observe the rejection
        raise AssertionError("expected a ConstraintError")

    # --- 4. Actual rejection: illegal character, error is locatable ------
    print("=" * 64)
    print("4) Validation boundary: illegal character is located")
    bad_seq = "ACGTACGU"
    print(f"sequence    : {bad_seq!r}")
    try:
        fold(bad_seq)
    except InvalidSequenceError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected an InvalidSequenceError")

    # --- 5. Boltzmann partition function: hand-checkable two structures ---
    print("=" * 64)
    print("5) Partition function (energies GC=-3, AU=-2, GU=-1; T=1)")
    seq5 = "GAAAC"  # n=5: exactly two admissible structures
    print(f"sequence    : {seq5}  (n={len(seq5)})")
    print("structures  : all-unpaired E=0 w=1 ; pair (0,4) GC E=-3 w=e^3")
    t0 = time.perf_counter()
    res5 = partition_fold(seq5)
    dt = time.perf_counter() - t0
    print(f"Z           : {res5['partition_function']:.12f} "
          f"(1 + e^3 = {1 + math.exp(3.0):.12f})")
    print(f"ln Z        : {res5['ln_partition_function']:.12f}")
    print(f"free energy : {res5['free_energy']:.12f} (= -T ln Z)")
    print(f"P(pair 0,4) : {res5['pair_probabilities'][(0, 4)]:.12f}")
    print(f"P(pos 0 unpaired): {res5['unpaired_probabilities'][0]:.12f}")
    print(f"elapsed     : {dt * 1000:.1f} ms")
    assert abs(
        res5["pair_probabilities"][(0, 4)]
        + res5["unpaired_probabilities"][0] - 1.0
    ) < 1e-12

    # --- 6. Boundary: n=20 cap, marginals normalise and forced pair is 1 ---
    print("=" * 64)
    print("6) Length boundary n=20 with a forced pair (probability exactly 1)")
    seq6 = "G" + "A" * 18 + "C"
    constraints6 = "(..................)"
    print(f"sequence    : {seq6}  (n={len(seq6)})")
    print(f"constraints : {constraints6!r}")
    res6 = partition_fold(seq6, constraints6, temperature=1.0)
    pp6 = res6["pair_probabilities"]
    print(f"ln Z        : {res6['ln_partition_function']:.6f}")
    print(f"P(pair 0,19): {pp6[(0, 19)]:.6f} (forced => 1)")
    worst = 0.0
    for k in range(20):
        total = res6["unpaired_probabilities"][k] + sum(
            p for (a, b), p in pp6.items() if a == k or b == k
        )
        worst = max(worst, abs(total - 1.0))
    print(f"max |marginal sum - 1| over all positions: {worst:.3e}")
    assert pp6[(0, 19)] == 1.0
    assert worst < 1e-12

    # --- 7. Actual rejection: non-positive temperature -------------------
    print("=" * 64)
    print("7) Parameter boundary: temperature must be strictly positive")
    print("sequence    : 'GAAAC', temperature=0")
    try:
        partition_fold("GAAAC", temperature=0)
    except InvalidParameterError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected an InvalidParameterError")

    print("=" * 64)
    print("demo finished: 2 max-score folds, 2 partition functions and "
          "3 live rejections.")


if __name__ == "__main__":
    main()
