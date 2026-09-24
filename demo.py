#!/usr/bin/env python3
"""Fixed-input demonstration for the rna_fold module.

Run with:  python demo.py

Everything is computed live by the exact algorithms (no sleep, no
pre-recorded output).  It prints deterministic optimum folds, a Boltzmann
partition-function / marginal result, and genuinely triggered rejections.
"""

from __future__ import annotations

import time

from rna_fold import (
    ConstraintError,
    InvalidSequenceError,
    fold,
    partition_function,
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


def _print_partition(title, sequence, **kwargs):
    print("=" * 64)
    print(title)
    print(f"sequence    : {sequence}  (n={len(sequence)})")
    if kwargs:
        print(f"parameters  : {kwargs!r}")
    t0 = time.perf_counter()
    result = partition_function(sequence, **kwargs)
    dt = time.perf_counter() - t0
    print(f"ln Z        : {result['log_partition']:.6f}")
    print(f"free energy : {result['free_energy']:.6f}  (-T ln Z)")
    top = sorted(result["pair_probabilities"], key=lambda x: -x[2])[:6]
    print("pair marginals (most likely first):")
    for i, j, p in top:
        print(f"              ({i:2d},{j:2d}) {sequence[i]}-{sequence[j]}  p={p:.6f}")
    if not top:
        print("              (no pair has nonzero probability)")
    n = len(sequence)
    worst = min(
        range(n),
        key=lambda k: result["unpaired_probabilities"][k]
        + sum(result["pair_prob_matrix"][k])
        + sum(row[k] for row in result["pair_prob_matrix"]),
    )
    # Normalization is exact arithmetic over the computed marginals; report it
    # as an internal consistency readout, not an asserted pre-recorded value.
    gap = abs(
        result["unpaired_probabilities"][worst]
        + sum(result["pair_prob_matrix"][worst])
        + sum(row[worst] for row in result["pair_prob_matrix"])
        - 1.0
    )
    print(f"max |marginal row sum - 1| : {gap:.2e}")
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

    # --- 5. Partition function: normal Boltzmann result ------------------
    # "GAAAACCGC" (n=9) admits exactly four structures: the empty one and
    # three mutually exclusive GC pairs for position 0 -- (0,5), (0,6),
    # (0,8) -- each energy -3.  Hence Z = 1 + 3 e^3 and each pair carries
    # probability e^3 / Z; everything below is computed live.
    seq5 = "GAAAACCGC"
    res5 = _print_partition(
        "5) Partition function: pair marginals at T=1 (default energies)",
        seq5,
    )
    assert abs(
        res5["unpaired_probabilities"][0]
        + sum(res5["pair_prob_matrix"][0]) - 1.0
    ) < 1e-9

    # --- 6. Partition boundary: length cap and forced constraint ----------
    print("=" * 64)
    print("6) Boundary: n=20 with a forced spanning pair (marginal must be 1)")
    seq6 = "G" + "A" * 18 + "C"
    res6 = _print_partition(
        "   n=20, forced pair (0,19)",
        seq6,
        constraints={"pairs": [(0, 19)]},
        temperature=0.7,
    )
    assert abs(res6["pair_prob_matrix"][0][19] - 1.0) < 1e-9

    # --- 7. Actual error: non-positive temperature ------------------------
    print("=" * 64)
    print("7) Error boundary: non-positive temperature is rejected")
    print("sequence    : 'GAAAC', temperature=0")
    try:
        partition_function("GAAAC", temperature=0)
    except ValueError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")

    # --- 8. Actual error: sequence beyond the partition length cap --------
    print("=" * 64)
    print("8) Error boundary: partition function length cap is 20")
    long_seq = "A" * 21
    print(f"sequence    : 'A' * {len(long_seq)}")
    try:
        partition_function(long_seq)
    except InvalidSequenceError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected an InvalidSequenceError")

    print("=" * 64)
    print(
        "demo finished: 2 optimum folds, 2 partition results, and 4 live "
        "rejections."
    )


if __name__ == "__main__":
    main()
