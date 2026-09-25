#!/usr/bin/env python3
"""Fixed-input demonstration for the rna_fold module.

Run with:  python demo.py

Everything is computed live by the exact algorithm (no sleep, no pre-recorded
output).  It prints one normal, successful fold and one hard-constraint
conflict that is actually rejected.
"""

from __future__ import annotations

import time

from rna_fold import (
    ConstraintError,
    InvalidSequenceError,
    fold,
)


def _print_result(title, sequence, constraints=None, **scoring):
    print("=" * 64)
    print(title)
    print(f"sequence    : {sequence}  (n={len(sequence)})")
    if constraints is not None:
        print(f"constraints : {constraints!r}")
    if scoring:
        print(f"scoring     : {scoring!r}")
    t0 = time.perf_counter()
    result = fold(sequence, constraints, **scoring)
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

    # --- 3. New scoring: stacking bonus changes the optimal structure ----
    seq3 = "GUGAUCCUAUGC"
    res3a = _print_result("3a) Baseline scoring (no stacking bonus)", seq3)
    res3b = _print_result(
        "3b) Same sequence with stack_score=3 (adjacent nested pairs "
        "(i,j)+(i+1,j-1) earn +3 each)",
        seq3,
        stack_score=3,
    )
    assert res3a["structure"] != res3b["structure"]

    # --- 4. New scoring: per-position unpaired scores --------------------
    seq4 = "GAAAU"
    res4 = _print_result(
        "4) Per-position unpaired scores: +1 per unpaired position makes "
        "the weak GU pair not worth forming",
        seq4,
        unpaired_scores=[1] * len(seq4),
    )
    assert res4["structure"] == "....."

    # --- 5. Actual rejection: conflicting hard constraints ---------------
    # (0,9)=G-C and (4,13)=A-U are each individually legal, but they cross
    # (0 < 4 < 9 < 13): a pseudoknot, rejected up front with locations.
    print("=" * 64)
    print("5) Conflict boundary: crossing forced pairs are rejected")
    print(f"sequence    : {seq1}  (n={len(seq1)})")
    bad = {"pairs": [(0, 9), (4, 13)]}
    print(f"constraints : {bad!r}")
    try:
        fold(seq1, bad)
    except ConstraintError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - demo must observe the rejection
        raise AssertionError("expected a ConstraintError")

    # --- 6. Actual rejection: illegal character, error is locatable ------
    print("=" * 64)
    print("6) Validation boundary: illegal character is located")
    bad_seq = "ACGTACGU"
    print(f"sequence    : {bad_seq!r}")
    try:
        fold(bad_seq)
    except InvalidSequenceError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected an InvalidSequenceError")

    print("=" * 64)
    print("demo finished: 4 successful folds and 2 live rejections.")


if __name__ == "__main__":
    main()
