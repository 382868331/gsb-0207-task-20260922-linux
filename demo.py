#!/usr/bin/env python3
"""Fixed-input demonstration for the rna_fold module.

Run with:  python demo.py

Everything is computed live by the exact algorithm (no sleep, no pre-recorded
output).  It prints normal folds, a tied-optima batch, and two hard input
boundaries that are actually rejected.
"""

from __future__ import annotations

import time

from rna_fold import (
    ConstraintError,
    InvalidSequenceError,
    fold,
    fold_batch,
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

    batch = fold_batch(seq1, forced, batch_size=2)
    print(f"batch score : {batch['score']}")
    print(f"total count : {batch['count']} (mod 1_000_000_007)")
    for idx, item in enumerate(batch["structures"], start=1):
        print(f"  optimum {idx}: {item['structure']}")
    print(f"has more    : {batch['has_more']}")
    assert len(batch["structures"]) == res2["count"]
    assert batch["next_cursor"] is None

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

    print("=" * 64)
    print("demo finished: normal folds, a tied-optima batch, and 2 rejections.")


if __name__ == "__main__":
    main()
