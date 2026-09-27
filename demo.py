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
    CursorError,
    InvalidSequenceError,
    claim_optimal_structures,
    fold,
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

    # --- 5. Batched claiming of tied-optimal structures -----------------
    # This sequence has four distinct maximum-score structures.  Claim them
    # in batches of 2 using the opaque continuation cursor; order is stable
    # (strict dot-bracket lex order) and nothing is returned twice.
    tie_seq = "CGUCACGACCGG"
    print("=" * 64)
    print("5) Batched claiming of tied-optimal structures (limit=2)")
    print(f"sequence    : {tie_seq}  (n={len(tie_seq)})")
    cursor = None
    batch_no = 0
    seen = set()
    while True:
        batch_no += 1
        batch = claim_optimal_structures(tie_seq, None, 2, cursor)
        for structure in batch["structures"]:
            assert structure not in seen, "structure returned twice"
            seen.add(structure)
            print(f"  batch {batch_no}: {structure}  (score {batch['score']})")
        cursor = batch["next_cursor"]
        if batch["done"]:
            break
    print(f"total claimed: {len(seen)} distinct structures "
          f"(count field: {batch['count']})")

    # --- 6. Cursor is bound to its concrete input -----------------------
    print("=" * 64)
    print("6) Continuation cursor bound to the input: reuse on a changed")
    print("   sequence is rejected")
    first = claim_optimal_structures(tie_seq, None, 1)
    print(f"sequence    : {tie_seq!r}, first structure: "
          f"{first['structures'][0]!r}")
    try:
        claim_optimal_structures("CGUCACGACCGU", None, 1, first["next_cursor"])
    except CursorError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("expected a CursorError")

    print("=" * 64)
    print("demo finished: folds, tied-optima batches, and live rejections.")


if __name__ == "__main__":
    main()
