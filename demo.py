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
    score_structure,
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

    # --- 2b. Adjacent nesting bonus changes the optimum -----------------
    # GGAACUU (n=7): a lone GC pair (0,4) scores 3; the directly nested
    # pair of weak GU pairs (0,6)+(1,5) scores only 1+1 by itself, but a
    # stack bonus of 3 makes 1+1+3=5 and flips the optimum.  This is a
    # synthetic discrete scoring rule, NOT a real RNA free-energy model.
    seq_b = "GGAACUU"
    res_no = _print_result(
        "2b) No stack bonus (stack_score=0): lone GC wins",
        seq_b,
        stack_score=0,
    )
    assert res_no["structure"] == "(...).."
    res_b = _print_result(
        "2b) Adjacent nesting bonus stack_score=3: nested GU pair wins",
        seq_b,
        stack_score=3,
    )
    assert res_b["structure"] == "((...))"
    assert res_b["score"] == 1 + 1 + 3
    # Independent rescoring of the returned dot-bracket agrees.
    assert (
        score_structure(seq_b, res_b["structure"], stack_score=3)
        == res_b["score"]
    )

    # --- 2c. Per-position unpaired payoff -------------------------------
    # Paying position 0 to stay single changes whether it pairs.
    res_c = _print_result(
        "2c) Unpaired payoff unpaired_scores=[5,0,...]: position 0 stays open",
        seq_b,
        unpaired_scores=[5, 0, 0, 0, 0, 0, 0],
    )
    assert res_c["structure"][0] == "."

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
    print("demo finished: successful folds (incl. stack/unpaired scoring) "
          "and 2 live rejections.")


if __name__ == "__main__":
    main()
