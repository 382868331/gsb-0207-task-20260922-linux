#!/usr/bin/env python3
"""Live demonstration of the optional single-H-type pseudoknot mode.

Run with:  python demo_iteration.py

All output is computed on the spot by the exact algorithms (no sleep, no
pre-recorded values).  It shows:

1. a pseudoknot-free fold (old behaviour);
2. the same sequence folded with the single-H pseudoknot switch on;
3. the penalty boundary at which the optimum falls back to the old fold;
4. an actually triggered rejection: forced pairs demand a *second*
   crossing, which the single-H topology forbids.
"""

from __future__ import annotations

import time

from rna_fold import UnsatisfiableConstraintError, fold


# stem1 = (0,7),(1,6); stem2 = (3,10),(4,9); four GC stacks, n=12
SEQ = "GGAGGACCACCA"


def _show(title: str, **kwargs) -> dict:
    print("=" * 68)
    print(title)
    print(f"sequence    : {SEQ}  (n={len(SEQ)})")
    if kwargs:
        print(f"arguments   : {kwargs}")
    t0 = time.perf_counter()
    result = fold(SEQ, **kwargs)
    dt = (time.perf_counter() - t0) * 1000
    print(f"structure   : {result['structure']}")
    print(f"score       : {result['score']}")
    print(f"pseudoknot  : {result['pseudoknot']}")
    print(f"elapsed     : {dt:.1f} ms")
    return result


def main() -> None:
    free = _show("1) Pseudoknot-free fold (old interface, unchanged)")
    assert free["pseudoknot"] is None
    assert "[" not in free["structure"]

    pk = _show(
        "2) Single-H pseudoknot enabled: two crossing GC stems win",
        allow_pseudoknot=True,
    )
    assert pk["pseudoknot"] is not None
    assert pk["score"] > free["score"]

    light = _show(
        "3) Small penalty: the crossing is still optimal",
        allow_pseudoknot=True,
        pseudoknot_penalty=1,
    )
    assert light["pseudoknot"] is not None

    heavy = _show(
        "4) Large penalty: the optimum falls back to the old structure",
        allow_pseudoknot=True,
        pseudoknot_penalty=5,
    )
    assert heavy["pseudoknot"] is None
    assert heavy["structure"] == free["structure"]
    assert heavy["score"] == free["score"]

    print("=" * 68)
    print("5) Boundary: forced pairs demanding a SECOND crossing rejected")
    print(f"sequence    : {SEQ}  (n={len(SEQ)})")
    bad = {"pairs": [(0, 7), (3, 10), (1, 9)]}
    print(f"constraints : {bad!r}")
    try:
        fold(SEQ, bad, allow_pseudoknot=True)
    except UnsatisfiableConstraintError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - demo must observe the rejection
        raise AssertionError("expected UnsatisfiableConstraintError")

    print("=" * 68)
    print("demo finished: pseudoknot win, penalty fallback, hard rejection.")


if __name__ == "__main__":
    main()
