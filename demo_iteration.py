#!/usr/bin/env python3
"""Fixed-input demonstration of the single H-type pseudoknot mode.

Run with:  python demo_iteration.py

Everything is computed live by the exact algorithm (no sleep, no
pre-recorded output).  It shows the new pseudoknot mode finding and
marking two crossing stems, plus boundary cases: a penalty high enough
to push the optimum back to the pseudoknot-free fold, and the 24 nt
length limit of the mode.
"""

from __future__ import annotations

import time

from rna_fold import InvalidSequenceError, fold


def _print_result(title, sequence, constraints=None, **kwargs):
    print("=" * 68)
    print(title)
    print(f"sequence    : {sequence}  (n={len(sequence)})")
    if constraints is not None:
        print(f"constraints : {constraints!r}")
    if kwargs:
        print(f"options     : {kwargs}")
    t0 = time.perf_counter()
    result = fold(sequence, constraints, **kwargs)
    dt = time.perf_counter() - t0
    print(f"structure   : {result['structure']}")
    print(f"score       : {result['score']}")
    print(f"opt count   : {result['count']} (mod 1_000_000_007)")
    pk = result.get("pseudoknot")
    if pk is not None:
        s1 = ", ".join(f"({i},{j})" for i, j in pk["stem1"])
        s2 = ", ".join(f"({i},{j})" for i, j in pk["stem2"])
        print(f"pseudoknot  : stem1 () = {s1}")
        print(f"              stem2 [] = {s2}")
        print(f"              penalty charged once = {pk['penalty']}")
    else:
        print("pseudoknot  : none (pseudoknot-free optimum)")
    print(f"elapsed     : {dt * 1000:.1f} ms")
    return result


def main() -> None:
    # --- 1. New feature: one H-type pseudoknot wins ---------------------
    # Two GC stems whose best arrangement crosses (i < k < j < l).
    seq = "GGGAAGGGACCCAACCC"
    res1 = _print_result(
        "1) Pseudoknot mode: two crossing stems, marked () and []",
        seq,
        pseudoknot_penalty=2,
    )
    pk = res1["pseudoknot"]
    assert pk is not None and "[" in res1["structure"]
    i, j = pk["stem1"][0]
    k, l = pk["stem2"][0]
    assert i < k < j < l  # the defining H-type interleaving
    base = fold(seq)
    assert res1["score"] == base["score"] + 3 - 2  # 1 extra GC pair - penalty

    # --- 2. Boundary: a large penalty pushes the optimum back ----------
    res2 = _print_result(
        "2) Boundary: penalty 50 makes the pseudoknot not worth it",
        seq,
        pseudoknot_penalty=50,
    )
    assert res2["pseudoknot"] is None
    assert res2["score"] == base["score"]
    assert res2["structure"] == base["structure"]

    # --- 3. Boundary: the mode caps sequences at 24 nt ------------------
    print("=" * 68)
    print("3) Boundary: pseudoknot mode rejects sequences longer than 24")
    long_seq = "G" * 25
    print(f"sequence    : {long_seq!r}  (n=25)")
    try:
        fold(long_seq, pseudoknot_penalty=0)
    except InvalidSequenceError as exc:
        print(f"raised      : {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - demo must observe the rejection
        raise AssertionError("expected an InvalidSequenceError")
    # ...while the classic pseudoknot-free interface still accepts it.
    ok = fold(long_seq)
    print(f"classic fold of the same 25-mer still works: score {ok['score']}")

    print("=" * 68)
    print("demo_iteration finished: 1 pseudoknot fold and 2 live boundaries.")


if __name__ == "__main__":
    main()
