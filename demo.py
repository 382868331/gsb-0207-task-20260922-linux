"""Deterministic offline demo for the rna_fold package.

Runs a real fold on a deterministically generated 48-nt sequence, shows a
constrained fold, and then triggers two genuine rejections (conflicting
hard constraints and an invalid base).  No sleeps, no prerecorded output.
"""

import random

from rna_fold import ConstraintError, SequenceError, fold


def main():
    # --- 1. Normal fold on a deterministically generated sequence. --------
    rng = random.Random(20260923)
    seq = "".join(rng.choice("ACGU") for _ in range(48))
    result = fold(seq)
    print("=== Unconstrained fold (48 nt, deterministically generated) ===")
    print(f"sequence : {seq}")
    print(f"structure: {result.structure}")
    print(f"score    : {result.score}")
    print(f"#optimal structures (mod 1_000_000_007): {result.count}")
    print(f"#pairs that may appear in an optimum : {len(result.pairs)}")
    print(f"pairs    : {sorted(result.pairs)}")

    # --- 2. Hard constraints: force one pair, forbid others. --------------
    forced_pair = sorted(result.pairs)[0]
    unpaired = {i for i, ch in enumerate(result.structure) if ch == "("}
    unpaired.difference_update(forced_pair)
    constrained = fold(seq, {"forced_pairs": [forced_pair],
                             "unpaired": unpaired})
    print("\n=== Constrained fold ===")
    print(f"forced pair {forced_pair}, forced-unpaired {sorted(unpaired)}")
    print(f"structure: {constrained.structure}")
    print(f"score    : {constrained.score}")
    print(f"#optimal structures: {constrained.count}")

    # --- 3. Genuine rejection: crossing forced pairs. ---------------------
    print("\n=== Rejection: crossing forced pairs ===")
    seq2 = "GAAAC" * 4  # G at 0,5,10,15 and C at 4,9,14,19
    print(f"sequence : {seq2}, forced pairs (0, 9) and (5, 14)")
    try:
        fold(seq2, {"forced_pairs": [(0, 9), (5, 14)]})
    except ConstraintError as exc:
        print(f"rejected with ConstraintError: {exc}")

    # --- 4. Genuine rejection: invalid base. ------------------------------
    print("\n=== Rejection: invalid base ===")
    try:
        fold("ACGUNACGU")
    except SequenceError as exc:
        print(f"rejected with SequenceError: {exc}")


if __name__ == "__main__":
    main()
