"""Independent brute-force reference for the single-H pseudoknot mode.

This module never imports the code under test.  It enumerates admissible
matchings from scratch by recursively pairing the first free position.  A
matching is accepted when it is either non-crossing or contains exactly one
H-type crossing: two stacked stems whose outer pairs satisfy
``i < k < j < l``, with every other pair non-crossing relative to them.
Arbitrary crossing networks are rejected.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

PAIR_SCORES = {
    ("G", "C"): 3,
    ("C", "G"): 3,
    ("A", "U"): 2,
    ("U", "A"): 2,
    ("G", "U"): 1,
    ("U", "G"): 1,
}

MIN_HAIRPIN = 3
MOD = 1_000_000_007


def analyze(
    pairs: FrozenSet[Tuple[int, int]]
) -> Optional[Tuple[str, Optional[Tuple[set, set]]]]:
    """Classify a matching.

    Returns ``('none', None)``, ``('single-h', (stem1, stem2))`` or ``None``
    when the crossing pattern is not one admissible H pseudoknot.
    """
    plist = sorted(pairs)
    crosses: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for x in range(len(plist)):
        for y in range(x + 1, len(plist)):
            (a, b), (c, d) = plist[x], plist[y]
            if a < c < b < d:
                crosses.append((plist[x], plist[y]))

    if not crosses:
        return ("none", None)

    # Fix the outermost crossing pair of stems.
    (a, b), (c, d) = min(crosses, key=lambda z: (z[0][0], z[1][0]))

    # Each stem is the *maximal* consecutive nested run starting at its
    # outer pair: include every nested pair that exists.
    stem1 = {(a, b)}
    h = 1
    while (a + h, b - h) in pairs:
        stem1.add((a + h, b - h))
        h += 1
    stem2 = {(c, d)}
    h = 1
    while (c + h, d - h) in pairs:
        stem2.add((c + h, d - h))
        h += 1

    # The four stack blocks must be disjoint and ordered around the
    # sequence:  left1 < left2 < right1 < right2, i.e. every left arm of
    # stem 1 precedes every left arm of stem 2, which precedes every right
    # arm of stem 1, which precedes every right arm of stem 2.
    p, q = len(stem1), len(stem2)
    i1, j1 = a + p - 1, b - p + 1
    i2, j2 = c + q - 1, d - q + 1
    # Left-arm blocks [a,i1] and [c,i2] disjoint (c > i1), right-arm
    # blocks [j1,b] and [j2,d] disjoint (j2 > b), plus the innermost
    # crossing i2 < j1.  Outer crossing already gives a<c<b<d.
    # Gaps may be empty; blocks are disjoint when c >= i1 + 1 and
    # j2 >= b + 1, while the innermost crossing requires i2 <= j1.
    if not (i1 < c and i2 <= j1 and b < j2):
        return None
    # The strict four-block order forces both maximal stacks to have the
    # same height and every layer to cross at the same depth.
    if p != q:
        return None
    for h in range(p):
        if not (a + h < c + h < b - h < d - h):
            return None

    # Every crossing in the matching must be between these two stems only.
    for pair1, pair2 in crosses:
        if not (
            pair1 in stem1
            and pair2 in stem2
            or pair2 in stem1
            and pair1 in stem2
        ):
            return None
    return ("single-h", (stem1, stem2))


def _enumerate(
    seq: str,
    i: int,
    n: int,
    used: List[bool],
    pairs: List[Tuple[int, int]],
    out: List[FrozenSet[Tuple[int, int]]],
) -> None:
    if i == n:
        out.append(frozenset(pairs))
        return
    if used[i]:
        _enumerate(seq, i + 1, n, used, pairs, out)
        return
    _enumerate(seq, i + 1, n, used, pairs, out)  # i unpaired
    for t in range(i + MIN_HAIRPIN + 1, n):
        if used[t] or (seq[i], seq[t]) not in PAIR_SCORES:
            continue
        used[i] = used[t] = True
        pairs.append((i, t))
        _enumerate(seq, i + 1, n, used, pairs, out)
        pairs.pop()
        used[i] = used[t] = False


def _render(
    pairs: FrozenSet[Tuple[int, int]],
    n: int,
    stems: Optional[Tuple[set, set]],
) -> str:
    chars = ["."] * n
    stem2 = stems[1] if stems is not None else set()
    for x, y in sorted(pairs):
        if (x, y) in stem2:
            chars[x], chars[y] = "[", "]"
        else:
            chars[x], chars[y] = "(", ")"
    return "".join(chars)


def brute_force_pk(
    seq: str,
    forced_pairs: Optional[FrozenSet[Tuple[int, int]]] = None,
    forced_unpaired: Optional[frozenset] = None,
    penalty: int = 0,
) -> Dict[str, object]:
    """Exact reference optimum for the single-H pseudoknot mode."""
    forced_pairs = forced_pairs or frozenset()
    forced_unpaired = forced_unpaired or frozenset()
    n = len(seq)

    all_matchings: List[FrozenSet[Tuple[int, int]]] = []
    _enumerate(
        seq,
        0,
        n,
        [p in forced_unpaired for p in range(n)],
        [],
        all_matchings,
    )

    best_score: Optional[int] = None
    best_structs: List[str] = []
    all_pairs: set = set()
    exact_count = 0

    for pairs in all_matchings:
        result = analyze(pairs)
        if result is None:
            continue
        kind, stems = result
        if not all(p in pairs for p in forced_pairs):
            continue
        if forced_unpaired & {p for pr in pairs for p in pr}:
            continue
        score = sum(PAIR_SCORES[(seq[a], seq[b])] for a, b in pairs)
        if kind == "single-h":
            score -= penalty
        db = _render(pairs, n, stems)
        if best_score is None or score > best_score:
            best_score = score
            best_structs = [db]
            all_pairs = set(pairs)
        elif score == best_score:
            best_structs.append(db)
            all_pairs |= pairs
        exact_count += 1

    if best_score is None:
        return {
            "feasible": False,
            "score": None,
            "structure": None,
            "count": 0,
            "pairs": frozenset(),
            "exact_count": 0,
        }

    return {
        "feasible": True,
        "score": best_score,
        "structure": min(best_structs),
        "count": len(best_structs) % MOD,
        "exact_count": len(best_structs),
        "pairs": frozenset(all_pairs),
    }
