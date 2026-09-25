"""Independent brute-force reference for RNA folding (n <= 14).

This reference does NOT import or call the module under test.  It enumerates
every admissible non-crossing matching from scratch (recursive interval
decomposition), then keeps only matchings that contain all forced pairs, so
its expected values are derived independently of the production DP.
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

# A concrete structure: (score, dot-bracket string, frozenset of pairs).
Structure = Tuple[int, str, FrozenSet[Tuple[int, int]]]


def _enumerate(
    seq: str,
    force_unpaired: List[bool],
    unpaired_scores: List[int],
    stack_score: int,
) -> List[Structure]:
    """All legal non-crossing matchings, without forced-pair filtering."""
    n = len(seq)
    memo: Dict[Tuple[int, int], List[Structure]] = {}

    def go(i: int, j: int) -> List[Structure]:
        if i >= j:
            return [(0, "", frozenset())]
        key = (i, j)
        if key in memo:
            return memo[key]

        out: List[Structure] = []

        # Branch A: i unpaired (always legal; when i is forced unpaired it
        # is the only branch taken).
        for sc, s, pr in go(i + 1, j):
            out.append((sc + unpaired_scores[i], "." + s, pr))

        # Branch B: i paired with t.
        for t in range(i + MIN_HAIRPIN + 1, j):
            if force_unpaired[i] or force_unpaired[t]:
                continue
            w = PAIR_SCORES.get((seq[i], seq[t]))
            if w is None:
                continue
            for sc1, s1, p1 in go(i + 1, t):
                for sc2, s2, p2 in go(t + 1, j):
                    bonus = (
                        stack_score
                        if (i + 1, t - 1) in p1
                        else 0
                    )
                    out.append(
                        (
                            w + bonus + sc1 + sc2,
                            "(" + s1 + ")" + s2,
                            p1 | p2 | frozenset(((i, t),)),
                        )
                    )

        memo[key] = out
        return out

    return go(0, n)


def brute_force(
    seq: str,
    forced_pairs: Optional[FrozenSet[Tuple[int, int]]] = None,
    forced_unpaired: Optional[frozenset] = None,
    unpaired_scores: Optional[List[int]] = None,
    stack_score: int = 0,
) -> Dict[str, object]:
    """Exact reference optimum over structures satisfying all constraints.

    Forced pairs are enforced by *post-filtering* independently enumerated
    matchings, a deliberately different mechanism from the production code's
    closed-interval DP.
    """
    forced_pairs = forced_pairs or frozenset()
    forced_unpaired = forced_unpaired or frozenset()
    n = len(seq)
    up = [p in forced_unpaired for p in range(n)]
    if unpaired_scores is None:
        unpaired_scores = [0] * n

    required: Dict[int, int] = {}
    for a, b in forced_pairs:
        required[a] = b
        required[b] = a

    best_score: Optional[int] = None
    best_structs: List[Structure] = []
    exact_count = 0
    all_pairs: set = set()

    for sc, s, pr in _enumerate(seq, up, unpaired_scores, stack_score):
        ok = all(p in pr for p in forced_pairs)
        if not ok:
            continue
        exact_count += 1
        if best_score is None or sc > best_score:
            best_score = sc
            best_structs = [(sc, s, pr)]
        elif sc == best_score:
            best_structs.append((sc, s, pr))

    if best_score is None:
        return {
            "feasible": False,
            "score": None,
            "structure": None,
            "count": 0,
            "pairs": frozenset(),
            "exact_count": 0,
        }

    opt = [x for x in best_structs if x[0] == best_score]
    for _, _, pr in opt:
        all_pairs.update(pr)
    lex_min = min(s for _, s, _ in opt)
    opt_count = len(opt)

    return {
        "feasible": True,
        "score": best_score,
        "structure": lex_min,
        "count": opt_count % MOD,
        "exact_count": opt_count,
        "pairs": frozenset(all_pairs),
    }
