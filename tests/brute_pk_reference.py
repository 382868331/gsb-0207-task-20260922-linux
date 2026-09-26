"""Independent brute-force reference for the single H-type pseudoknot mode.

This reference does NOT import or call the module under test.  It enumerates
every matching (set of pairs) for n <= ~14 from scratch via bitmask
recursion, keeps only matchings whose topology is either non-crossing or a
single H-type pseudoknot (two consecutive nested stems interleaved as
i < k < j < l), applies forced constraints by post-filtering, and scores
pseudoknot structures with the caller's penalty.  Expected values are
therefore derived independently of the production DP + stem enumeration.
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


def _cross(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    a, b = p1
    c, d = p2
    return (a < c < b < d) or (c < a < d < b)


def _is_stem(group: List[Tuple[int, int]]) -> bool:
    """Consecutive nested pairs: (lo, hi), (lo+1, hi-1), ..."""
    group = sorted(group)
    for x in range(len(group) - 1):
        a, b = group[x]
        c, d = group[x + 1]
        if c != a + 1 or d != b - 1:
            return False
    return True


def topology_has_single_h_pseudoknot(
    pairs: FrozenSet[Tuple[int, int]],
) -> Optional[bool]:
    """None if the topology is inadmissible, else True iff it has one H-pk."""
    plist = sorted(pairs)
    crossing = set()
    for x in range(len(plist)):
        for y in range(x + 1, len(plist)):
            if _cross(plist[x], plist[y]):
                crossing.add(plist[x])
                crossing.add(plist[y])
    if not crossing:
        return False  # plain non-crossing structure

    # Every pair involved in a crossing must belong to one of exactly two
    # stems; in an H-pseudoknot every stem1 pair crosses every stem2 pair.
    clist = sorted(crossing)
    first = clist[0]
    stem1 = {p for p in clist if not _cross(p, first)}
    stem2 = set(clist) - stem1
    if not stem1 or not stem2:
        return None
    for p in stem1:
        for q in stem2:
            if not _cross(p, q):
                return None
    if not _is_stem(sorted(stem1)) or not _is_stem(sorted(stem2)):
        return None
    # Interleaving i < k < j < l of the stem endpoints.
    s1 = sorted(stem1)
    s2 = sorted(stem2)
    i, j = s1[0][0], s1[0][1]
    k, l = s2[0][0], s2[0][1]
    if not (i < k < j < l):
        return None
    return True


def _enumerate_matchings(
    seq: str, force_unpaired: List[bool]
) -> List[FrozenSet[Tuple[int, int]]]:
    """All matchings (any topology) respecting pair legality."""
    n = len(seq)
    memo: Dict[int, List[FrozenSet[Tuple[int, int]]]] = {}

    def go(mask: int) -> List[FrozenSet[Tuple[int, int]]]:
        if mask == 0:
            return [frozenset()]
        got = memo.get(mask)
        if got is not None:
            return got
        p = (mask & -mask).bit_length() - 1
        out: List[FrozenSet[Tuple[int, int]]] = []
        # p stays unpaired.
        out.extend(go(mask & ~(1 << p)))
        # p pairs with q.
        if not force_unpaired[p]:
            rest = mask & ~(1 << p)
            m = rest
            while m:
                q = (m & -m).bit_length() - 1
                m &= m - 1
                if q - p < MIN_HAIRPIN + 1 or force_unpaired[q]:
                    continue
                if (seq[p], seq[q]) not in PAIR_SCORES:
                    continue
                for sub in go(rest & ~(1 << q)):
                    out.append(sub | frozenset(((p, q),)))
        memo[mask] = out
        return out

    return go((1 << n) - 1)


def _render(seq: str, pairs: FrozenSet[Tuple[int, int]], has_pk: bool) -> str:
    """Dot-bracket; with a pseudoknot the second stem uses []."""
    chars = ["."] * len(seq)
    if not has_pk:
        for a, b in pairs:
            chars[a] = "("
            chars[b] = ")"
        return "".join(chars)
    crossing = set()
    plist = sorted(pairs)
    for x in range(len(plist)):
        for y in range(x + 1, len(plist)):
            if _cross(plist[x], plist[y]):
                crossing.add(plist[x])
                crossing.add(plist[y])
    first = min(crossing)
    stem1 = {p for p in crossing if not _cross(p, first)}
    stem2 = set(crossing) - stem1
    # stem1 is the stem whose leftmost endpoint comes first (i < k).
    if min(p[0] for p in stem1) > min(p[0] for p in stem2):
        stem1, stem2 = stem2, stem1
    for a, b in pairs:
        if (a, b) in stem1:
            chars[a] = "("
            chars[b] = ")"
        elif (a, b) in stem2:
            chars[a] = "["
            chars[b] = "]"
        else:
            chars[a] = "("
            chars[b] = ")"
    return "".join(chars)


def brute_force_pk(
    seq: str,
    penalty: int,
    forced_pairs: Optional[FrozenSet[Tuple[int, int]]] = None,
    forced_unpaired: Optional[frozenset] = None,
) -> Dict[str, object]:
    """Exact reference optimum over non-crossing and single-H-pk structures."""
    forced_pairs = forced_pairs or frozenset()
    forced_unpaired = forced_unpaired or frozenset()
    n = len(seq)
    up = [p in forced_unpaired for p in range(n)]

    best_score: Optional[int] = None
    best_structs: List[Tuple[str, FrozenSet[Tuple[int, int]]]] = []
    all_pairs: set = set()

    for pr in _enumerate_matchings(seq, up):
        if not all(p in pr for p in forced_pairs):
            continue
        topo = topology_has_single_h_pseudoknot(pr)
        if topo is None:
            continue  # arbitrary crossing nets are not admissible
        sc = sum(PAIR_SCORES[(seq[a], seq[b])] for a, b in pr)
        if topo:
            sc -= penalty
        if best_score is None or sc > best_score:
            best_score = sc
            best_structs = [(_render(seq, pr, topo), pr)]
        elif sc == best_score:
            best_structs.append((_render(seq, pr, topo), pr))

    if best_score is None:
        return {
            "feasible": False,
            "score": None,
            "structure": None,
            "exact_count": 0,
            "pairs": frozenset(),
        }

    for _, pr in best_structs:
        all_pairs.update(pr)
    return {
        "feasible": True,
        "score": best_score,
        "structure": min(s for s, _ in best_structs),
        "exact_count": len(best_structs),
        "pairs": frozenset(all_pairs),
    }
