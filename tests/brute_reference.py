"""Independent brute-force reference for RNA folding (n <= 14).

This reference does NOT import or call the module under test.  It enumerates
every admissible non-crossing matching from scratch (recursive interval
decomposition), then keeps only matchings that contain all forced pairs, so
its expected values are derived independently of the production DP.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

PAIR_SCORES = {
    ("G", "C"): 3,
    ("C", "G"): 3,
    ("A", "U"): 2,
    ("U", "A"): 2,
    ("G", "U"): 1,
    ("U", "G"): 1,
}

DEFAULT_PAIR_ENERGIES = {
    "GC": -3,
    "CG": -3,
    "AU": -2,
    "UA": -2,
    "GU": -1,
    "UG": -1,
}

MIN_HAIRPIN = 3
MOD = 1_000_000_007

# A concrete structure: (score, dot-bracket string, frozenset of pairs).
Structure = Tuple[int, str, FrozenSet[Tuple[int, int]]]


def _enumerate(
    seq: str, force_unpaired: List[bool]
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
            out.append((sc, "." + s, pr))

        # Branch B: i paired with t.
        for t in range(i + MIN_HAIRPIN + 1, j):
            if force_unpaired[i] or force_unpaired[t]:
                continue
            w = PAIR_SCORES.get((seq[i], seq[t]))
            if w is None:
                continue
            for sc1, s1, p1 in go(i + 1, t):
                for sc2, s2, p2 in go(t + 1, j):
                    out.append(
                        (
                            w + sc1 + sc2,
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

    required: Dict[int, int] = {}
    for a, b in forced_pairs:
        required[a] = b
        required[b] = a

    best_score: Optional[int] = None
    best_structs: List[Structure] = []
    exact_count = 0
    all_pairs: set = set()

    for sc, s, pr in _enumerate(seq, up):
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


# ---------------------------------------------------------------------------
# Independent partition-function reference: enumerate every structure as a
# frozenset of pairs (same recursive interval split as the score reference),
# then sum Boltzmann weights directly.  No log-space DP is used here, so the
# expected marginals come from a genuinely different computation.
# ---------------------------------------------------------------------------

_LEGAL = frozenset(DEFAULT_PAIR_ENERGIES)


def _all_structures(
    seq: str,
    force_unpaired: List[bool],
    i: int = 0,
    j: Optional[int] = None,
    memo: Optional[Dict[Tuple[int, int], List[FrozenSet[Tuple[int, int]]]]] = None,
) -> List[FrozenSet[Tuple[int, int]]]:
    if j is None:
        j = len(seq)
        memo = {}
    key = (i, j)
    if key in memo:
        return memo[key]
    out: List[FrozenSet[Tuple[int, int]]] = []
    if i < j:
        # i unpaired
        for rest in _all_structures(seq, force_unpaired, i + 1, j, memo):
            out.append(rest)
        # i paired with t
        for t in range(i + MIN_HAIRPIN + 1, j):
            if force_unpaired[i] or force_unpaired[t]:
                continue
            if seq[i] + seq[t] not in _LEGAL:
                continue
            for inner in _all_structures(seq, force_unpaired, i + 1, t, memo):
                for outer in _all_structures(seq, force_unpaired, t + 1, j, memo):
                    out.append(inner | outer | frozenset(((i, t),)))
    else:
        out.append(frozenset())
    memo[key] = out
    return out


def brute_partition(
    seq: str,
    temperature: float = 1.0,
    pair_energies: Optional[Mapping[str, int]] = None,
    unpaired_energy: int = 0,
    forced_pairs: Optional[FrozenSet[Tuple[int, int]]] = None,
    forced_unpaired: Optional[frozenset] = None,
) -> Dict[str, object]:
    """Exact Boltzmann reference by direct enumeration of structures."""
    energies = dict(DEFAULT_PAIR_ENERGIES)
    if pair_energies:
        energies.update(pair_energies)
    forced_pairs = forced_pairs or frozenset()
    forced_unpaired = forced_unpaired or frozenset()
    n = len(seq)
    up = [p in forced_unpaired for p in range(n)]

    log_terms: List[Tuple[float, FrozenSet[Tuple[int, int]]]] = []
    for pairs in _all_structures(seq, up):
        if not all(p in pairs for p in forced_pairs):
            continue
        energy = unpaired_energy * (n - 2 * len(pairs))
        for a, b in pairs:
            energy += energies[seq[a] + seq[b]]
        log_terms.append((-energy / temperature, pairs))

    # Direct log-sum-exp, reference style (small inputs only).
    m = max(lw for lw, _ in log_terms)
    z = math.fsum(math.exp(lw - m) for lw, _ in log_terms)
    ln_z = m + math.log(z)

    unpaired = [0.0] * n
    marginals: Dict[Tuple[int, int], float] = {}
    for lw, pairs in log_terms:
        w = math.exp(lw - m) / z  # probability of this structure
        paired_positions: set = set()
        for a, b in pairs:
            marginals[(a, b)] = marginals.get((a, b), 0.0) + w
            paired_positions.add(a)
            paired_positions.add(b)
        for k in range(n):
            if k not in paired_positions:
                unpaired[k] += w

    min_energy = min(
        unpaired_energy * (n - 2 * len(pairs))
        + sum(energies[seq[a] + seq[b]] for a, b in pairs)
        for _, pairs in log_terms
    )
    return {
        "ln_z": ln_z,
        "min_energy": min_energy,
        "unpaired": tuple(unpaired),
        "pair_probabilities": marginals,
        "structure_count": len(log_terms),
    }
