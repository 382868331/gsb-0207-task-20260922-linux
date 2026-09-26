"""RNA secondary structure prediction with forced pairings.

Single public entry point: :func:`fold`.

The implementation is an exact, non-crossing (pseudoknot-free) interval
dynamic program.  It performs an exhaustive search over every admissible
structure -- it never samples -- so a returned optimum is proven optimal and a
rejected constraint set is proven unsatisfiable.  See README.md for the full
contract.

An optional single-H-type pseudoknot mode is available: at most one pair of
crossing stems (``i < k < j < l``), each stem a stack of consecutively nested
pairs, and the three loop gaps folded by the original pseudoknot-free DP.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "fold",
    "InvalidSequenceError",
    "ConstraintError",
    "UnsatisfiableConstraintError",
    "BudgetExceededError",
    "MIN_HAIRPIN",
    "MAX_SEQUENCE_LENGTH",
    "MAX_PSEUDOKNOT_LENGTH",
    "PAIR_SCORES",
    "MOD",
]

MIN_HAIRPIN = 3
"""Minimum number of *unpaired* positions inside a pair (j - i >= 4)."""

MAX_SEQUENCE_LENGTH = 120

MAX_PSEUDOKNOT_LENGTH = 24
"""Sequence length cap when the single-H pseudoknot mode is enabled."""

MOD = 1_000_000_007

# Independent (symmetric) score table.  Anything absent is not a legal pair.
PAIR_SCORES: Dict[Tuple[str, str], int] = {
    ("G", "C"): 3,
    ("C", "G"): 3,
    ("A", "U"): 2,
    ("U", "A"): 2,
    ("G", "U"): 1,
    ("U", "G"): 1,
}

_ALPHABET = frozenset("ACGU")
_NEG = -10**18

# Bracket styles accepted when constraints are given as a string.
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}


class InvalidSequenceError(ValueError):
    """The RNA sequence is empty-of-contract (bad type/length/character)."""


class ConstraintError(ValueError):
    """Constraints are malformed, mutually conflicting, or impossible.

    The message always identifies the offending position (0-based) or the
    offending constraint entry.
    """


class UnsatisfiableConstraintError(ConstraintError):
    """No admissible structure can satisfy all forced constraints.

    This is a *proven* infeasibility: the exact DP found the restricted
    solution space to be empty.  It is distinct from :class:`BudgetExceededError`.
    """


class BudgetExceededError(RuntimeError):
    """Search budget exhausted before optimality could be proven.

    The current implementation is an exact O(n**3) dynamic program over an
    input capped at ``MAX_SEQUENCE_LENGTH``; it enumerates the whole solution
    space deterministically, so this exception is part of the fixed error
    semantics but is never raised by :func:`fold`.  It exists to keep
    "budget exhausted" permanently distinct from
    :class:`UnsatisfiableConstraintError` ("proven no solution").
    """


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _as_int(value: object, where: str) -> int:
    """Return ``value`` as int, rejecting bool and non-finite floats.

    Per the task-wide numeric contract: booleans are not integers, and
    NaN/Infinity are reported as such rather than coerced.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            if math.isnan(value):
                raise ValueError(f"{where}: NaN is not a valid integer")
            if math.isinf(value):
                raise ValueError(f"{where}: Infinity is not a valid integer")
            raise TypeError(f"{where}: expected integer, got float {value!r}")
        got = type(value).__name__
        raise TypeError(f"{where}: expected integer, got {got}")
    return value


def _validate_sequence(sequence: object, allow_pk: bool = False) -> str:
    if not isinstance(sequence, str):
        raise TypeError(
            f"sequence: expected str, got {type(sequence).__name__}"
        )
    n = len(sequence)
    cap = MAX_PSEUDOKNOT_LENGTH if allow_pk else MAX_SEQUENCE_LENGTH
    if n > cap:
        raise InvalidSequenceError(
            f"sequence length {n} exceeds maximum {cap}"
            + (" in pseudoknot mode" if allow_pk else "")
        )
    for i, ch in enumerate(sequence):
        if ch not in _ALPHABET:
            raise InvalidSequenceError(
                f"sequence position {i} (0-based): illegal character "
                f"{ch!r}; allowed characters are A, C, G, U"
            )
    return sequence


def _check_pair_geometry(a: int, b: int, seq: str, where: str) -> None:
    """Reject a forced pair that no admissible structure could contain.

    These are *hard* conflicts with the problem's pairing rules, reported at
    the offending constraint rather than silently ignored.  With every forced
    pair individually admissible, pairwise non-crossing and endpoint disjoint,
    the set is always jointly realizable; the DP keeps a further
    unsatisfiability guard as defense in depth.
    """
    n = len(seq)
    if not (0 <= a < b < n):
        raise ConstraintError(
            f"{where}=({a}, {b}): require 0 <= i < j < {n}"
        )
    if b - a <= MIN_HAIRPIN:
        raise ConstraintError(
            f"{where}=({a}, {b}): pairs enclose at least {MIN_HAIRPIN} "
            f"unpaired positions, so j - i must be >= {MIN_HAIRPIN + 1}"
        )
    if (seq[a], seq[b]) not in PAIR_SCORES:
        raise ConstraintError(
            f"{where}=({a}, {b}): {seq[a]!r}-{seq[b]!r} is not an allowed "
            f"pair (allowed: AU, UA, GC, CG, GU, UG)"
        )


def _normalise_constraints(
    constraints: object, seq: str, allow_pk: bool = False
) -> Tuple[List[int], List[int]]:
    """Validate every constraint form; return (partner, force_unpaired).

    ``partner[p]`` is the forced mate of p or -1; ``force_unpaired`` holds
    positions that must stay unpaired.
    """
    n = len(seq)
    partner = [-1] * n
    forced: List[Tuple[int, int]] = []
    unpaired: set[int] = set()

    if constraints is None:
        return partner, [0] * n

    pairs_iter: Iterable[object]
    unpaired_iter: Optional[Iterable[object]] = None

    if isinstance(constraints, str):
        if len(constraints) != n:
            raise ConstraintError(
                f"constraints string length {len(constraints)} != sequence "
                f"length {n}"
            )
        string_pairs: List[Tuple[int, int]] = []
        # One stack per bracket family: a closing bracket always matches
        # the most recent unmatched opening of its own family, so families
        # may cross each other (only meaningful in pseudoknot mode).
        stacks: Dict[str, List[int]] = {ch: [] for ch in _OPEN_TO_CLOSE}
        for i, ch in enumerate(constraints):
            if ch == "." or ch == "x":
                if ch == "x":
                    unpaired.add(i)
            elif ch in _OPEN_TO_CLOSE:
                stacks[ch].append(i)
            elif ch in _CLOSE_TO_OPEN:
                open_ch = _CLOSE_TO_OPEN[ch]
                if not stacks[open_ch]:
                    raise ConstraintError(
                        f"constraints position {i} (0-based): closing "
                        f"{ch!r} has no matching opening bracket"
                    )
                open_pos = stacks[open_ch].pop()
                string_pairs.append((open_pos, i))
            else:
                raise ConstraintError(
                    f"constraints position {i} (0-based): illegal character "
                    f"{ch!r}; allowed: . x ( ) [ ] {{ }} < >"
                )
        leftover = [
            (oc, pos)
            for oc, st in stacks.items()
            for pos in st
        ]
        if leftover:
            open_ch, open_pos = leftover[0]
            raise ConstraintError(
                f"constraints position {open_pos} (0-based): opening "
                f"{open_ch!r} is never closed"
            )
        pairs_iter = string_pairs
    elif isinstance(constraints, Mapping):
        unknown = set(constraints) - {"pairs", "unpaired"}
        if unknown:
            raise ConstraintError(
                f"constraints: unknown key(s) {sorted(unknown)!r}; allowed "
                f"keys are 'pairs' and 'unpaired'"
            )
        raw_pairs = constraints.get("pairs", ())
        raw_unpaired = constraints.get("unpaired", ())
        if isinstance(raw_pairs, (str, bytes)) or isinstance(
            raw_unpaired, (str, bytes)
        ):
            raise TypeError(
                "constraints: 'pairs' and 'unpaired' must be iterables of "
                "int pairs / ints, not strings"
            )
        try:
            pairs_iter = list(raw_pairs)
            unpaired_iter = list(raw_unpaired)
        except TypeError as exc:
            raise TypeError("constraints: value is not iterable") from exc
    else:
        raise TypeError(
            "constraints: expected None, a dot-bracket-like str, or a dict "
            f"with 'pairs'/'unpaired', got {type(constraints).__name__}"
        )

    for idx, item in enumerate(pairs_iter):
        where = f"pairs[{idx}]"
        if not isinstance(item, (list, tuple)) or isinstance(item, (bool, str)):
            raise TypeError(
                f"{where}: expected a (i, j) pair of integers, got "
                f"{type(item).__name__}"
            )
        if len(item) != 2:
            raise ConstraintError(
                f"{where}: pair must have exactly 2 elements, got {len(item)}"
            )
        a = _as_int(item[0], f"{where}[0]")
        b = _as_int(item[1], f"{where}[1]")
        _check_pair_geometry(a, b, seq, where)
        forced.append((a, b))

    if unpaired_iter is not None:
        for idx, item in enumerate(unpaired_iter):
            p = _as_int(item, f"unpaired[{idx}]")
            if not 0 <= p < n:
                raise ConstraintError(
                    f"unpaired[{idx}]={p}: position out of range "
                    f"[0, {n - 1}]"
                )
            unpaired.add(p)

    # Cross-validation: endpoints, duplicates, crossings.
    for idx, (a, b) in enumerate(forced):
        if a in unpaired or b in unpaired:
            raise ConstraintError(
                f"pairs[{idx}]=({a}, {b}): endpoint is also forced unpaired"
            )
        if partner[a] != -1:
            raise ConstraintError(
                f"pairs[{idx}]=({a}, {b}): position {a} is already used by "
                f"another forced pair"
            )
        if partner[b] != -1:
            raise ConstraintError(
                f"pairs[{idx}]=({a}, {b}): position {b} is already used by "
                f"another forced pair"
            )
        partner[a] = b
        partner[b] = a

    for idx, (a, b) in enumerate(forced):
        for jdx in range(idx + 1, len(forced)):
            c, d = forced[jdx]
            if (a, b) == (c, d):
                raise ConstraintError(
                    f"pairs[{jdx}]=({c}, {d}): duplicate of pairs[{idx}]"
                )
            # crossing: a < c < b < d or c < a < d < b.  In the default
            # mode any crossing is structurally illegal.  In the single-H
            # mode one crossing is in principle legal; the exact DP is the
            # single decision point and proves (un)satisfiability, including
            # the "no second pseudoknot" rule.
            if not allow_pk and ((a < c < b < d) or (c < a < d < b)):
                raise ConstraintError(
                    f"pairs[{idx}]=({a}, {b}) and pairs[{jdx}]=({c}, {d}) "
                    f"cross each other; pseudoknots are forbidden"
                )

    force_up = [0] * n
    for p in unpaired:
        force_up[p] = 1
    return partner, force_up


# ---------------------------------------------------------------------------
# Core dynamic program
# ---------------------------------------------------------------------------


def _iter_h_cores(
    a: int,
    j: int,
    seq: str,
    partner: List[int],
    force_up: List[int],
    closed: List[List[bool]],
    best0: List[List[int]],
    ways0: List[List[int]],
):
    """Yield every single-H pseudoknot core anchored with outer pair (a, b).

    Stem 1 is ``(a,b),(a+1,b-1),...`` and stem 2 is ``(c,d),(c+1,d-1),...``
    with ``a < c < b < d`` for the outer pairs.  The two stems have equal
    height (every layer crosses).  Each yielded tuple is
    ``(c, b, d, p, q, score, gap_ways)`` where ``p``/``q`` are the stack
    heights, ``score`` is the exact best score of the core (stack pairs plus
    the three loop gaps folded by the pseudoknot-free DP; no penalty charged
    here), and ``gap_ways`` is the product of the optimal-structure counts
    of the three gaps.
    """
    n = len(seq)
    # (a, b): outermost pair of stem 1.
    for b in range(a + MIN_HAIRPIN + 1, j):
        if force_up[a] or force_up[b]:
            continue
        if partner[a] != -1 and partner[a] != b:
            continue
        if partner[b] != -1 and partner[b] != a:
            continue
        if (seq[a], seq[b]) not in PAIR_SCORES:
            continue
        # (c, d): outermost pair of stem 2, a < c < b < d < j.
        for c in range(a + 1, b):
            for d in range(max(b + 1, c + MIN_HAIRPIN + 1), j):
                if force_up[c] or force_up[d]:
                    continue
                if partner[c] != -1 and partner[c] != d:
                    continue
                if partner[d] != -1 and partner[d] != c:
                    continue
                if (seq[c], seq[d]) not in PAIR_SCORES:
                    continue
                # Grow the two stacks; in the strict four-block H layout
                # both stems have the same height (every layer crosses).
                feas1 = {1: PAIR_SCORES[(seq[a], seq[b])]}
                p = 1
                while True:
                    x = a + p
                    y = b - p
                    if y - x < MIN_HAIRPIN + 1:
                        break
                    if force_up[x] or force_up[y]:
                        break
                    if partner[x] != -1 and partner[x] != y:
                        break
                    if partner[y] != -1 and partner[y] != x:
                        break
                    w = PAIR_SCORES.get((seq[x], seq[y]))
                    if w is None:
                        break
                    feas1[p + 1] = feas1[p] + w
                    p += 1
                feas2 = {1: PAIR_SCORES[(seq[c], seq[d])]}
                q = 1
                while True:
                    x = c + q
                    y = d - q
                    if y - x < MIN_HAIRPIN + 1:
                        break
                    if force_up[x] or force_up[y]:
                        break
                    if partner[x] != -1 and partner[x] != y:
                        break
                    if partner[y] != -1 and partner[y] != x:
                        break
                    w = PAIR_SCORES.get((seq[x], seq[y]))
                    if w is None:
                        break
                    feas2[q + 1] = feas2[q] + w
                    q += 1

                for height in set(feas1) & set(feas2):
                    p = q = height
                    w1, w2 = feas1[height], feas2[height]
                    # Innermost endpoints.
                    i1, j1 = a + p - 1, b - p + 1
                    i2, j2 = c + q - 1, d - q + 1
                    # The four stack blocks occupy disjoint, ordered
                    # ranges; loop gaps may be empty.  Equal heights make
                    # every layer cross: [a,i1],[c,i2],[j1,b],[j2,d].
                    if not (i1 < c and i2 <= j1 and b < j2):
                        continue
                    g1, g2, g3 = (
                        (i1 + 1, c),
                        (i2 + 1, j1),
                        (b + 1, j2),
                    )
                    if (
                        not closed[g1[0]][g1[1]]
                        or not closed[g2[0]][g2[1]]
                        or not closed[g3[0]][g3[1]]
                    ):
                        continue
                    s1 = best0[g1[0]][g1[1]]
                    s2 = best0[g2[0]][g2[1]]
                    s3 = best0[g3[0]][g3[1]]
                    if _NEG in (s1, s2, s3):
                        continue
                    vcore = w1 + w2 + s1 + s2 + s3
                    wcore = (
                        ways0[g1[0]][g1[1]]
                        * ways0[g2[0]][g2[1]]
                        % MOD
                        * ways0[g3[0]][g3[1]]
                        % MOD
                    )
                    yield (c, b, d, p, q, vcore, wcore)


def _h_gap_subintervals(
    a: int,
    b: int,
    c: int,
    d: int,
    p: int,
    q: int,
    j: int,
) -> Tuple[Tuple[int, int, bool], ...]:
    """Pseudoknot-free sub-intervals covering all non-stem positions."""
    i1, j1 = a + p - 1, b - p + 1
    i2, j2 = c + q - 1, d - q + 1
    return (
        (i1 + 1, c, False),    # L1
        (i2 + 1, j1, False),   # L2
        (b + 1, j2, False),    # L3
        (d + 1, j, False),     # trailing region
    )


def _h_core_pair_list(
    a: int, b: int, c: int, d: int, p: int, q: int
) -> List[Tuple[int, int]]:
    """All stack pairs of an H core, both stems outer-to-inner."""
    out: List[Tuple[int, int]] = []
    for h in range(p):
        out.append((a + h, b - h))
    for h in range(q):
        out.append((c + h, d - h))
    return out


def _render_h_core(
    a: int,
    b: int,
    c: int,
    d: int,
    p: int,
    q: int,
    j: int,
    lex0: List[List[str]],
) -> str:
    """Extended dot-bracket witness for an anchored H core plus its tail.

    Stem 1 uses ``()`` and stem 2 uses ``[]`` so the two crossing stems are
    explicitly marked.  Loop gaps and the trailing region are filled with
    the lexicographically smallest pseudoknot-free structures.
    """
    i1, j1 = a + p - 1, b - p + 1
    i2, j2 = c + q - 1, d - q + 1
    chars = ["."] * (j - a)

    for h in range(p):
        chars[a + h - a] = "("
        chars[b - h - a] = ")"
    for h in range(q):
        chars[c + h - a] = "["
        chars[d - h - a] = "]"

    for lo, hi, piece in (
        (i1 + 1, c, lex0[i1 + 1][c]),
        (i2 + 1, j1, lex0[i2 + 1][j1]),
        (b + 1, j2, lex0[b + 1][j2]),
        (d + 1, j, lex0[d + 1][j]),
    ):
        for off, ch in enumerate(piece):
            chars[lo - a + off] = ch
    return "".join(chars)


def _pseudoknot_info(structure: str) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Extract the two crossing stems from an extended dot-bracket string.

    The ``()`` family is reused for ordinary pairs, so the two stems are
    recovered from the unique crossing rather than from bracket families.
    """
    round_stack: List[int] = []
    square_stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(structure):
        if ch == "(":
            round_stack.append(i)
        elif ch == ")":
            pairs.append((round_stack.pop(), i))
        elif ch == "[":
            square_stack.append(i)
        elif ch == "]":
            pairs.append((square_stack.pop(), i))

    ordered = sorted(pairs)
    crossing = [
        (ordered[x], ordered[y])
        for x in range(len(ordered))
        for y in range(x + 1, len(ordered))
        if ordered[x][0] < ordered[y][0] < ordered[x][1] < ordered[y][1]
    ]
    (a, b), (c, d) = min(crossing, key=lambda z: (z[0][0], z[1][0]))
    stem1_pairs = [
        (x, y) for x, y in ordered if a <= x and y <= b and (x - a) == (b - y)
    ]
    stem2_pairs = [
        (x, y) for x, y in ordered if c <= x and y <= d and (x - c) == (d - y)
    ]
    return {
        "stem1": tuple(sorted(stem1_pairs)),
        "stem2": tuple(sorted(stem2_pairs)),
    }


def fold(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    *,
    allow_pseudoknot: bool = False,
    pseudoknot_penalty: int = 0,
) -> Dict[str, object]:
    """Fold ``sequence`` into a maximum-score secondary structure.

    :param sequence: RNA sequence over ``A C G U`` (case sensitive),
        length <= 120 by default (length <= 24 when
        ``allow_pseudoknot=True``).
    :param constraints: ``None``, or either

        * a string of the sequence length: ``.`` free, ``x`` forced
          unpaired, and matched brackets ``() [] {} <>`` forcing the paired
          positions; or
        * a dict with ``"pairs"``: iterable of ``(i, j)`` integer pairs and
          ``"unpaired"``: iterable of integer positions (all 0-based).

        Forced pairs must be complementary, satisfy ``j - i >= 4`` and share
        no endpoint.  By default they must not cross; when
        ``allow_pseudoknot=True`` a crossing consistent with at most one
        single-H pseudoknot is admitted, and constraints requiring more than
        one crossing are rejected as unsatisfiable.
    :param allow_pseudoknot: when ``True``, admit at most one single-H-type
        pseudoknot (two crossing stems, ``i < k < j < l``); the
        pseudoknot-free optimum always remains a candidate.
    :param pseudoknot_penalty: non-negative integer subtracted once from the
        score of structures containing the pseudoknot.
    :returns: dict with

        * ``score`` (int): maximum total score (GC=3, AU=2, GU=1);
        * ``structure`` (str): lexicographically smallest dot-bracket string
          among all maximum-score structures (ASCII order, so ``'(' < ')' <
          '.'``).  In pseudoknot mode the two crossing stems are marked with
          two bracket families, ``()`` and ``[]`` (stem 1 = the stem whose
          outer left endpoint is smaller); structures without a pseudoknot
          still use only ``()``.
        * ``count`` (int): number of maximum-score structures, modulo
          1_000_000_007;
        * ``pairs`` (tuple of ``(i, j)``): every pair that occurs in *at
          least one* maximum-score structure, sorted;
        * ``pseudoknot``: ``None`` for a pseudoknot-free optimum, otherwise a
          dict ``{"stem1": ((i, j), ...), "stem2": ((k, l), ...)}`` with
          the two crossing stems ordered outer-to-inner.
    :raises TypeError: wrong value types (including ``bool`` as int).
    :raises InvalidSequenceError: bad length or illegal character.
    :raises ConstraintError: malformed or conflicting constraints.
    :raises UnsatisfiableConstraintError: proven that no admissible structure
        satisfies the forced constraints.
    """

    if not isinstance(allow_pseudoknot, bool):
        raise TypeError(
            "allow_pseudoknot: expected bool, got "
            f"{type(allow_pseudoknot).__name__}"
        )
    penalty = _as_int(pseudoknot_penalty, "pseudoknot_penalty")
    if penalty < 0:
        raise ValueError(
            f"pseudoknot_penalty={penalty}: must be a non-negative integer"
        )

    seq = _validate_sequence(sequence, allow_pseudoknot)
    partner, force_up = _normalise_constraints(
        constraints, seq, allow_pseudoknot
    )
    n = len(seq)

    # A *closed* interval [i, j) is one where every position p in [i, j)
    # with a forced mate also has that mate inside [i, j).  The DP only
    # considers closed intervals, so any candidate pair (i, t) that crossed a
    # forced pair (a, b) -- a < i < b < t or i < a < t < b -- would split an
    # endpoint from its mate and is automatically unreachable.
    # closed[i][j] is maintained as: (forced endpoints in interval) ==
    # 2 * (forced pairs fully inside the interval).
    closed: List[List[bool]] = [
        [False] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        endpoint_count = 0
        inside_pairs = 0
        closed[i][i] = True
        for j in range(i + 1, n + 1):
            p = j - 1
            if partner[p] != -1:
                endpoint_count += 1
                m = partner[p]
                if i <= m < p:  # the right endpoint of a pair fully inside
                    inside_pairs += 1
            closed[i][j] = endpoint_count == 2 * inside_pairs

    def pair_allowed(i: int, t: int) -> bool:
        if force_up[i] or force_up[t]:
            return False
        if partner[i] != -1 and partner[i] != t:
            return False
        if partner[t] != -1 and partner[t] != i:
            return False
        if (seq[i], seq[t]) not in PAIR_SCORES:
            return False
        return True

    # Half-open intervals [i, j), 0 <= i <= j <= n.  best0/ways0 enumerate
    # structures with zero pseudoknots inside the interval; best1/ways1
    # enumerate structures with exactly one single-H pseudoknot.
    best0: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    ways0: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    best1: List[List[int]] = [[_NEG] * (n + 1) for _ in range(n + 1)]
    ways1: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        ways0[i][i] = 1

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            top0 = _NEG
            total = 0
            top1 = _NEG
            total1 = 0

            # Option 1: position i stays unpaired.
            if partner[i] == -1 and closed[i + 1][j]:
                v = best0[i + 1][j]
                w = ways0[i + 1][j]
                if v > top0:
                    top0, total = v, w
                elif v == top0:
                    total = (total + w) % MOD
                v1 = best1[i + 1][j]
                w1 = ways1[i + 1][j]
                if v1 != _NEG:
                    if v1 > top1:
                        top1, total1 = v1, w1
                    elif v1 == top1:
                        total1 = (total1 + w1) % MOD

            # Option 2: i pairs with t (a non-crossing pair).
            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t):
                    continue
                if not closed[i + 1][t] or not closed[t + 1][j]:
                    continue
                wgt = PAIR_SCORES[(seq[i], seq[t])]
                bi, bo = best0[i + 1][t], best0[t + 1][j]
                if bi != _NEG and bo != _NEG:
                    v = wgt + bi + bo
                    w = (ways0[i + 1][t] * ways0[t + 1][j]) % MOD
                    if v > top0:
                        top0, total = v, w
                    elif v == top0:
                        total = (total + w) % MOD
                # Exactly one pseudoknot in either sub-interval.
                for (v_in, w_in), (v_out, w_out) in (
                    ((best1[i + 1][t], ways1[i + 1][t]),
                     (best0[t + 1][j], ways0[t + 1][j])),
                    ((best0[i + 1][t], ways0[i + 1][t]),
                     (best1[t + 1][j], ways1[t + 1][j])),
                ):
                    if v_in == _NEG or v_out == _NEG:
                        continue
                    v = wgt + v_in + v_out
                    w = (w_in * w_out) % MOD
                    if v > top1:
                        top1, total1 = v, w
                    elif v == top1:
                        total1 = (total1 + w) % MOD

            # Option 3 (pseudoknot mode only): the unique single-H pseudo-
            # knot is anchored exactly at this interval's left edge (i is
            # the outer left endpoint of stem 1).  Leading/trailing free
            # regions are absorbed recursively by options 1 and 2, so every
            # crossing configuration is enumerated exactly once here and a
            # second crossing can never appear.
            if allow_pseudoknot:
                for c, b, d, p, q, vcore, wcore in _iter_h_cores(
                    i, j, seq, partner, force_up, closed, best0, ways0
                ):
                    v_tail = best0[d + 1][j]
                    if v_tail == _NEG or not closed[d + 1][j]:
                        continue
                    v = vcore + v_tail - penalty
                    w = (wcore * ways0[d + 1][j]) % MOD
                    if v > top1:
                        top1, total1 = v, w
                    elif v == top1:
                        total1 = (total1 + w) % MOD

            best0[i][j] = top0
            ways0[i][j] = total
            best1[i][j] = top1
            ways1[i][j] = total1

    root_score = best0[0][n]
    root_pk = False
    if allow_pseudoknot and best1[0][n] > root_score:
        root_score = best1[0][n]
        root_pk = True

    if root_score == _NEG:
        raise UnsatisfiableConstraintError(
            "constraints are unsatisfiable: no admissible structure contains "
            "all forced pairs"
        )

    # --- Reconstruction -------------------------------------------------
    # Choices inside a zero-pseudoknot interval: each candidate is a
    # (score, string, sub-intervals to propagate) tuple.  Enumerating the
    # same candidates three times (score DP / lex / reachability) keeps the
    # three phases independent, matching the original implementation.
    def choices0(i: int, j: int):
        if partner[i] == -1 and closed[i + 1][j]:
            yield (
                best0[i + 1][j],
                "." + lex0[i + 1][j],
                ((i + 1, j, False),),
                (i, -1),
                None,
            )
        for t in range(i + MIN_HAIRPIN + 1, j):
            if not pair_allowed(i, t):
                continue
            if not closed[i + 1][t] or not closed[t + 1][j]:
                continue
            s_in = best0[i + 1][t]
            s_out = best0[t + 1][j]
            if s_in == _NEG or s_out == _NEG:
                continue
            yield (
                PAIR_SCORES[(seq[i], seq[t])] + s_in + s_out,
                "(" + lex0[i + 1][t] + ")" + lex0[t + 1][j],
                ((i + 1, t, False), (t + 1, j, False)),
                (i, t),
                None,
            )

    def choices1(i: int, j: int):
        # Exactly one pseudoknot: i unpaired, or i paired (pk in either
        # side), or the pseudoknot anchored at i (option 3).
        if partner[i] == -1 and closed[i + 1][j] and best1[i + 1][j] != _NEG:
            yield (
                best1[i + 1][j],
                "." + lex1[i + 1][j],
                ((i + 1, j, True),),
                (i, -1),
                None,
            )
        for t in range(i + MIN_HAIRPIN + 1, j):
            if not pair_allowed(i, t):
                continue
            if not closed[i + 1][t] or not closed[t + 1][j]:
                continue
            wgt = PAIR_SCORES[(seq[i], seq[t])]
            # pseudoknot inside (i+1, t)
            v_in, v_out = best1[i + 1][t], best0[t + 1][j]
            if v_in != _NEG and v_out != _NEG:
                yield (
                    wgt + v_in + v_out,
                    "(" + lex1[i + 1][t] + ")" + lex0[t + 1][j],
                    ((i + 1, t, True), (t + 1, j, False)),
                    (i, t),
                    None,
                )
            # pseudoknot inside (t+1, j)
            v_in, v_out = best0[i + 1][t], best1[t + 1][j]
            if v_in != _NEG and v_out != _NEG:
                yield (
                    wgt + v_in + v_out,
                    "(" + lex0[i + 1][t] + ")" + lex1[t + 1][j],
                    ((i + 1, t, False), (t + 1, j, True)),
                    (i, t),
                    None,
                )
        if allow_pseudoknot:
            for c, b, d, p, q, vcore, _wcore in _iter_h_cores(
                i, j, seq, partner, force_up, closed, best0, ways0
            ):
                v_tail = best0[d + 1][j]
                if v_tail == _NEG or not closed[d + 1][j]:
                    continue
                yield (
                    vcore + v_tail - penalty,
                    _render_h_core(
                        i, b, c, d, p, q, j, lex0
                    ),
                    _h_gap_subintervals(i, b, c, d, p, q, j),
                    (i, b),
                    (b, c, d, p, q),
                )

    # Lexicographically smallest optimal structure per interval (pk-free).
    lex: List[List[str]] = [[""] * (n + 1) for _ in range(n + 1)]
    lex0 = lex
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            if best0[i][j] == _NEG:
                continue  # infeasible interval: no witness string
            target = best0[i][j]
            chosen: Optional[str] = None

            for choice in choices0(i, j):
                _sc, candidate = choice[0], choice[1]
                if _sc == target and (chosen is None or candidate < chosen):
                    chosen = candidate

            assert chosen is not None and len(chosen) == length
            lex[i][j] = chosen

    # Lexicographically smallest optimal structure with exactly one pk.
    lex1: List[List[Optional[str]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    if allow_pseudoknot:
        for length in range(1, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                if best1[i][j] == _NEG:
                    continue
                chosen1: Optional[str] = None
                for choice in choices1(i, j):
                    _sc, candidate = choice[0], choice[1]
                    if _sc == best1[i][j] and (
                        chosen1 is None or candidate < chosen1
                    ):
                        chosen1 = candidate
                assert chosen1 is not None and len(chosen1) == length
                lex1[i][j] = chosen1

    # Reachability of (interval, has-pk) states along optimal decisions.
    reach0 = [[False] * (n + 1) for _ in range(n + 1)]
    reach1 = [[False] * (n + 1) for _ in range(n + 1)]
    reach0[0][n] = best0[0][n] == root_score
    reach1[0][n] = (
        allow_pseudoknot and best1[0][n] == root_score
    )
    can_pair: set[Tuple[int, int]] = set()

    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length
            if reach0[i][j]:
                for choice in choices0(i, j):
                    if choice[0] != best0[i][j]:
                        continue
                    pr, subs = choice[3], choice[2]
                    if pr[1] != -1:
                        can_pair.add(pr)
                    for x, y, has_pk in subs:
                        (reach1 if has_pk else reach0)[x][y] = True
            if allow_pseudoknot and reach1[i][j]:
                for choice in choices1(i, j):
                    if choice[0] != best1[i][j]:
                        continue
                    pr, subs, desc = choice[3], choice[2], choice[4]
                    if pr[1] != -1:
                        can_pair.add(pr)
                    for x, y, has_pk in subs:
                        (reach1 if has_pk else reach0)[x][y] = True
                    if desc is not None:
                        for sx, sy in _h_core_pair_list(i, *desc):
                            can_pair.add((sx, sy))

    structure = lex0[0][n]
    root_is_pk = False
    if (
        allow_pseudoknot
        and best1[0][n] == root_score
        and lex1[0][n] is not None
        and lex1[0][n] < structure
    ):
        structure = lex1[0][n]
        root_is_pk = True
    if best1[0][n] == root_score and best0[0][n] != root_score:
        structure = lex1[0][n]
        root_is_pk = True
    total_count = ways1[0][n] if root_is_pk else ways0[0][n]
    if allow_pseudoknot and best1[0][n] == best0[0][n]:
        total_count = (ways0[0][n] + ways1[0][n]) % MOD

    pk_info = _pseudoknot_info(structure) if root_is_pk else None

    return {
        "score": root_score,
        "structure": structure,
        "count": total_count,
        "pairs": tuple(sorted(can_pair)),
        "pseudoknot": pk_info,
    }
