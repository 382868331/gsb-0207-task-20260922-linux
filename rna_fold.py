"""RNA secondary structure prediction with forced pairings and extra scores.

Single public entry point: :func:`fold`.

The implementation is an exact, non-crossing (pseudoknot-free) interval
dynamic program.  It performs an exhaustive search over every admissible
structure -- it never samples -- so a returned optimum is proven optimal and a
rejected constraint set is proven unsatisfiable.  See README.md for the full
contract.

Scoring (all integer, all optional additions default to zero):

* per-base-pair scores (GC=3, AU=2, GU=1);
* per-position unpaired scores (each unpaired position contributes its own
  configured integer value);
* a stacking bonus added once for every directly nested adjacent pair
  (i, j) + (i+1, j-1) present in the structure.
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
    "PAIR_SCORES",
    "MOD",
]

MIN_HAIRPIN = 3
"""Minimum number of *unpaired* positions inside a pair (j - i >= 4)."""

MAX_SEQUENCE_LENGTH = 120

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


def _validate_sequence(sequence: object) -> str:
    if not isinstance(sequence, str):
        raise TypeError(
            f"sequence: expected str, got {type(sequence).__name__}"
        )
    n = len(sequence)
    if n > MAX_SEQUENCE_LENGTH:
        raise InvalidSequenceError(
            f"sequence length {n} exceeds maximum {MAX_SEQUENCE_LENGTH}"
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
    constraints: object, seq: str
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
        stack: List[Tuple[str, int]] = []
        for i, ch in enumerate(constraints):
            if ch == "." or ch == "x":
                if ch == "x":
                    unpaired.add(i)
            elif ch in _OPEN_TO_CLOSE:
                stack.append((ch, i))
            elif ch in _CLOSE_TO_OPEN:
                if not stack:
                    raise ConstraintError(
                        f"constraints position {i} (0-based): closing "
                        f"{ch!r} has no matching opening bracket"
                    )
                open_ch, open_pos = stack.pop()
                if _CLOSE_TO_OPEN[ch] != open_ch:
                    raise ConstraintError(
                        f"constraints position {i} (0-based): {ch!r} does "
                        f"not match {open_ch!r} opened at position {open_pos}"
                    )
                string_pairs.append((open_pos, i))
            else:
                raise ConstraintError(
                    f"constraints position {i} (0-based): illegal character "
                    f"{ch!r}; allowed: . x ( ) [ ] {{ }} < >"
                )
        if stack:
            open_ch, open_pos = stack[-1]
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
            # crossing: a < c < b < d or c < a < d < b
            if (a < c < b < d) or (c < a < d < b):
                raise ConstraintError(
                    f"pairs[{idx}]=({a}, {b}) and pairs[{jdx}]=({c}, {d}) "
                    f"cross each other; pseudoknots are forbidden"
                )

    force_up = [0] * n
    for p in unpaired:
        force_up[p] = 1
    return partner, force_up


def _normalise_unpaired_scores(value: object, n: int) -> List[int]:
    """Validate the per-position unpaired score table.

    Accepts None (all zero), a sequence of exactly n integers, or a mapping
    {position: score}; positions absent from the mapping default to 0.
    """
    if value is None:
        return [0] * n
    if isinstance(value, Mapping):
        scores = [0] * n
        for key, val in value.items():
            p = _as_int(key, "unpaired_scores: key")
            if not 0 <= p < n:
                raise ValueError(
                    f"unpaired_scores: position {p} out of range "
                    f"[0, {n - 1}]"
                )
            scores[p] = _as_int(val, f"unpaired_scores[{p}]")
        return scores
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "unpaired_scores: expected a sequence of n integers or a "
            f"mapping, got {type(value).__name__}"
        )
    try:
        items = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "unpaired_scores: expected a sequence of n integers or a "
            f"mapping, got {type(value).__name__}"
        ) from None
    if len(items) != n:
        raise ValueError(
            f"unpaired_scores: expected {n} entries (one per position), "
            f"got {len(items)}"
        )
    return [_as_int(v, f"unpaired_scores[{i}]") for i, v in enumerate(items)]


# ---------------------------------------------------------------------------
# Core dynamic program
# ---------------------------------------------------------------------------


def fold(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    *,
    unpaired_scores: "Optional[Iterable[int] | Mapping[int, int]]" = None,
    stack_score: int = 0,
) -> Dict[str, object]:
    """Fold ``sequence`` into a maximum-score non-crossing secondary structure.

    :param sequence: RNA sequence over ``A C G U`` (case sensitive),
        length <= 120.
    :param constraints: ``None``, or either

        * a string of the sequence length: ``.`` free, ``x`` forced
          unpaired, and matched brackets ``() [] {} <>`` forcing the paired
          positions; or
        * a dict with ``"pairs"``: iterable of ``(i, j)`` integer pairs and
          ``"unpaired"``: iterable of integer positions (all 0-based).

        Forced pairs must be complementary, satisfy ``j - i >= 4``, share no
        endpoint, and not cross.
    :param unpaired_scores: optional per-position integer score awarded when
        that position stays unpaired.  Either a sequence of exactly
        ``len(sequence)`` integers or a mapping ``{position: score}``
        (missing positions default to 0).  Defaults to all zero.
    :param stack_score: integer bonus added once for every directly nested
        adjacent pair: both ``(i, j)`` and ``(i+1, j-1)`` present in the
        structure.  May be negative (a penalty).  Defaults to 0.
    :returns: dict with

        * ``score`` (int): maximum total score (pair scores + unpaired
          scores + stacking bonuses);
        * ``structure`` (str): lexicographically smallest dot-bracket string
          among all maximum-score structures (ASCII order, so ``'(' < ')' <
          '.'``);
        * ``count`` (int): number of maximum-score structures, modulo
          1_000_000_007;
        * ``pairs`` (tuple of ``(i, j)``): every pair that occurs in *at
          least one* maximum-score structure, sorted.
    :raises TypeError: wrong value types (including ``bool`` as int).
    :raises InvalidSequenceError: bad length or illegal character.
    :raises ValueError: malformed ``unpaired_scores`` (bad length, position
        out of range, non-integer entry).
    :raises ConstraintError: malformed or conflicting constraints.
    :raises UnsatisfiableConstraintError: proven that no admissible structure
        satisfies the forced constraints.
    """
    seq = _validate_sequence(sequence)
    partner, force_up = _normalise_constraints(constraints, seq)
    n = len(seq)
    up = _normalise_unpaired_scores(unpaired_scores, n)
    stack = _as_int(stack_score, "stack_score")

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

    # Half-open intervals [i, j), 0 <= i <= j <= n.  Two score tables are
    # needed because the stacking bonus depends on whether the interval is
    # directly enclosed by a pair:
    #
    #   B[i][j]: best score of [i, j) with no enclosing pair (top level or
    #            the region after a pair's right endpoint).
    #   E[i][j]: best score of [i, j) given it is directly enclosed by the
    #            pair (i-1, j); choosing the pair (i, j-1) here forms the
    #            adjacent nested stack (i-1, j)+(i, j-1) and earns the bonus.
    #            Once position i stays unpaired, or i pairs with some
    #            t < j-1, nothing inside the remainder can be adjacent to
    #            the enclosing pair any more, so those remainders fall back
    #            to the B table.
    #
    # The decomposition is canonical (always peel the leftmost position of
    # the interval), so every structure corresponds to exactly one
    # decomposition path and is counted exactly once.
    best: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    enc: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    ways: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    enc_ways: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        ways[i][i] = 1
        enc_ways[i][i] = 1

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length

            # --- E[i][j]: interval enclosed by the pair (i-1, j) ---------
            top = _NEG
            total = 0

            # Option 1: position i stays unpaired (impossible if its mate is
            # forced, or if its mate would be stranded outside the
            # remainder).  The enclosing pair can no longer stack with
            # anything inside, so the remainder is a plain B interval.
            if partner[i] == -1 and closed[i + 1][j]:
                v = up[i] + best[i + 1][j]
                w = ways[i + 1][j]
                if v > top:
                    top, total = v, w
                elif v == top:
                    total = (total + w) % MOD

            # Option 2: i pairs with t.
            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t):
                    continue
                if not closed[i + 1][t] or not closed[t + 1][j]:
                    continue
                s_in = enc[i + 1][t]
                s_out = best[t + 1][j]
                if s_in == _NEG or s_out == _NEG:
                    continue
                # The pair (i, j-1) sits directly inside the enclosing pair
                # (i-1, j): exactly the adjacent nested configuration.
                bonus = stack if t == j - 1 else 0
                v = PAIR_SCORES[(seq[i], seq[t])] + bonus + s_in + s_out
                w = (enc_ways[i + 1][t] * ways[t + 1][j]) % MOD
                if v > top:
                    top, total = v, w
                elif v == top:
                    total = (total + w) % MOD

            enc[i][j] = top
            enc_ways[i][j] = total

            # --- B[i][j]: no enclosing pair ------------------------------
            top = _NEG
            total = 0

            if partner[i] == -1 and closed[i + 1][j]:
                v = up[i] + best[i + 1][j]
                w = ways[i + 1][j]
                if v > top:
                    top, total = v, w
                elif v == top:
                    total = (total + w) % MOD

            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t):
                    continue
                if not closed[i + 1][t] or not closed[t + 1][j]:
                    continue
                s_in = enc[i + 1][t]
                s_out = best[t + 1][j]
                if s_in == _NEG or s_out == _NEG:
                    continue
                v = PAIR_SCORES[(seq[i], seq[t])] + s_in + s_out
                w = (enc_ways[i + 1][t] * ways[t + 1][j]) % MOD
                if v > top:
                    top, total = v, w
                elif v == top:
                    total = (total + w) % MOD

            best[i][j] = top
            ways[i][j] = total

    if best[0][n] == _NEG:
        raise UnsatisfiableConstraintError(
            "constraints are unsatisfiable: no admissible structure contains "
            "all forced pairs"
        )

    # Lexicographically smallest optimal structure per interval.
    lex: List[List[str]] = [[""] * (n + 1) for _ in range(n + 1)]
    enc_lex: List[List[str]] = [[""] * (n + 1) for _ in range(n + 1)]
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            # (table, lex_table, rest_table, rest_lex, is_enc): an enclosed
            # interval falls back to the plain B tables for its unpaired and
            # after-pair remainders (see the DP comment above).
            for table, lex_table, rest, rest_lex, is_enc in (
                (enc, enc_lex, best, lex, True),
                (best, lex, best, lex, False),
            ):
                if table[i][j] == _NEG:
                    continue  # infeasible interval: no witness string
                target = table[i][j]
                chosen: Optional[str] = None

                if (
                    partner[i] == -1
                    and closed[i + 1][j]
                    and up[i] + rest[i + 1][j] == target
                ):
                    chosen = "." + rest_lex[i + 1][j]

                for t in range(i + MIN_HAIRPIN + 1, j):
                    if not pair_allowed(i, t):
                        continue
                    if not closed[i + 1][t] or not closed[t + 1][j]:
                        continue
                    s_in = enc[i + 1][t]
                    s_out = rest[t + 1][j]
                    if s_in == _NEG or s_out == _NEG:
                        continue
                    bonus = stack if (is_enc and t == j - 1) else 0
                    v = PAIR_SCORES[(seq[i], seq[t])] + bonus + s_in + s_out
                    if v != target:
                        continue
                    candidate = (
                        "(" + enc_lex[i + 1][t] + ")" + rest_lex[t + 1][j]
                    )
                    if chosen is None or candidate < chosen:
                        chosen = candidate

                # Feasible intervals always have a witness (root is feasible
                # and table values are maxima over enumerated options).
                assert chosen is not None and len(chosen) == length
                lex_table[i][j] = chosen

    # Pairs appearing in at least one global optimum: propagate reachability
    # of intervals along optimal decisions from the root [0, n).
    reachable = [[False] * (n + 1) for _ in range(n + 1)]
    enc_reachable = [[False] * (n + 1) for _ in range(n + 1)]
    reachable[0][n] = True
    can_pair: set[Tuple[int, int]] = set()

    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length
            for table, reach, rest, rest_reach, is_enc in (
                (enc, enc_reachable, best, reachable, True),
                (best, reachable, best, reachable, False),
            ):
                if not reach[i][j]:
                    continue
                target = table[i][j]

                # Feasibility is tracked by table != _NEG only.  The ways
                # counters are reduced modulo MOD and may legitimately be 0
                # for a non-empty set of structures, so they must never gate
                # reachability: otherwise a pair appearing only behind a
                # count that is 0 mod MOD would be wrongly omitted.
                if (
                    partner[i] == -1
                    and closed[i + 1][j]
                    and up[i] + rest[i + 1][j] == target
                ):
                    rest_reach[i + 1][j] = True

                for t in range(i + MIN_HAIRPIN + 1, j):
                    if not pair_allowed(i, t):
                        continue
                    if not closed[i + 1][t] or not closed[t + 1][j]:
                        continue
                    s_in = enc[i + 1][t]
                    s_out = rest[t + 1][j]
                    if s_in == _NEG or s_out == _NEG:
                        continue
                    bonus = stack if (is_enc and t == j - 1) else 0
                    v = PAIR_SCORES[(seq[i], seq[t])] + bonus + s_in + s_out
                    if v != target:
                        continue
                    can_pair.add((i, t))
                    enc_reachable[i + 1][t] = True
                    rest_reach[t + 1][j] = True

    return {
        "score": best[0][n],
        "structure": lex[0][n],
        "count": ways[0][n],
        "pairs": tuple(sorted(can_pair)),
    }
