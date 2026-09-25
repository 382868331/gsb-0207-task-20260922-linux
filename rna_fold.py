"""RNA secondary structure prediction with forced pairings.

Single public entry point: :func:`fold`.

The implementation is an exact, non-crossing (pseudoknot-free) interval
dynamic program.  It performs an exhaustive search over every admissible
structure -- it never samples -- so a returned optimum is proven optimal and a
rejected constraint set is proven unsatisfiable.  See README.md for the full
contract.
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
    "score_structure",
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


# ---------------------------------------------------------------------------
# Core dynamic program
# ---------------------------------------------------------------------------


def _normalise_scoring(
    unpaired_scores: object, stack_score: object, n: int
) -> Tuple[List[int], int]:
    """Validate the optional scoring parameters.

    ``unpaired_scores`` must be a length-``n`` iterable of integers (strings
    and bytes rejected); ``stack_score`` must be a single integer.  Booleans
    and non-finite floats follow the module-wide numeric contract.
    """
    if unpaired_scores is None:
        scores = [0] * n
    else:
        if isinstance(unpaired_scores, (str, bytes)):
            raise TypeError(
                "unpaired_scores: expected an iterable of integers, not a "
                "string"
            )
        try:
            raw = list(unpaired_scores)
        except TypeError as exc:
            raise TypeError("unpaired_scores: value is not iterable") from exc
        if len(raw) != n:
            raise ConstraintError(
                f"unpaired_scores length {len(raw)} != sequence length {n}"
            )
        scores = [
            _as_int(v, f"unpaired_scores[{k}]") for k, v in enumerate(raw)
        ]

    if stack_score is None:
        stack = 0
    else:
        stack = _as_int(stack_score, "stack_score")
    return scores, stack


def score_structure(
    sequence: str,
    structure: str,
    unpaired_scores: "Optional[Iterable[int]]" = None,
    stack_score: int = 0,
) -> int:
    """Score a concrete dot-bracket ``structure`` under the full scoring rule.

    Pair scores are GC/CG=3, AU/UA=2, GU/UG=1; every unpaired position adds
    ``unpaired_scores[p]`` (default 0); every directly nested adjacent pair
    stack ``(i, j)`` + ``(i+1, j-1)`` adds ``stack_score`` once (default 0).

    The structure must be a balanced dot-bracket string of the sequence
    length whose pairs are complementary, satisfy the minimum hairpin length,
    and never cross (a dot-bracket string cannot express crossings).
    :class:`InvalidSequenceError` / :class:`ConstraintError` are raised on
    malformed input, with position information.
    """
    seq = _validate_sequence(sequence)
    n = len(seq)
    if not isinstance(structure, str):
        raise TypeError(
            f"structure: expected str, got {type(structure).__name__}"
        )
    if len(structure) != n:
        raise ConstraintError(
            f"structure length {len(structure)} != sequence length {n}"
        )
    up_scores, stack = _normalise_scoring(unpaired_scores, stack_score, n)

    total = 0
    open_stack: List[int] = []
    partner_of: Dict[int, int] = {}
    for p, ch in enumerate(structure):
        if ch == ".":
            total += up_scores[p]
        elif ch == "(":
            open_stack.append(p)
        elif ch == ")":
            if not open_stack:
                raise ConstraintError(
                    f"structure position {p} (0-based): closing bracket has "
                    "no matching opening bracket"
                )
            a = open_stack.pop()
            b = p
            if b - a <= MIN_HAIRPIN:
                raise ConstraintError(
                    f"structure pair ({a}, {b}): pairs enclose at least "
                    f"{MIN_HAIRPIN} unpaired positions, so j - i must be >= "
                    f"{MIN_HAIRPIN + 1}"
                )
            w = PAIR_SCORES.get((seq[a], seq[b]))
            if w is None:
                raise ConstraintError(
                    f"structure pair ({a}, {b}): {seq[a]!r}-{seq[b]!r} is "
                    "not an allowed pair (allowed: AU, UA, GC, CG, GU, UG)"
                )
            total += w
            partner_of[a] = b
            partner_of[b] = a
        else:
            raise ConstraintError(
                f"structure position {p} (0-based): illegal character "
                f"{ch!r}; allowed: ( ) ."
            )
    if open_stack:
        p = open_stack[-1]
        raise ConstraintError(
            f"structure position {p} (0-based): opening bracket is never "
            "closed"
        )
    for a, b in partner_of.items():
        if a < b and b >= 2 and partner_of.get(b - 1, -1) == a + 1:
            total += stack
    return total


def fold(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    *,
    unpaired_scores: "Optional[Iterable[object]]" = None,
    stack_score: "Optional[int]" = None,
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
    :param unpaired_scores: optional length-``n`` iterable of integers;
        leaving position ``p`` unpaired adds ``unpaired_scores[p]`` to the
        total.  Defaults to all zeros; negative values are allowed (a cost
        for leaving the position unpaired).
    :param stack_score: optional integer added **once** for every directly
        nested adjacent pair of pairs, i.e. whenever both ``(i, j)`` and
        ``(i + 1, j - 1)`` occur.  Defaults to 0; may be negative.
    :returns: dict with

        * ``score`` (int): maximum total score = sum of pair scores
          (GC=3, AU=2, GU=1) + unpaired-position scores + adjacent nesting
          bonuses;
        * ``structure`` (str): lexicographically smallest dot-bracket string
          among all maximum-score structures (ASCII order, so ``'(' < ')' <
          '.'``);
        * ``count`` (int): number of maximum-score structures, modulo
          1_000_000_007;
        * ``pairs`` (tuple of ``(i, j)``): every pair that occurs in *at
          least one* maximum-score structure, sorted.
    :raises TypeError: wrong value types (including ``bool`` as int).
    :raises InvalidSequenceError: bad length or illegal character.
    :raises ConstraintError: malformed or conflicting constraints.
    :raises UnsatisfiableConstraintError: proven that no admissible structure
        satisfies the forced constraints.
    """
    seq = _validate_sequence(sequence)
    partner, force_up = _normalise_constraints(constraints, seq)
    n = len(seq)
    up_scores, stack = _normalise_scoring(unpaired_scores, stack_score, n)

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

    # Three DP tables over half-open intervals [i, j):
    #
    # f  [i][j]: best score / count of ALL admissible structures;
    # c  [i][j]: structures whose boundary forms the pair (i, j-1);
    # nb [i][j]: structures whose boundary does NOT form that pair.
    # f is the disjoint union of c and nb.  The boundary state is required
    # because the adjacent nesting bonus depends on whether the inner
    # interval ACTUALLY pairs its own boundary; it cannot be recovered from a
    # dot-bracket after the fact, and keeping c and nb disjoint guarantees
    # every structure is counted exactly once even when several interval
    # decompositions lead to the same matching.
    #
    # Transitions:
    #   f[i][j] = i unpaired: up[i] + f[i+1][j]
    #           | i pairs t  : w(i,t) + f[t+1][j]
    #                        + max-merge(c[i+1][t] + stack, nb[i+1][t])
    #   c[i][j] keeps only the t = j-1 pairing terms;
    #   nb[i][j] keeps the unpaired term and pairing terms t < j-1.
    f_best: List[List[int]] = [[_NEG] * (n + 1) for _ in range(n + 1)]
    f_ways: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    c_best: List[List[int]] = [[_NEG] * (n + 1) for _ in range(n + 1)]
    c_ways: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    nb_best: List[List[int]] = [[_NEG] * (n + 1) for _ in range(n + 1)]
    nb_ways: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        f_best[i][i] = 0
        f_ways[i][i] = 1
        nb_best[i][i] = 0
        nb_ways[i][i] = 1

    def merge(top: int, total: int, v: int, w: int):
        if v > top:
            return v, w
        if v == top:
            return top, (total + w) % MOD
        return top, total

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            top = nb_top = c_top = _NEG
            total = nb_total = c_total = 0

            # Term A: position i stays unpaired -> f and nb, never c.
            if partner[i] == -1 and closed[i + 1][j]:
                v = up_scores[i] + f_best[i + 1][j]
                w = f_ways[i + 1][j]
                top, total = merge(top, total, v, w)
                nb_top, nb_total = merge(nb_top, nb_total, v, w)

            # Term B: i pairs with t; the interior split on the boundary
            # state decides whether the stack bonus is earned exactly once.
            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t) or not closed[t + 1][j]:
                    continue
                s_out = f_best[t + 1][j]
                if s_out == _NEG:
                    continue
                w_out = f_ways[t + 1][j]
                pair_w = PAIR_SCORES[(seq[i], seq[t])]

                in_top = _NEG
                in_total = 0
                if c_best[i + 1][t] != _NEG:
                    in_top, in_total = merge(
                        in_top, in_total,
                        stack + c_best[i + 1][t], c_ways[i + 1][t],
                    )
                if nb_best[i + 1][t] != _NEG:
                    in_top, in_total = merge(
                        in_top, in_total,
                        nb_best[i + 1][t], nb_ways[i + 1][t],
                    )
                if in_top == _NEG:
                    continue
                v = pair_w + in_top + s_out
                w = (in_total * w_out) % MOD
                top, total = merge(top, total, v, w)
                if t == j - 1:
                    c_top, c_total = merge(c_top, c_total, v, w)
                else:
                    nb_top, nb_total = merge(nb_top, nb_total, v, w)

            f_best[i][j], f_ways[i][j] = top, total
            c_best[i][j], c_ways[i][j] = c_top, c_total
            nb_best[i][j], nb_ways[i][j] = nb_top, nb_total

    if f_best[0][n] == _NEG:
        raise UnsatisfiableConstraintError(
            "constraints are unsatisfiable: no admissible structure contains "
            "all forced pairs"
        )

    # Replay every table with the same decision terms, collecting the lex
    # smallest witness fragment and propagating interval reachability.
    # Feasibility is tracked by best != _NEG only: ways is reduced modulo
    # MOD and may legitimately be 0 for a non-empty structure set, so it must
    # never gate reachability (otherwise a pair appearing only behind a count
    # that is 0 mod MOD would be wrongly omitted).
    f_lex: List[List[Optional[str]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    c_lex: List[List[Optional[str]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    nb_lex: List[List[Optional[str]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        f_lex[i][i] = nb_lex[i][i] = ""

    f_reach = [[False] * (n + 1) for _ in range(n + 1)]
    c_reach = [[False] * (n + 1) for _ in range(n + 1)]
    nb_reach = [[False] * (n + 1) for _ in range(n + 1)]
    f_reach[0][n] = True
    can_pair: set[Tuple[int, int]] = set()

    def interior_options(i: int, t: int):
        """Optimal interior decompositions of pair (i, t): (value, kind)."""
        if c_best[i + 1][t] != _NEG:
            yield stack + c_best[i + 1][t], "c"
        if nb_best[i + 1][t] != _NEG:
            yield nb_best[i + 1][t], "nb"

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            f_pieces: List[str] = []
            c_pieces: List[str] = []
            nb_pieces: List[str] = []

            if partner[i] == -1 and closed[i + 1][j]:
                v = up_scores[i] + f_best[i + 1][j]
                if v == f_best[i][j]:
                    f_pieces.append("." + f_lex[i + 1][j])
                if v == nb_best[i][j]:
                    nb_pieces.append("." + f_lex[i + 1][j])

            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t) or not closed[t + 1][j]:
                    continue
                s_out = f_best[t + 1][j]
                if s_out == _NEG:
                    continue
                pair_w = PAIR_SCORES[(seq[i], seq[t])]
                in_best = max(
                    stack + c_best[i + 1][t],
                    nb_best[i + 1][t],
                )
                if in_best == _NEG:
                    continue
                v = pair_w + in_best + s_out
                if v not in (f_best[i][j], c_best[i][j], nb_best[i][j]):
                    continue
                for in_val, kind in interior_options(i, t):
                    if in_val != in_best:
                        continue
                    inner_lex = c_lex if kind == "c" else nb_lex
                    frag = (
                        "(" + inner_lex[i + 1][t] + ")"
                        + f_lex[t + 1][j]
                    )
                    if v == f_best[i][j]:
                        f_pieces.append(frag)
                    if t == j - 1 and v == c_best[i][j]:
                        c_pieces.append(frag)
                    if t < j - 1 and v == nb_best[i][j]:
                        nb_pieces.append(frag)

            if f_best[i][j] != _NEG:
                assert f_pieces and all(len(p) == length for p in f_pieces)
                f_lex[i][j] = min(f_pieces)
            if c_best[i][j] != _NEG:
                assert c_pieces and all(len(p) == length for p in c_pieces)
                c_lex[i][j] = min(c_pieces)
            if nb_best[i][j] != _NEG:
                assert nb_pieces and all(len(p) == length for p in nb_pieces)
                nb_lex[i][j] = min(nb_pieces)

    def propagate_table(
        i: int,
        j: int,
        target: int,
        t_filter: str,
    ) -> None:
        """Mark subintervals/pairs used by one optimal decision of [i, j)."""
        if (
            t_filter != "last"
            and partner[i] == -1
            and closed[i + 1][j]
            and up_scores[i] + f_best[i + 1][j] == target
        ):
            f_reach[i + 1][j] = True

        for t in range(i + MIN_HAIRPIN + 1, j):
            if not pair_allowed(i, t) or not closed[t + 1][j]:
                continue
            if t_filter == "last" and t != j - 1:
                continue
            if t_filter == "notlast" and t == j - 1:
                continue
            s_out = f_best[t + 1][j]
            if s_out == _NEG:
                continue
            pair_w = PAIR_SCORES[(seq[i], seq[t])]
            in_best = max(
                stack + c_best[i + 1][t],
                nb_best[i + 1][t],
            )
            v = pair_w + in_best + s_out
            if v != target:
                continue
            can_pair.add((i, t))
            f_reach[t + 1][j] = True
            for in_val, kind in interior_options(i, t):
                if in_val != in_best:
                    continue
                if kind == "c":
                    c_reach[i + 1][t] = True
                else:
                    nb_reach[i + 1][t] = True

    # One descending pass over all three boundary states.  Every marked
    # subinterval is strictly shorter, so it is visited later in this same
    # loop regardless of which table marked it.
    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length
            if f_reach[i][j]:
                propagate_table(i, j, f_best[i][j], "all")
            if nb_reach[i][j]:
                propagate_table(i, j, nb_best[i][j], "notlast")
            if c_reach[i][j]:
                propagate_table(i, j, c_best[i][j], "last")

    return {
        "score": f_best[0][n],
        "structure": f_lex[0][n],
        "count": f_ways[0][n],
        "pairs": tuple(sorted(can_pair)),
    }
