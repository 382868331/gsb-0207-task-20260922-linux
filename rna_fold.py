"""RNA secondary structure prediction with forced pairings.

Public entry points:

* :func:`fold` -- best score, lex-smallest optimum, modular count, pair set;
* :func:`claim_optimal_structures` -- stable, batched, lazy claiming of the
  tied-optimal structures via an input-bound opaque continuation cursor.

The implementation is an exact, non-crossing (pseudoknot-free) interval
dynamic program.  It performs an exhaustive search over every admissible
structure -- it never samples -- so a returned optimum is proven optimal and a
rejected constraint set is proven unsatisfiable.  See README.md for the full
contract.
"""

from __future__ import annotations

import base64
import hashlib
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "fold",
    "claim_optimal_structures",
    "InvalidSequenceError",
    "ConstraintError",
    "CursorError",
    "UnsatisfiableConstraintError",
    "BudgetExceededError",
    "MIN_HAIRPIN",
    "MAX_SEQUENCE_LENGTH",
    "MAX_BATCH_LIMIT",
    "PAIR_SCORES",
    "MOD",
]

MIN_HAIRPIN = 3
"""Minimum number of *unpaired* positions inside a pair (j - i >= 4)."""

MAX_SEQUENCE_LENGTH = 120

MAX_BATCH_LIMIT = 100
"""Maximum number of tied-optimal structures returned in one claim batch."""

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


class CursorError(ValueError):
    """A continuation cursor is malformed, tampered with, or does not match
    the current (sequence, constraints) input.

    Cursors bind to one concrete input: a cursor produced for one sequence
    (or for different constraints) can never be reused for another input.
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


def _solve(
    seq: str, partner: List[int], force_up: List[int]
) -> "_OptimalTables":
    """Run the exact interval DP and build all reconstruction tables.

    The DP itself never materialises any concrete structure beyond the
    per-interval lexicographically smallest witness (``lex``).  ``alts``
    stores, per feasible interval, the *optimal* decisions as lightweight
    templates; :class:`_OptimalEnumerator` walks them on demand so tied
    optima are generated lazily.
    """
    n = len(seq)

    # A *closed* interval [i, j) is one where every position p in [i, j)
    # with a forced mate also has that mate inside [i, j).  The DP only
    # considers closed intervals, so any candidate pair (i, t) that crossed a
    # forced pair (a, b) -- a < i < b < t or i < a < t < b -- would split an
    # endpoint from its mate and is automatically unreachable.
    # closed[i][j] is maintained as: (forced endpoints in interval) ==
    # 2 * (forced pairs fully inside the interval).
    closed: List[List[bool]] = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        endpoint_count = 0
        inside_pairs = 0
        closed[i][i] = True
        for j in range(i + 1, n + 1):
            p = j - 1
            if partner[p] != -1:
                endpoint_count += 1
                m = partner[p]
                if i <= m < p:  # right endpoint of a pair fully inside
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

    # Half-open intervals [i, j), 0 <= i <= j <= n.
    best: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    ways: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        ways[i][i] = 1

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            top = _NEG
            total = 0

            # Option 1: position i stays unpaired (impossible if its mate is
            # forced, or if its mate would be stranded outside the remainder).
            if partner[i] == -1 and closed[i + 1][j]:
                v = best[i + 1][j]
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
                s_in = best[i + 1][t]
                s_out = best[t + 1][j]
                if s_in == _NEG or s_out == _NEG:
                    continue
                v = PAIR_SCORES[(seq[i], seq[t])] + s_in + s_out
                w = (ways[i + 1][t] * ways[t + 1][j]) % MOD
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

    # Lexicographically smallest optimal structure per interval, together
    # with the list of optimal decision templates for each interval.
    #
    # A template node is either ("lit", ch) -- a fixed dot-bracket character
    # -- or ("sub", a, b) -- any optimal structure of sub-interval [a, b).
    # Templates are paired-first to match ASCII order ('(' = 40 < '.' = 46),
    # but callers always take the minimum explicitly, so order is not relied
    # upon for correctness.
    lex: List[List[str]] = [[""] * (n + 1) for _ in range(n + 1)]
    alts: List[List[List[Tuple[object, ...]]]] = [
        [[] for _ in range(n + 1)] for _ in range(n + 1)
    ]
    for i in range(n + 1):
        lex[i][i] = ""

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            if best[i][j] == _NEG:
                continue  # infeasible interval: no witness or template
            target = best[i][j]
            templates: List[List[Tuple[object, ...]]] = []
            chosen: Optional[str] = None

            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t):
                    continue
                if not closed[i + 1][t] or not closed[t + 1][j]:
                    continue
                s_in = best[i + 1][t]
                s_out = best[t + 1][j]
                if s_in == _NEG or s_out == _NEG:
                    continue
                v = PAIR_SCORES[(seq[i], seq[t])] + s_in + s_out
                if v != target:
                    continue
                template = [
                    ("lit", "("),
                    ("sub", i + 1, t),
                    ("lit", ")"),
                    ("sub", t + 1, j),
                ]
                templates.append(template)
                candidate = (
                    "(" + lex[i + 1][t] + ")" + lex[t + 1][j]
                )
                if chosen is None or candidate < chosen:
                    chosen = candidate

            if (
                partner[i] == -1
                and closed[i + 1][j]
                and best[i + 1][j] == target
            ):
                templates.append(
                    [("lit", "."), ("sub", i + 1, j)]
                )
                candidate = "." + lex[i + 1][j]
                if chosen is None or candidate < chosen:
                    chosen = candidate

            # Feasible intervals always have a witness (root is feasible and
            # table values are maxima over enumerated options).
            assert chosen is not None and len(chosen) == length
            lex[i][j] = chosen
            alts[i][j] = templates

    return _OptimalTables(
        seq=seq,
        partner=partner,
        force_up=force_up,
        closed=closed,
        best=best,
        ways=ways,
        lex=lex,
        alts=alts,
    )


class _OptimalTables:
    """Bundle of DP tables shared by :func:`fold` and the lazy enumerator."""

    __slots__ = ("seq", "partner", "force_up", "closed", "best", "ways",
                 "lex", "alts")

    def __init__(
        self,
        seq: str,
        partner: List[int],
        force_up: List[int],
        closed: List[List[bool]],
        best: List[List[int]],
        ways: List[List[int]],
        lex: List[List[str]],
        alts: List[List[List[Tuple[object, ...]]]],
    ) -> None:
        self.seq = seq
        self.partner = partner
        self.force_up = force_up
        self.closed = closed
        self.best = best
        self.ways = ways
        self.lex = lex
        self.alts = alts


class _OptimalEnumerator:
    """Lazy enumeration of optimal dot-bracket strings in strict lex order.

    The only state carried between claims is the previously returned string
    (kept in the caller's opaque cursor).  Each successor is found by a
    bounded search for "smallest optimal completion strictly greater than the
    bound" over decision templates; unclaimed structures are never expanded.
    """

    def __init__(self, tables: _OptimalTables) -> None:
        self.t = tables
        self._ng_cache: Dict[Tuple[int, int, str], Optional[str]] = {}
        self._member_cache: Dict[Tuple[int, int, str], bool] = {}

    # -- membership ------------------------------------------------------

    def is_member(self, i: int, j: int, s: str) -> bool:
        """True iff ``s`` is an optimal structure of closed interval [i, j)."""
        if len(s) != j - i:
            return False
        key = (i, j, s)
        cached = self._member_cache.get(key)
        if cached is not None:
            return cached
        ok = self._check_member(i, j, s)
        self._member_cache[key] = ok
        return ok

    def _check_member(self, i: int, j: int, s: str) -> bool:
        if i == j:
            return s == ""
        if self.t.best[i][j] == _NEG:
            return False
        for template in self.t.alts[i][j]:
            if self._template_matches(template, s):
                return True
        return False

    def _template_matches(
        self, template: List[Tuple[object, ...]], s: str
    ) -> bool:
        pos = 0
        for node in template:
            if node[0] == "lit":
                if pos >= len(s) or s[pos] != node[1]:
                    return False
                pos += 1
            else:
                a, b = node[1], node[2]
                length = b - a
                if not self.is_member(a, b, s[pos:pos + length]):
                    return False
                pos += length
        return pos == len(s)

    # -- strict successor ------------------------------------------------

    def next_greater(self, i: int, j: int, bound: str) -> Optional[str]:
        """Smallest optimal structure of [i, j) strictly greater than bound."""
        if i == j or self.t.best[i][j] == _NEG:
            return None
        if len(bound) != j - i:
            return None
        key = (i, j, bound)
        if key in self._ng_cache:
            return self._ng_cache[key]
        result: Optional[str] = None
        for template in self.t.alts[i][j]:
            candidate = self._render_greater(template, bound)
            if candidate is not None and (
                result is None or candidate < result
            ):
                result = candidate
        self._ng_cache[key] = result
        return result

    def _render_greater(
        self, template: List[Tuple[object, ...]], bound: str
    ) -> Optional[str]:
        """Smallest instantiation of ``template`` strictly > ``bound``.

        A strict increase happens at exactly one node: a fixed literal is
        greater than the bound character, or a sub-interval contributes its
        own strict successor.  Every earlier node must equal the bound slice;
        every later node contributes its own lexicographic minimum.  Each
        sub-interval is an independent dot-bracket segment, so its minimum
        needs no compatibility adjustment against the fixed brackets the
        template places around it.
        """
        spans: List[Tuple[str, int, int, object]] = []
        pos = 0
        for node in template:
            if node[0] == "lit":
                ch = node[1]
                spans.append(("lit", pos, pos + 1, ch))
                pos += 1
            else:
                a, b = node[1], node[2]
                spans.append(("sub", pos, pos + (b - a), (a, b)))
                pos += b - a

        total = pos
        if len(bound) != total:
            return None
        best: Optional[str] = None

        for div in range(len(spans)):
            equal_parts: List[str] = []
            feasible = True
            for kind, p0, p1, payload in spans[:div]:
                if kind == "lit":
                    if payload != bound[p0]:
                        feasible = False
                        break
                    equal_parts.append(payload)
                else:
                    a, b = payload
                    sub_bound = bound[p0:p1]
                    if not self.is_member(a, b, sub_bound):
                        feasible = False
                        break
                    equal_parts.append(sub_bound)
            if not feasible:
                continue

            kind, p0, p1, payload = spans[div]
            if kind == "lit":
                if payload <= bound[p0]:
                    continue
                diverging = payload
            else:
                a, b = payload
                jump = self.next_greater(a, b, bound[p0:p1])
                if jump is None:
                    continue
                diverging = jump

            suffix_parts: List[str] = []
            for kind2, _q0, _q1, payload2 in spans[div + 1:]:
                if kind2 == "lit":
                    suffix_parts.append(payload2)
                else:
                    a, b = payload2
                    suffix_parts.append(self.t.lex[a][b])

            candidate = "".join(equal_parts) + diverging + "".join(
                suffix_parts
            )
            assert len(candidate) == total
            if best is None or candidate < best:
                best = candidate
        return best


def fold(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
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
    :returns: dict with

        * ``score`` (int): maximum total score (GC=3, AU=2, GU=1);
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
    tables = _solve(seq, partner, force_up)
    best = tables.best

    # Pairs appearing in at least one global optimum: propagate reachability
    # of intervals along optimal decisions from the root [0, n).
    n = len(seq)
    reachable = [[False] * (n + 1) for _ in range(n + 1)]
    reachable[0][n] = True
    can_pair: set[Tuple[int, int]] = set()

    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length
            if not reachable[i][j]:
                continue
            target = best[i][j]

            # Feasibility is tracked by best != _NEG only.  ways is reduced
            # modulo MOD and may legitimately be 0 for a non-empty set of
            # structures, so it must never gate reachability: otherwise a
            # pair appearing only behind a count that is 0 mod MOD would be
            # wrongly omitted.
            if (
                partner[i] == -1
                and tables.closed[i + 1][j]
                and best[i + 1][j] == target
            ):
                reachable[i + 1][j] = True

            for template in tables.alts[i][j]:
                first = template[0]
                if first[0] != "lit" or first[1] != "(":
                    continue
                t = template[1][2]  # inner sub-interval is [i + 1, t)
                can_pair.add((i, t))
                for node in template:
                    if node[0] == "sub":
                        reachable[node[1]][node[2]] = True

    return {
        "score": best[0][n],
        "structure": tables.lex[0][n],
        "count": tables.ways[0][n],
        "pairs": tuple(sorted(can_pair)),
    }


# ---------------------------------------------------------------------------
# Batched claiming of tied-optimal structures
# ---------------------------------------------------------------------------

_CURSOR_VERSION = "v1"
_CURSOR_SECRET_MARKER = "rna-fold-opt-claim"


def _fingerprint(
    seq: str, partner: List[int], force_up: List[int]
) -> str:
    """Stable token binding one cursor to one concrete input.

    The fingerprint covers the validated sequence and the *normalised*
    constraints, so equivalent constraint spellings share a cursor while a
    different sequence or different forced pairs/unpaired positions never do.
    """
    forced_pairs = sorted(
        (i, partner[i])
        for i in range(len(partner))
        if partner[i] > i
    )
    forced_up = sorted(p for p, flag in enumerate(force_up) if flag)
    payload = (
        f"n={len(seq)}|seq={seq}|pairs={forced_pairs}|unpaired={forced_up}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_cursor(
    fingerprint: str, after: str, score: int
) -> str:
    body = f"{_CURSOR_VERSION}|{fingerprint}|{score}|{after}"
    sig = hashlib.sha256(
        (body + "|" + _CURSOR_SECRET_MARKER).encode("utf-8")
    ).hexdigest()[:32]
    raw = f"{body}|{sig}"
    return "rf1_" + base64.urlsafe_b64encode(
        raw.encode("utf-8")
    ).decode("ascii").rstrip("=")


def _decode_cursor(cursor: object) -> Tuple[str, str, int, str]:
    """Decode and integrity-check a cursor; raise :class:`CursorError`."""
    if not isinstance(cursor, str) or not cursor.startswith("rf1_"):
        raise CursorError("cursor: malformed continuation token")
    try:
        padded = cursor[4:] + "=" * (-len(cursor[4:]) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise CursorError("cursor: malformed continuation token") from exc
    parts = raw.split("|")
    if len(parts) != 5 or parts[0] != _CURSOR_VERSION:
        raise CursorError("cursor: unsupported or malformed continuation token")
    _, fingerprint, score_text, after, sig = parts
    body = raw[: raw.rfind("|")]
    expected = hashlib.sha256(
        (body + "|" + _CURSOR_SECRET_MARKER).encode("utf-8")
    ).hexdigest()[:32]
    if not _constant_time_eq(sig, expected):
        raise CursorError("cursor: integrity check failed (tampered token)")
    try:
        score = int(score_text)
    except ValueError as exc:
        raise CursorError("cursor: malformed score field") from exc
    return fingerprint, after, score, sig


def _constant_time_eq(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= ord(x) ^ ord(y)
    return diff == 0


def claim_optimal_structures(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    limit: int = 10,
    cursor: Optional[str] = None,
) -> Dict[str, object]:
    """Claim one batch of tied-optimal structures in stable order.

    Every returned structure is a maximum-score admissible structure: it
    respects every forced pair and forced-unpaired position, the minimum
    hairpin gap, complementarity and non-crossing rules.  Across batches the
    structures are emitted in strictly increasing dot-bracket lexicographic
    order (ASCII), so the same structure is never returned twice.

    Structures are generated lazily: only the ``limit`` (plus one probe)
    successors are searched per call; structures that have not been claimed
    yet are never expanded into memory.

    :param sequence: RNA sequence over ``A C G U`` (case sensitive),
        length <= 120.
    :param constraints: same forms as :func:`fold`.
    :param limit: batch size, an integer with ``1 <= limit <= 100``.
        ``bool`` is rejected like other integer parameters.
    :param cursor: ``None`` for the first claim, or the ``next_cursor``
        string returned by the previous claim.  A cursor is bound to one
        concrete (sequence, constraints) input and is rejected on mismatch.
    :returns: dict with

        * ``structures`` (tuple of str): this batch, sorted lexicographically;
        * ``score`` (int): the common optimal score;
        * ``count`` (int): total number of optimal structures, modulo
          1_000_000_007 (independent of how many were claimed);
        * ``next_cursor`` (str | None): opaque continuation token, or
          ``None`` when every optimal structure has been claimed;
        * ``done`` (bool): ``True`` exactly when ``next_cursor is None``.
    :raises TypeError: wrong value types (including ``bool`` as int).
    :raises InvalidSequenceError: bad length or illegal character.
    :raises ConstraintError: malformed or conflicting constraints.
    :raises CursorError: malformed/tampered cursor, or cursor bound to a
        different sequence or different constraints.
    :raises UnsatisfiableConstraintError: proven unsatisfiable constraints.
    """
    limit = _as_int(limit, "limit")
    if not 1 <= limit <= MAX_BATCH_LIMIT:
        raise ValueError(
            f"limit={limit}: must satisfy 1 <= limit <= {MAX_BATCH_LIMIT}"
        )
    if cursor is not None and not isinstance(cursor, str):
        raise CursorError(
            f"cursor: expected str or None, got {type(cursor).__name__}"
        )

    seq = _validate_sequence(sequence)
    partner, force_up = _normalise_constraints(constraints, seq)
    fingerprint = _fingerprint(seq, partner, force_up)

    after: Optional[str] = None
    if cursor is not None:
        cursor_fp, cursor_after, cursor_score, _sig = _decode_cursor(cursor)
        if cursor_fp != fingerprint:
            raise CursorError(
                "cursor does not match this input: it was created for a "
                "different sequence or different constraints"
            )
        if len(cursor_after) != len(seq):
            raise CursorError(
                "cursor: bound structure length does not match the input"
            )
        after = cursor_after

    tables = _solve(seq, partner, force_up)

    if after is not None:
        if cursor_score != tables.best[0][len(seq)]:
            # Covered in practice by the fingerprint, but never hand out a
            # cursor whose recorded score disagrees with the solved input.
            raise CursorError("cursor: recorded score disagrees with input")
        enumerator = _OptimalEnumerator(tables)
        if not enumerator.is_member(0, len(seq), after):
            raise CursorError(
                "cursor: bound structure is not an optimal structure of "
                "this input"
            )
        current = enumerator.next_greater(0, len(seq), after)
    else:
        enumerator = _OptimalEnumerator(tables)
        current = tables.lex[0][len(seq)]

    batch: List[str] = []
    n = len(seq)
    while current is not None and len(batch) < limit:
        batch.append(current)
        current = enumerator.next_greater(0, n, current)

    done = current is None
    if done:
        next_cursor = None
    else:
        # One probe (``current``) beyond the batch was searched; it is not
        # returned now, so unclaimed structures stay unexpanded.
        next_cursor = _encode_cursor(
            fingerprint, batch[-1], tables.best[0][n]
        )

    return {
        "structures": tuple(batch),
        "score": tables.best[0][n],
        "count": tables.ways[0][n],
        "next_cursor": next_cursor,
        "done": done,
    }
