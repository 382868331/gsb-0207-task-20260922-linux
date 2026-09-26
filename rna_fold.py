"""RNA secondary structure prediction with forced pairings.

Single public entry point: :func:`fold`.

The implementation is an exact, non-crossing (pseudoknot-free) interval
dynamic program.  It performs an exhaustive search over every admissible
structure -- it never samples -- so a returned optimum is proven optimal and a
rejected constraint set is proven unsatisfiable.  An optional mode (keyword
argument ``pseudoknot_penalty``) additionally allows exactly one H-type
pseudoknot (two consecutive nested stems interleaved as i < k < j < l) for
sequences up to 24 nt.  See README.md for the full contract.
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
    "MAX_PSEUDOKNOT_SEQUENCE_LENGTH",
    "PAIR_SCORES",
    "MOD",
]

MIN_HAIRPIN = 3
"""Minimum number of *unpaired* positions inside a pair (j - i >= 4)."""

MAX_SEQUENCE_LENGTH = 120

MAX_PSEUDOKNOT_SEQUENCE_LENGTH = 24
"""Maximum sequence length when the optional H-type pseudoknot mode is on."""

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


def _solve(
    seq: str,
    partner: List[int],
    force_up: List[int],
    orig_pos: Optional[List[int]] = None,
) -> Dict[str, object]:
    """Exact non-crossing DP for one (sequence, constraints) system.

    Returns a dict with the interval tables ``best`` / ``ways`` / ``lex`` /
    ``closed``, the ``pair_allowed`` predicate, and ``can_pair`` -- the set
    of pairs appearing in at least one maximum-score structure of the whole
    sequence.  ``best[0][n] == _NEG`` signals proven infeasibility.

    ``orig_pos`` maps each position to its index in the original sequence;
    the minimum-hairpin span is measured on those original coordinates so
    that pairs wrapping a removed pseudoknot block keep their true span.
    """
    n = len(seq)
    if orig_pos is None:
        orig_pos = list(range(n))

    # A *closed* interval [i, j) is one where every position p in [i, j)
    # with a forced mate also has that mate inside [i, j).  The DP only
    # considers closed intervals, so any candidate pair (i, t) that crossed
    # a forced pair (a, b) -- a < i < b < t or i < a < t < b -- would split
    # an endpoint from its mate and is automatically unreachable.
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
                if i <= m < p:  # the right endpoint of a pair fully inside
                    inside_pairs += 1
            closed[i][j] = endpoint_count == 2 * inside_pairs

    def pair_allowed(i: int, t: int) -> bool:
        if orig_pos[t] - orig_pos[i] < MIN_HAIRPIN + 1:
            return False
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
            for t in range(i + 1, j):
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

    # Lexicographically smallest optimal structure per interval.
    lex: List[List[str]] = [[""] * (n + 1) for _ in range(n + 1)]
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            if best[i][j] == _NEG:
                continue  # infeasible interval: no witness string
            target = best[i][j]
            chosen: Optional[str] = None

            if (
                partner[i] == -1
                and closed[i + 1][j]
                and best[i + 1][j] == target
            ):
                chosen = "." + lex[i + 1][j]

            for t in range(i + 1, j):
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
                candidate = "(" + lex[i + 1][t] + ")" + lex[t + 1][j]
                if chosen is None or candidate < chosen:
                    chosen = candidate

            # Feasible intervals always have a witness (table values are
            # maxima over enumerated options).
            assert chosen is not None and len(chosen) == length
            lex[i][j] = chosen

    # Pairs appearing in at least one global optimum: propagate reachability
    # of intervals along optimal decisions from the root [0, n).
    reachable = [[False] * (n + 1) for _ in range(n + 1)]
    reachable[0][n] = True
    can_pair: set = set()

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
                and closed[i + 1][j]
                and best[i + 1][j] == target
            ):
                reachable[i + 1][j] = True

            for t in range(i + 1, j):
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
                can_pair.add((i, t))
                reachable[i + 1][t] = True
                reachable[t + 1][j] = True

    return {
        "best": best,
        "ways": ways,
        "lex": lex,
        "closed": closed,
        "pair_allowed": pair_allowed,
        "can_pair": can_pair,
    }


def fold(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    *,
    pseudoknot_penalty: Optional[int] = None,
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
        * ``pseudoknot``: only present when ``pseudoknot_penalty`` is
          given; ``None`` when the reported structure is pseudoknot-free,
          else a dict ``{"stem1", "stem2", "penalty"}`` with the two
          crossing stems as tuples of ``(i, j)`` pairs.
    :param pseudoknot_penalty: optional integer enabling the single H-type
        pseudoknot mode.  When given (not ``None``), the search also
        considers structures containing exactly one H-type pseudoknot: two
        stems of consecutive nested pairs whose endpoints interleave as
        ``i < k < j < l``.  The objective is the usual pair score minus
        ``pseudoknot_penalty`` (charged once) for pseudoknot structures;
        pseudoknot-free structures stay candidates, so the optimum only
        crosses when it genuinely wins.  In this mode the sequence length
        is limited to ``MAX_PSEUDOKNOT_SEQUENCE_LENGTH`` (24), and in the
        ``structure`` string the second stem is marked with ``[]`` instead
        of ``()``.
    :raises TypeError: wrong value types (including ``bool`` as int).
    :raises InvalidSequenceError: bad length or illegal character.
    :raises ConstraintError: malformed or conflicting constraints.
    :raises UnsatisfiableConstraintError: proven that no admissible structure
        satisfies the forced constraints.
    """
    seq = _validate_sequence(sequence)
    partner, force_up = _normalise_constraints(constraints, seq)
    n = len(seq)

    pk_penalty: Optional[int] = None
    if pseudoknot_penalty is not None:
        pk_penalty = _as_int(pseudoknot_penalty, "pseudoknot_penalty")
        if n > MAX_PSEUDOKNOT_SEQUENCE_LENGTH:
            raise InvalidSequenceError(
                f"sequence length {n} exceeds maximum "
                f"{MAX_PSEUDOKNOT_SEQUENCE_LENGTH} for pseudoknot mode"
            )

    sol = _solve(seq, partner, force_up)
    best = sol["best"]
    ways = sol["ways"]
    lex = sol["lex"]
    closed = sol["closed"]
    pair_allowed = sol["pair_allowed"]
    can_pair = sol["can_pair"]

    if best[0][n] == _NEG:
        raise UnsatisfiableConstraintError(
            "constraints are unsatisfiable: no admissible structure contains "
            "all forced pairs"
        )

    if pk_penalty is None:
        return {
            "score": best[0][n],
            "structure": lex[0][n],
            "count": ways[0][n],
            "pairs": tuple(sorted(can_pair)),
        }

    # ------------------------------------------------------------------
    # Single H-type pseudoknot mode (n <= MAX_PSEUDOKNOT_SEQUENCE_LENGTH).
    #
    # A pseudoknot structure = two stems of consecutive nested pairs whose
    # endpoints interleave (stem1 pairs (lo1+a, hi1-a), stem2 pairs
    # (lo2+a, hi2-a), with lo1 < lo2 <= hi1 < hi2 in the interleaving
    # sense i < k < j < l), plus five intervening/outside segments that
    # fold with the exact non-crossing DP above.  The objective charges
    # pk_penalty once; the pseudoknot-free optimum stays a candidate.
    # ------------------------------------------------------------------
    forced_list = [(p, partner[p]) for p in range(n) if partner[p] > p]
    forced_index = {p: k for k, p in enumerate(forced_list)}
    full_mask = (1 << len(forced_list)) - 1

    # inside_mask[i][j]: bitmask of forced pairs with both endpoints in
    # [i, j); touch_mask[i][j]: forced pairs with at least one endpoint in
    # [i, j).  A topology is feasible only if every forced pair is either
    # one of its stem pairs, lies entirely inside one of its three inner
    # segments, or lies entirely outside the pseudoknot block.
    inside_mask = [[0] * (n + 1) for _ in range(n + 1)]
    touch_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        m_in = 0
        m_touch = 0
        for j in range(i + 1, n + 1):
            p = j - 1
            q = partner[p]
            if q != -1:
                bit = 1 << forced_index[(min(p, q), max(p, q))]
                m_touch |= bit
                if 0 <= q < p and q >= i:
                    m_in |= bit
            inside_mask[i][j] = m_in
            touch_mask[i][j] = m_touch

    # The positions outside the pseudoknot block [lo1, hi2] fold as one
    # non-crossing system on the concatenated sequence seq[:lo1] +
    # seq[hi2+1:]; pairs there may wrap the whole block.  Memoized.
    out_cache: Dict[Tuple[int, int], Dict[str, object]] = {}

    def outside_solve(lo1: int, hi2: int) -> Dict[str, object]:
        key = (lo1, hi2)
        got = out_cache.get(key)
        if got is not None:
            return got
        shift = hi2 + 1 - lo1
        m = n - shift
        out_seq = seq[:lo1] + seq[hi2 + 1:]
        out_partner = [-1] * m
        out_up = [0] * m
        for p in range(lo1):
            out_up[p] = force_up[p]
        for p in range(hi2 + 1, n):
            out_up[p - shift] = force_up[p]
        for a, b in forced_list:
            # Both endpoints outside the block (a pair may also wrap it).
            if b < lo1 or a > hi2 or (a < lo1 and b > hi2):
                na = a if a < lo1 else a - shift
                nb = b if b < lo1 else b - shift
                out_partner[na] = nb
                out_partner[nb] = na
        orig_pos = list(range(lo1)) + list(range(hi2 + 1, n))
        got = _solve(out_seq, out_partner, out_up, orig_pos)
        out_cache[key] = got
        return got

    # All legal stems: (lo, hi, length, score, forced-mask); the pairs are
    # (lo + a, hi - a) for a in range(length) -- consecutive and nested.
    stems: List[Tuple[int, int, int, int, int]] = []
    for lo in range(n):
        for hi in range(lo + MIN_HAIRPIN + 1, n):
            length = 0
            sc = 0
            m = 0
            while True:
                p = lo + length
                q = hi - length
                if q - p < MIN_HAIRPIN + 1 or not pair_allowed(p, q):
                    break
                length += 1
                sc += PAIR_SCORES[(seq[p], seq[q])]
                if partner[p] == q:
                    m |= 1 << forced_index[(p, q)]
                stems.append((lo, hi, length, sc, m))
    stems_by_lo: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    for st in stems:
        stems_by_lo.setdefault(st[0], []).append(st)

    # Pairs appearing in at least one optimal structure of interval
    # [a, b), memoized; mirrors the global reachability pass above.
    ip_memo: Dict[Tuple[int, int], "frozenset"] = {}

    def interval_opt_pairs(a: int, b: int):
        if a >= b or best[a][b] == _NEG:
            return frozenset()
        key = (a, b)
        got = ip_memo.get(key)
        if got is not None:
            return got
        res: set = set()
        target = best[a][b]
        if (
            partner[a] == -1
            and closed[a + 1][b]
            and best[a + 1][b] == target
        ):
            res |= interval_opt_pairs(a + 1, b)
        for t in range(a + MIN_HAIRPIN + 1, b):
            if not pair_allowed(a, t):
                continue
            if not closed[a + 1][t] or not closed[t + 1][b]:
                continue
            s_in = best[a + 1][t]
            s_out = best[t + 1][b]
            if s_in == _NEG or s_out == _NEG:
                continue
            if PAIR_SCORES[(seq[a], seq[t])] + s_in + s_out != target:
                continue
            res.add((a, t))
            res |= interval_opt_pairs(a + 1, t)
            res |= interval_opt_pairs(t + 1, b)
        out = frozenset(res)
        ip_memo[key] = out
        return out

    opt_score = best[0][n]
    opt_count = ways[0][n]
    opt_structure = lex[0][n]
    opt_pairs = set(can_pair)
    opt_pk: Optional[Dict[str, object]] = None

    for lo1, hi1, s1, sc1, m1 in stems:
        for k in range(lo1 + s1, hi1 - s1 + 1):
            if best[lo1 + s1][k] == _NEG:
                continue
            for lo2, hi2, s2, sc2, m2 in stems_by_lo.get(k, ()):
                # stem2's left half must end before stem1's right half
                # begins, and stem2's right half must start after hi1.
                if lo2 + s2 > hi1 - s1 + 1 or hi2 - s2 < hi1:
                    continue
                out = outside_solve(lo1, hi2)
                out_best = out["best"]
                m_out = lo1 + (n - hi2 - 1)
                if out_best[0][m_out] == _NEG:
                    continue
                regions = (
                    (lo1 + s1, lo2),
                    (lo2 + s2, hi1 - s1 + 1),
                    (hi1 + 1, hi2 - s2 + 1),
                )
                mask = (
                    m1 | m2 | (full_mask & ~touch_mask[lo1][hi2 + 1])
                )
                val = sc1 + sc2 - pk_penalty + out_best[0][m_out]
                cnt = out["ways"][0][m_out]
                feasible = True
                for a, b in regions:
                    rb = best[a][b]
                    if rb == _NEG:
                        feasible = False
                        break
                    val += rb
                    cnt = cnt * ways[a][b] % MOD
                    mask |= inside_mask[a][b]
                if not feasible or mask != full_mask or val < opt_score:
                    continue
                # (New) optimum: build the witness structure.
                chars = ["."] * n
                shift = hi2 + 1 - lo1
                for idx, ch in enumerate(out["lex"][0][m_out]):
                    chars[idx if idx < lo1 else idx + shift] = ch
                for a, b in regions:
                    chars[a:b] = lex[a][b]
                for a in range(s1):
                    chars[lo1 + a] = "("
                    chars[hi1 - a] = ")"
                for a in range(s2):
                    chars[lo2 + a] = "["
                    chars[hi2 - a] = "]"
                structure = "".join(chars)
                stem1 = tuple((lo1 + a, hi1 - a) for a in range(s1))
                stem2 = tuple((lo2 + a, hi2 - a) for a in range(s2))
                topo_pairs = set(stem1) | set(stem2)
                for a, b in regions:
                    topo_pairs |= interval_opt_pairs(a, b)
                for a, b in out["can_pair"]:
                    topo_pairs.add(
                        (
                            a if a < lo1 else a + shift,
                            b if b < lo1 else b + shift,
                        )
                    )
                pk_info = {
                    "stem1": stem1,
                    "stem2": stem2,
                    "penalty": pk_penalty,
                }
                if val > opt_score:
                    opt_score = val
                    opt_count = cnt
                    opt_structure = structure
                    opt_pairs = topo_pairs
                    opt_pk = pk_info
                else:
                    opt_count = (opt_count + cnt) % MOD
                    opt_pairs |= topo_pairs
                    if structure < opt_structure:
                        opt_structure = structure
                        opt_pk = pk_info

    return {
        "score": opt_score,
        "structure": opt_structure,
        "count": opt_count,
        "pairs": tuple(sorted(opt_pairs)),
        "pseudoknot": opt_pk,
    }
