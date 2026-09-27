"""RNA secondary structure prediction with forced pairings.

Public entry points: :func:`fold` and :func:`fold_batch`.

The implementation is an exact, non-crossing (pseudoknot-free) interval
dynamic program.  It performs an exhaustive search over every admissible
structure -- it never samples -- so a returned optimum is proven optimal and a
rejected constraint set is proven unsatisfiable.  See README.md for the full
contract.
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import hmac
import json
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "fold",
    "fold_batch",
    "InvalidSequenceError",
    "ConstraintError",
    "UnsatisfiableConstraintError",
    "BudgetExceededError",
    "InvalidCursorError",
    "MIN_HAIRPIN",
    "MAX_SEQUENCE_LENGTH",
    "MAX_BATCH_SIZE",
    "PAIR_SCORES",
    "MOD",
]

MIN_HAIRPIN = 3
"""Minimum number of *unpaired* positions inside a pair (j - i >= 4)."""

MAX_SEQUENCE_LENGTH = 120

MAX_BATCH_SIZE = 1000

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


class InvalidCursorError(ValueError):
    """A batch continuation cursor is malformed or belongs to another input."""


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


def _compute_fold(
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
                candidate = (
                    "(" + lex[i + 1][t] + ")" + lex[t + 1][j]
                )
                if chosen is None or candidate < chosen:
                    chosen = candidate

            # Feasible intervals always have a witness (root is feasible and
            # table values are maxima over enumerated options).
            assert chosen is not None and len(chosen) == length
            lex[i][j] = chosen

    # Pairs appearing in at least one global optimum: propagate reachability
    # of intervals along optimal decisions from the root [0, n).
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
                and closed[i + 1][j]
                and best[i + 1][j] == target
            ):
                reachable[i + 1][j] = True

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
                can_pair.add((i, t))
                reachable[i + 1][t] = True
                reachable[t + 1][j] = True

    return {
        "score": best[0][n],
        "structure": lex[0][n],
        "count": ways[0][n],
        "pairs": tuple(sorted(can_pair)),
        "_model": {
            "sequence": seq,
            "partner": partner,
            "force_up": force_up,
            "closed": closed,
            "best": best,
            "ways": ways,
            "lex": lex,
        },
    }


def fold(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
) -> Dict[str, object]:
    result = _compute_fold(sequence, constraints)
    return {
        "score": result["score"],
        "structure": result["structure"],
        "count": result["count"],
        "pairs": result["pairs"],
    }


def _structure_pairs(structure: str) -> Tuple[Tuple[int, int], ...]:
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            pairs.append((stack.pop(), i))
    return tuple(pairs)


def _input_fingerprint(
    sequence: str,
    partner: List[int],
    force_up: List[int],
) -> str:
    payload = json.dumps(
        {
            "sequence": sequence,
            "partner": partner,
            "force_up": force_up,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


def _encode_cursor(offset: int, fingerprint: str) -> str:
    payload = json.dumps(
        {"v": 1, "offset": offset, "fingerprint": fingerprint},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    signature = hashlib.blake2b(
        payload, key=b"rna-fold-batch-cursor-v1", digest_size=32
    ).digest()
    return (
        base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
        + "."
        + base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    )


def _decode_cursor(cursor: object, expected_fingerprint: str) -> int:
    if not isinstance(cursor, str):
        raise InvalidCursorError(
            f"cursor: expected str, got {type(cursor).__name__}"
        )
    parts = cursor.split(".")
    if len(parts) != 2:
        raise InvalidCursorError("cursor: malformed continuation token")
    try:
        payload_raw = base64.urlsafe_b64decode(
            parts[0] + "=" * (-len(parts[0]) % 4)
        )
        signature = base64.urlsafe_b64decode(
            parts[1] + "=" * (-len(parts[1]) % 4)
        )
        payload = json.loads(payload_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor: malformed continuation token") from exc

    expected_signature = hashlib.blake2b(
        payload_raw,
        key=b"rna-fold-batch-cursor-v1",
        digest_size=32,
    ).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise InvalidCursorError("cursor: signature verification failed")
    if not isinstance(payload, dict) or payload.get("v") != 1:
        raise InvalidCursorError("cursor: unsupported cursor version")
    if payload.get("fingerprint") != expected_fingerprint:
        raise InvalidCursorError(
            "cursor does not belong to this sequence and constraint set"
        )
    offset = payload.get("offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidCursorError("cursor: offset must be a non-negative integer")
    return offset


def _next_optimal_structure(
    frontier: List[Tuple[str, int, str, Tuple[object, ...]]],
    sequence: str,
    model: Dict[str, object],
) -> Optional[str]:
    partner = model["partner"]
    force_up = model["force_up"]
    closed = model["closed"]
    best = model["best"]
    lex = model["lex"]

    def pair_allowed(i: int, t: int) -> bool:
        if force_up[i] or force_up[t]:
            return False
        if partner[i] != -1 and partner[i] != t:
            return False
        if partner[t] != -1 and partner[t] != i:
            return False
        return (sequence[i], sequence[t]) in PAIR_SCORES

    def push(prefix: str, suffix: Tuple[object, ...], order: int) -> None:
        minimum = [prefix]
        for token in suffix:
            if isinstance(token, str):
                minimum.append(token)
            else:
                a, b = token
                minimum.append(lex[a][b])
        heapq.heappush(
            frontier,
            ("".join(minimum), order, prefix, suffix),
        )

    while frontier:
        order = 0
        _, _, prefix, suffix = heapq.heappop(frontier)
        if not suffix:
            return prefix

        token = suffix[0]
        rest = suffix[1:]
        if isinstance(token, tuple):
            i, j = token
            if i >= j:
                push(prefix, rest, order)
                order += 1
                continue
        else:
            push(prefix + token, rest, order)
            order += 1
            continue

        target = best[i][j]

        if (
            partner[i] == -1
            and closed[i + 1][j]
            and best[i + 1][j] == target
        ):
            child_suffix = (".",)
            if i + 1 < j:
                child_suffix += ((i + 1, j),)
            push(prefix, child_suffix + rest, order)
            order += 1

        for t in range(i + MIN_HAIRPIN + 1, j):
            if not pair_allowed(i, t):
                continue
            if not closed[i + 1][t] or not closed[t + 1][j]:
                continue
            inside_score = best[i + 1][t]
            outside_score = best[t + 1][j]
            if inside_score == _NEG or outside_score == _NEG:
                continue
            value = (
                PAIR_SCORES[(sequence[i], sequence[t])]
                + inside_score
                + outside_score
            )
            if value != target:
                continue
            child_suffix: Tuple[object, ...] = (
                "(",
                (i + 1, t),
                ")",
            )
            if t + 1 < j:
                child_suffix += ((t + 1, j),)
            push(prefix, child_suffix + rest, order)
            order += 1

    return None


def fold_batch(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    batch_size: int = 10,
    cursor: Optional[str] = None,
) -> Dict[str, object]:
    """Claim one stable batch of tied-optimal structures.

    Structures are ordered by the same ASCII lexicographic dot-bracket order
    used by :func:`fold`.  The returned cursor is bound to both the sequence
    and the normalized hard constraints; it cannot resume a different input.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size: expected a positive integer")
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be in [1, {MAX_BATCH_SIZE}], got {batch_size}"
        )

    result = _compute_fold(sequence, constraints)
    seq = result["_model"]["sequence"]
    partner = result["_model"]["partner"]
    force_up = result["_model"]["force_up"]
    fingerprint = _input_fingerprint(seq, partner, force_up)
    offset = 0 if cursor is None else _decode_cursor(cursor, fingerprint)

    frontier: List[Tuple[str, int, str, Tuple[object, ...]]] = []
    heapq.heappush(
        frontier,
        (result["structure"], 0, "", ((0, len(seq)),)),
    )

    structures: List[Dict[str, object]] = []
    produced = 0
    next_structure = _next_optimal_structure(frontier, seq, result["_model"])
    while next_structure is not None and produced < offset + batch_size:
        if produced >= offset:
            structures.append(
                {
                    "structure": next_structure,
                    "pairs": _structure_pairs(next_structure),
                }
            )
        produced += 1
        next_structure = _next_optimal_structure(
            frontier, seq, result["_model"]
        )

    has_more = next_structure is not None or bool(frontier)
    next_offset = offset + len(structures)
    return {
        "score": result["score"],
        "count": result["count"],
        "structures": tuple(structures),
        "next_cursor": _encode_cursor(next_offset, fingerprint)
        if has_more
        else None,
        "has_more": has_more,
    }
