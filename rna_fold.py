"""RNA secondary structure: optimum folding and Boltzmann marginals.

Two public entry points:

* :func:`fold` -- an exact, non-crossing (pseudoknot-free) interval dynamic
  program returning a maximum-score structure.  It performs an exhaustive
  search over every admissible structure -- it never samples -- so a returned
  optimum is proven optimal and a rejected constraint set is proven
  unsatisfiable.
* :func:`partition_function` -- the same structure space weighted by
  ``exp(-E / T)`` with integer pair/unpaired energies, evaluated in log
  space; returns ``ln Z`` and base-pair / unpaired marginal probabilities
  for sequences up to :data:`MAX_PARTITION_LENGTH` (20).

See README.md for the full contract.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "fold",
    "partition_function",
    "InvalidSequenceError",
    "ConstraintError",
    "UnsatisfiableConstraintError",
    "BudgetExceededError",
    "MIN_HAIRPIN",
    "MAX_SEQUENCE_LENGTH",
    "MAX_PARTITION_LENGTH",
    "PAIR_SCORES",
    "DEFAULT_PAIR_ENERGIES",
    "MOD",
]

MIN_HAIRPIN = 3
"""Minimum number of *unpaired* positions inside a pair (j - i >= 4)."""

MAX_SEQUENCE_LENGTH = 120

#: The Boltzmann partition function is offered up to this length.  The inside/
#: outside DP itself is O(n**3), but the contract for this iteration caps the
#: stochastic entry point here (short sequences are cross-checked by exhaustive
#: enumeration in the tests).
MAX_PARTITION_LENGTH = 20

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

#: Default integer pair energies used by :func:`partition_function`.
#: Favorable pairs carry negative energy (GC strongest, GU weakest); the
#: table is the energetic counterpart of :data:`PAIR_SCORES`.  Keys are the
#: six oriented two-letter pair names; a user-supplied table must give all
#: six (integer) values.
DEFAULT_PAIR_ENERGIES: Dict[str, int] = {
    "GC": -3,
    "CG": -3,
    "AU": -2,
    "UA": -2,
    "GU": -1,
    "UG": -1,
}

_PAIR_ENERGY_KEYS = frozenset(DEFAULT_PAIR_ENERGIES)

_ALPHABET = frozenset("ACGU")
_NEG = -10**18
_NEG_INF = float("-inf")

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


def _validate_sequence(sequence: object, max_length: int = MAX_SEQUENCE_LENGTH) -> str:
    if not isinstance(sequence, str):
        raise TypeError(
            f"sequence: expected str, got {type(sequence).__name__}"
        )
    n = len(sequence)
    if n > max_length:
        raise InvalidSequenceError(
            f"sequence length {n} exceeds maximum {max_length}"
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
    }


# ---------------------------------------------------------------------------
# Boltzmann partition function and base-pair marginals
# ---------------------------------------------------------------------------


def _logsumexp(values: List[float]) -> float:
    """Numerically stable ``log(sum(exp(v)))``; ``-inf`` inputs are ignored.

    The shift keeps every exponent <= 0 (no overflow); ``math.fsum`` gives a
    correctly rounded sum of the (already small) exponent terms, so the result
    stays accurate even when the spread between terms is large.
    """
    finite = [v for v in values if v != _NEG_INF]
    if not finite:
        return _NEG_INF
    shift = max(finite)
    return shift + math.log(math.fsum(math.exp(v - shift) for v in finite))


def _validate_temperature(temperature: object) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(
            "temperature: expected a positive number, got "
            f"{type(temperature).__name__}"
        )
    try:
        t = float(temperature)
    except OverflowError as exc:
        raise ValueError(
            f"temperature: value {temperature!r} is too large to convert "
            "to a finite float"
        ) from exc
    if math.isnan(t):
        raise ValueError("temperature: NaN is not a valid temperature")
    if math.isinf(t):
        raise ValueError("temperature: Infinity is not a finite temperature")
    if t <= 0.0:
        raise ValueError(
            f"temperature: must be strictly positive, got {temperature!r}"
        )
    return t


def _validate_energies(
    pair_energies: object, unpaired_energy: object
) -> Tuple[Dict[Tuple[str, str], float], float, Dict[str, int]]:
    """Validate the integer energy model; return (oriented table, unpaired, copy)."""
    table: Dict[str, int] = dict(DEFAULT_PAIR_ENERGIES)
    if pair_energies is not None:
        if not isinstance(pair_energies, Mapping):
            raise TypeError(
                "pair_energies: expected a mapping of pair name to integer "
                f"energy, got {type(pair_energies).__name__}"
            )
        unknown = set(pair_energies) - _PAIR_ENERGY_KEYS
        if unknown:
            raise ValueError(
                f"pair_energies: unknown key(s) {sorted(unknown)!r}; allowed "
                f"keys are {sorted(_PAIR_ENERGY_KEYS)!r}"
            )
        for key, val in pair_energies.items():
            table[key] = _as_int(val, f"pair_energies[{key!r}]")

    u = _as_int(unpaired_energy, "unpaired_energy")

    def _as_float(value: int, where: str) -> float:
        try:
            f = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"{where}: integer {value!r} is too large to convert to "
                "a finite float"
            ) from exc
        if not math.isfinite(f):
            raise ValueError(
                f"{where}: integer {value!r} converts to a non-finite "
                "float; energy magnitudes must stay finite"
            )
        return f

    oriented: Dict[Tuple[str, str], float] = {}
    for name, energy in table.items():
        oriented[(name[0], name[1])] = _as_float(energy, f"pair_energies[{name!r}]")
    return oriented, _as_float(u, "unpaired_energy"), table


def partition_function(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    *,
    temperature: "int | float" = 1.0,
    pair_energies: Optional[Mapping[str, int]] = None,
    unpaired_energy: int = 0,
) -> Dict[str, object]:
    """Boltzmann partition function and base-pair marginal probabilities.

    Every admissible non-crossing structure (same legality rules and same
    hard constraints as :func:`fold`) gets Boltzmann weight
    ``exp(-E / T)`` with *integer* energies::

        E(structure) = sum of pair energies over its pairs
                       + unpaired_energy * (#unpaired positions)

    All computation runs in log space (inside/outside DP with a stable
    log-sum-exp), so weights that would overflow or underflow as plain
    exponentials do not hurt the result.

    :param sequence: RNA sequence over ``A C G U`` (case sensitive),
        length <= :data:`MAX_PARTITION_LENGTH` (20).
    :param constraints: same forms and rules as in :func:`fold`.
    :param temperature: positive ``T > 0`` (int or finite float; ``bool``,
        NaN, Infinity and non-positive values are rejected).
    :param pair_energies: optional mapping overriding the default integer
        pair energies (:data:`DEFAULT_PAIR_ENERGIES`: GC/CG=-3, AU/UA=-2,
        GU/UG=-1).  Keys are oriented ``"GC"`` ... ``"UG"``; partial
        overrides are allowed; values must be integers (``bool`` rejected).
    :param unpaired_energy: integer energy contributed by every unpaired
        position (default 0).
    :returns: dict with

        * ``log_partition`` (float): ``ln Z``;
        * ``free_energy`` (float): ``-T * ln Z``;
        * ``temperature`` (float): the temperature used;
        * ``pair_energies`` (dict): the full effective six-key integer table;
        * ``unpaired_energy`` (int);
        * ``pair_prob_matrix``: ``n x n`` list of float with
          ``pair_prob_matrix[i][j]`` the marginal probability of pair
          ``(i, j)`` for ``i < j`` (zero elsewhere);
        * ``pair_probabilities``: tuple of ``(i, j, p)`` for every legal
          pair with ``p > 0``, sorted by ``(i, j)``;
        * ``unpaired_probabilities``: list of float, marginal probability
          that each position is unpaired.

        The marginals satisfy, up to floating-point rounding,
        ``unpaired_probabilities[i] + sum_j matrix[i][j] + sum_j matrix[j][i]
        == 1``.
    :raises TypeError: wrong value types (including ``bool`` as a number).
    :raises ValueError: non-positive/NaN/Infinite temperature or unknown
        energy keys.
    :raises InvalidSequenceError: bad length (> 20) or illegal character.
    :raises ConstraintError: malformed or conflicting constraints.
    :raises UnsatisfiableConstraintError: proven that no admissible
        structure satisfies the forced constraints.
    """
    seq = _validate_sequence(sequence, MAX_PARTITION_LENGTH)
    partner, force_up = _normalise_constraints(constraints, seq)
    temp = _validate_temperature(temperature)
    energy_table, e_unpaired, effective = _validate_energies(
        pair_energies, unpaired_energy
    )
    n = len(seq)
    beta = 1.0 / temp
    w_unpaired = -e_unpaired * beta

    # Same closed-interval mechanism as fold(): the DP only moves across
    # intervals that do not strand a forced endpoint from its mate.
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
                if i <= m < p:
                    inside_pairs += 1
            closed[i][j] = endpoint_count == 2 * inside_pairs

    def pair_allowed(i: int, t: int) -> bool:
        if force_up[i] or force_up[t]:
            return False
        if partner[i] != -1 and partner[i] != t:
            return False
        if partner[t] != -1 and partner[t] != i:
            return False
        return (seq[i], seq[t]) in energy_table

    # ---- Inside pass: Q[i][j] = ln sum of weights on interval [i, j) -----
    q_in: List[List[float]] = [
        [_NEG_INF] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        q_in[i][i] = 0.0  # empty interval has one structure, weight 1

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            terms: List[float] = []

            # i unpaired.
            if partner[i] == -1 and closed[i + 1][j]:
                terms.append(w_unpaired + q_in[i + 1][j])

            # i paired with t.
            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t):
                    continue
                if not closed[i + 1][t] or not closed[t + 1][j]:
                    continue
                q1 = q_in[i + 1][t]
                q2 = q_in[t + 1][j]
                if q1 == _NEG_INF or q2 == _NEG_INF:
                    continue
                w_pair = -energy_table[(seq[i], seq[t])] * beta
                terms.append(w_pair + q1 + q2)

            q_in[i][j] = _logsumexp(terms)

    log_z = q_in[0][n]
    if log_z == _NEG_INF:
        raise UnsatisfiableConstraintError(
            "constraints are unsatisfiable: no admissible structure contains "
            "all forced pairs"
        )

    # ---- Outside pass: q_out[i][j] for contexts enclosing [i, j) ---------
    q_out: List[List[float]] = [
        [_NEG_INF] * (n + 1) for _ in range(n + 1)
    ]
    q_out[0][n] = 0.0

    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length
            qo = q_out[i][j]
            if qo == _NEG_INF:
                continue

            # Context where i is unpaired -> remainder interval [i+1, j).
            if (
                partner[i] == -1
                and closed[i + 1][j]
                and q_in[i + 1][j] != _NEG_INF
            ):
                cand = qo + w_unpaired
                # Every qualifying context adds weight mass -- this must be
                # logsumexp even when cand is smaller than the accumulated
                # value, never a max-style conditional update.
                q_out[i + 1][j] = _logsumexp(
                    [q_out[i + 1][j], cand]
                )

            # Context where i pairs with t -> children [i+1, t), [t+1, j).
            for t in range(i + MIN_HAIRPIN + 1, j):
                if not pair_allowed(i, t):
                    continue
                if not closed[i + 1][t] or not closed[t + 1][j]:
                    continue
                q1 = q_in[i + 1][t]
                q2 = q_in[t + 1][j]
                if q1 == _NEG_INF or q2 == _NEG_INF:
                    continue
                base = qo - energy_table[(seq[i], seq[t])] * beta
                cand_in = base + q2  # outside context for [i+1, t)
                cand_out = base + q1  # outside context for [t+1, j)
                q_out[i + 1][t] = _logsumexp(
                    [q_out[i + 1][t], cand_in]
                )
                q_out[t + 1][j] = _logsumexp(
                    [q_out[t + 1][j], cand_out]
                )

    # ---- Marginals --------------------------------------------------------
    matrix: List[List[float]] = [
        [0.0] * n for _ in range(n)
    ]
    triples: List[Tuple[int, int, float]] = []
    for i in range(n):
        for t in range(i + MIN_HAIRPIN + 1, n):
            if not pair_allowed(i, t) or not closed[i + 1][t]:
                continue
            # Pair (i, t) is created inside an interval [i, j) for any
            # j > t; q_out[i][j] sums every external context producing it.
            contexts: List[float] = []
            w_pair = -energy_table[(seq[i], seq[t])] * beta
            for j in range(t + 1, n + 1):
                if not closed[t + 1][j]:
                    continue
                q1 = q_in[i + 1][t]
                q2 = q_in[t + 1][j]
                qo = q_out[i][j]
                if (
                    q1 == _NEG_INF
                    or q2 == _NEG_INF
                    or qo == _NEG_INF
                ):
                    continue
                contexts.append(w_pair + q1 + q2 + qo)
            log_p = _logsumexp(contexts)
            if log_p == _NEG_INF:
                continue
            p = math.exp(log_p - log_z)
            if p > 0.0:
                matrix[i][t] = p
                triples.append((i, t, p))

    unpaired: List[float] = []
    for i in range(n):
        contexts = []
        if partner[i] == -1:
            for j in range(i + 1, n + 1):
                if not closed[i + 1][j]:
                    continue
                q1 = q_in[i + 1][j]
                qo = q_out[i][j]
                if q1 == _NEG_INF or qo == _NEG_INF:
                    continue
                contexts.append(w_unpaired + q1 + qo)
        log_p = _logsumexp(contexts)
        unpaired.append(
            0.0 if log_p == _NEG_INF else math.exp(log_p - log_z)
        )

    return {
        "log_partition": log_z,
        "free_energy": -temp * log_z,
        "temperature": temp,
        "pair_energies": effective,
        "unpaired_energy": int(e_unpaired),
        "pair_prob_matrix": matrix,
        "pair_probabilities": tuple(triples),
        "unpaired_probabilities": unpaired,
    }
