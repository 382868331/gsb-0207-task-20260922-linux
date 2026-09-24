"""RNA secondary structure prediction with forced pairings.

Public entry points:

* :func:`fold` -- exact maximum-score non-crossing folding (n <= 120), with
  the lex-smallest optimum, modular optimum count and optimal-pair set.
* :func:`partition_fold` -- Boltzmann partition function at positive
  temperature over integer per-pair / per-unpaired energies, computed in log
  space, with base-pair marginal probabilities (n <= 20).

Both programs perform an exhaustive, deterministic search over every
admissible structure -- they never sample -- so a returned optimum/probability
covers the whole admissible ensemble and a rejected constraint set is proven
unsatisfiable.  See README.md for the full contract.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

__all__ = [
    "fold",
    "partition_fold",
    "InvalidSequenceError",
    "ConstraintError",
    "UnsatisfiableConstraintError",
    "BudgetExceededError",
    "InvalidParameterError",
    "MIN_HAIRPIN",
    "MAX_SEQUENCE_LENGTH",
    "MAX_PARTITION_SEQUENCE_LENGTH",
    "PAIR_SCORES",
    "DEFAULT_PAIR_ENERGIES",
    "MOD",
]

MIN_HAIRPIN = 3
"""Minimum number of *unpaired* positions inside a pair (j - i >= 4)."""

MAX_SEQUENCE_LENGTH = 120
"""Length cap for the maximum-score :func:`fold`."""

MAX_PARTITION_SEQUENCE_LENGTH = 20
"""Length cap for the Boltzmann ensemble :func:`partition_fold`."""

MOD = 1_000_000_007

# Independent (symmetric) score table for the maximum-score program.  Anything
# absent is not a legal pair.
PAIR_SCORES: Dict[Tuple[str, str], int] = {
    ("G", "C"): 3,
    ("C", "G"): 3,
    ("A", "U"): 2,
    ("U", "A"): 2,
    ("G", "U"): 1,
    ("U", "G"): 1,
}

# Default integer energies (same units as the temperature) used by
# partition_fold: negative = favourable, so the low-temperature ensemble
# coincides with the maximum-score optimum of fold() (energies are the negated
# scores, unpaired energy 0).  Keys are the six oriented legal pair types.
DEFAULT_PAIR_ENERGIES: Dict[str, int] = {
    "GC": -3,
    "CG": -3,
    "AU": -2,
    "UA": -2,
    "GU": -1,
    "UG": -1,
}

_ALPHABET = frozenset("ACGU")
_LEGAL_PAIR_TYPES = frozenset(DEFAULT_PAIR_ENERGIES)
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


class InvalidParameterError(ValueError):
    """A numeric parameter (temperature, energy) is outside its contract.

    Temperature must be a positive finite number; energies must be finite
    integers and custom pair-energy keys must be legal pair types.  Messages
    identify the offending parameter.
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


def _validate_sequence(
    sequence: object, max_length: int = MAX_SEQUENCE_LENGTH
) -> str:
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
# Core dynamic programs
# ---------------------------------------------------------------------------


def _closed_intervals(n: int, partner: List[int]) -> List[List[bool]]:
    """Closedness table for forced pairs.

    A *closed* interval [i, j) is one where every position p in [i, j)
    with a forced mate also has that mate inside [i, j).  A DP that only
    considers closed intervals can never choose a candidate pair crossing a
    forced pair (a < i < b < t or i < a < t < b), because that choice would
    split an endpoint from its mate.  Shared by both public programs.

    closed[i][j] is maintained as: (forced endpoints in interval) ==
    2 * (forced pairs fully inside the interval).
    """
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
    return closed


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

    closed = _closed_intervals(n, partner)

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
# Boltzmann partition function (all arithmetic in log space)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Boltzmann partition function (all probability arithmetic in log space)
# ---------------------------------------------------------------------------

# Edge log-weights are always evaluated *relative to the exact integer
# minimum energy* of their interval (see partition_fold), so every exponent
# argument is <= 0: terms can underflow to 0.0 but can never overflow.
_FLOAT_EXP_BITS = 1024  # |x| >= 2**1024 cannot be a finite float


def _logaddexp(a: float, b: float) -> float:
    """log(exp(a) + exp(b)), stable and overflow-free for finite inputs."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _int_to_float(x: int) -> float:
    """Exact-or-infinite float conversion for arbitrary-size integers."""
    if x.bit_length() > _FLOAT_EXP_BITS:
        return math.inf if x > 0 else -math.inf
    return float(x)


def _validate_temperature(temperature: object) -> float:
    """Validate the positive, finite temperature.

    Type errors (``bool``, non-numeric) are :class:`TypeError`; finite but
    out-of-contract numbers (NaN, Infinity, <= 0) are
    :class:`InvalidParameterError`.
    """
    if isinstance(temperature, bool) or not isinstance(
        temperature, (int, float)
    ):
        raise TypeError(
            "temperature: expected a positive finite number, got "
            f"{type(temperature).__name__}"
        )
    t = float(temperature)
    if math.isnan(t):
        raise InvalidParameterError("temperature: NaN is not valid")
    if math.isinf(t):
        raise InvalidParameterError("temperature: Infinity is not valid")
    if t <= 0.0:
        raise InvalidParameterError(
            f"temperature: must be strictly positive, got {temperature!r}"
        )
    return t


def _energy_int(value: object, where: str) -> int:
    """Integer energy: NaN/Infinity are InvalidParameterError, other
    non-integers remain TypeError (same numeric contract as fold())."""
    try:
        return _as_int(value, where)
    except ValueError as exc:  # NaN / Infinity, not a type error
        raise InvalidParameterError(str(exc)) from exc


def _resolve_energies(
    pair_energies: object, unpaired_energy: object
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], int]:
    """Validate and normalise the integer energy tables.

    ``pair_energies`` is ``None`` (defaults) or a partial/full mapping over
    the six oriented legal pair-type keys ``"GC"`` etc.; unknown keys are
    rejected.  Returns (str-keyed table, tuple-keyed table, unpaired energy).
    """
    up = _energy_int(unpaired_energy, "unpaired_energy")
    table: Dict[str, int] = dict(DEFAULT_PAIR_ENERGIES)
    if pair_energies is None:
        raw = table
    else:
        if not isinstance(pair_energies, Mapping):
            raise TypeError(
                "pair_energies: expected None or a mapping with keys among "
                f"{sorted(_LEGAL_PAIR_TYPES)}, got "
                f"{type(pair_energies).__name__}"
            )
        for key, value in pair_energies.items():
            if not isinstance(key, str) or key not in _LEGAL_PAIR_TYPES:
                raise InvalidParameterError(
                    f"pair_energies: illegal key {key!r}; legal keys are the "
                    f"six oriented pair types {sorted(_LEGAL_PAIR_TYPES)}"
                )
            table[key] = _energy_int(value, f"pair_energies[{key!r}]")
        raw = table
    by_tuple = {(k[0], k[1]): v for k, v in raw.items()}
    return dict(raw), by_tuple, up


def partition_fold(
    sequence: str,
    constraints: "Optional[str | Mapping[str, Iterable[object]]]" = None,
    temperature: "float | int" = 1.0,
    pair_energies: Optional[Mapping[str, int]] = None,
    unpaired_energy: int = 0,
) -> Dict[str, object]:
    """Boltzmann partition function and base-pair marginal probabilities.

    Every admissible non-crossing structure (same legality and hard-constraint
    rules as :func:`fold`) contributes a Boltzmann weight
    ``exp(-E / temperature)``, where its integer energy ``E`` is the sum of
    pair energies plus the unpaired energy for each unpaired position.

    All DP aggregation is done in log space with a stable ``logaddexp``
    helper, so the result is well-defined even when the partition function
    itself is huge or tiny for n <= 20.

    :param sequence: RNA sequence over ``A C G U``, length <= 20.
    :param constraints: same three forms as :func:`fold` (``None``, string,
        or ``{"pairs": ..., "unpaired": ...}`` dict).
    :param temperature: positive finite number ``T > 0`` (same units as the
        energies).  Wrong types (``bool``, non-numeric) raise ``TypeError``;
        ``0``, negatives, ``NaN`` and ``Infinity`` raise
        :class:`InvalidParameterError`.
    :param pair_energies: ``None`` for the defaults
        (GC/CG=-3, AU/UA=-2, GU/UG=-1), or a mapping overriding any subset of
        the six keys ``"GC","CG","AU","UA","GU","UG"`` with finite integers.
    :param unpaired_energy: integer energy assigned to each unpaired position
        (default 0).
    :returns: dict with

        * ``partition_function`` (float): ``Z = sum_S exp(-E(S)/T)``;
        * ``ln_partition_function`` (float): ``ln(Z)``, the numerically
          stable primary quantity;
        * ``free_energy`` (float): ``-T * ln(Z)``;
        * ``temperature`` (float): the validated temperature used;
        * ``min_energy`` (int): exact minimum structure energy over the
          admissible ensemble (arbitrary-precision integer);
        * ``unpaired_probabilities`` (tuple[float, ...]): ``P(position k is
          unpaired)`` for each position;
        * ``pair_probabilities`` (dict): ``{(i, j): P(i paired with j)}`` for
          every legal pair with nonzero marginal, sorted by ``(i, j)``;
        * ``pair_energies`` (dict) / ``unpaired_energy`` (int): the resolved
          integer energy tables actually used.

        For every position ``k`` the marginals satisfy
        ``unpaired_probabilities[k] + sum_j pair_probabilities[(k,j)] == 1``
        (up to floating-point round-off); a forced pair has probability 1.
    :raises InvalidSequenceError: bad characters or length > 20.
    :raises ConstraintError: malformed or conflicting constraints.
    :raises InvalidParameterError: non-positive/non-finite temperature or an
        illegal energy-table key/value.
    :raises UnsatisfiableConstraintError: proven empty admissible ensemble
        (defense in depth; well-formed constraints always admit at least the
        structure containing exactly the forced pairs).
    """
    temp = _validate_temperature(temperature)
    energy_table, epair, e_up = _resolve_energies(
        pair_energies, unpaired_energy
    )
    seq = _validate_sequence(
        sequence, MAX_PARTITION_SEQUENCE_LENGTH
    )
    n = len(seq)
    partner, force_up = _normalise_constraints(constraints, seq)
    closed = _closed_intervals(n, partner)

    def pair_allowed(i: int, t: int) -> bool:
        if force_up[i] or force_up[t]:
            return False
        if partner[i] != -1 and partner[i] != t:
            return False
        if partner[t] != -1 and partner[t] != i:
            return False
        return (seq[i], seq[t]) in epair

    def transitions(i: int, j: int) -> Iterator[Tuple[str, int, int]]:
        """Admissible decompositions of [i, j): (kind, t, energy).

        kind == "u": i is unpaired (energy e_up, sole child [i+1, j), here
        t == i + 1); kind == "p": i pairs with t (the pair energy), children
        [i+1, t) and [t+1, j).  Only closed, admissible splits are yielded.
        """
        if partner[i] == -1 and closed[i + 1][j]:
            yield ("u", i + 1, e_up)
        for t in range(i + MIN_HAIRPIN + 1, j):
            if pair_allowed(i, t):
                if closed[i + 1][t] and closed[t + 1][j]:
                    yield ("p", t, epair[(seq[i], seq[t])])

    # ------------------------------------------------------------------
    # Pass 1: exact integer minimum-energy DP M[i][j].  Arbitrary-precision
    # integers only -- this pass never touches float -- so it anchors every
    # later log-space value and keeps all exponent arguments <= 0.  An
    # infeasible interval is None (no numeric sentinel: user-supplied
    # energies are unbounded integers and could exceed any finite marker).
    # ------------------------------------------------------------------
    M: List[List[Optional[int]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        M[i][i] = 0
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            best: Optional[int] = None
            for kind, t, e in transitions(i, j):
                if kind == "u":
                    child = M[i + 1][j]
                    if child is None:
                        continue
                    v = e + child
                else:
                    m_in, m_out = M[i + 1][t], M[t + 1][j]
                    if m_in is None or m_out is None:
                        continue
                    v = e + m_in + m_out
                if best is None or v < best:
                    best = v
            M[i][j] = best

    if M[0][n] is None:
        raise UnsatisfiableConstraintError(
            "constraints are unsatisfiable: no admissible structure contains "
            "all forced pairs"
        )
    min_energy = M[0][n]

    def shifted_term(extra: int, child_mins: List[int], base: int) -> float:
        """Relative log weight of one decomposition: always <= 0.

        Returns -(extra + sum(child minima) - base) / T, the log of the
        edge's total weight relative to the minimum energy of its interval.
        A gap too large for a finite float is already below machine zero
        (gap >= 2**1024 -> exp(-gap/T) underflows), so -inf is exact there.
        """
        gap = extra + sum(child_mins) - base
        if gap <= 0:
            return 0.0
        if gap.bit_length() > _FLOAT_EXP_BITS:
            return _NEG_INF
        return -float(gap) / temp

    # ------------------------------------------------------------------
    # Pass 2: inside partition DP in log space.  Q[i][j] holds the log
    # partition function of [i, j) SHIFTED by +M[i][j]/T, so it lies in
    # (-inf, 0]: the minimum-energy term contributes exp(0) = 1.
    # ------------------------------------------------------------------
    Q: List[List[float]] = [
        [_NEG_INF] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        Q[i][i] = 0.0

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            base = M[i][j]
            if base is None:
                continue  # infeasible interval: Q stays -inf
            acc = _NEG_INF
            for kind, t, e in transitions(i, j):
                if kind == "u":
                    child = M[i + 1][j]
                    if child is None:
                        continue
                    d = shifted_term(e, [child], base)
                    acc = _logaddexp(acc, d + Q[i + 1][j])
                else:
                    m_in, m_out = M[i + 1][t], M[t + 1][j]
                    if m_in is None or m_out is None:
                        continue
                    d = shifted_term(e, [m_in, m_out], base)
                    acc = _logaddexp(
                        acc, d + Q[i + 1][t] + Q[t + 1][j]
                    )
            Q[i][j] = acc

    # Absolute ln Z: minimum term exactly, plus the shifted log of the
    # ensemble around it.  Big integers are converted with overflow guard.
    ln_z = _int_to_float(-min_energy) / temp + Q[0][n]

    # ------------------------------------------------------------------
    # Pass 3: outside DP in matching shifted frames.  R[i][j] is the log
    # total context weight of subproblem [i, j) (path edges + sibling
    # subtrees), expressed relative to M[i][j] so that
    # exp(R) * Z'[i][j] sums full-structure shifted weights reaching [i, j).
    # ------------------------------------------------------------------
    R: List[List[float]] = [
        [_NEG_INF] * (n + 1) for _ in range(n + 1)
    ]
    R[0][n] = 0.0

    for length in range(n, 0, -1):
        for i in range(0, n - length + 1):
            j = i + length
            r = R[i][j]
            if r == _NEG_INF:
                continue
            base = M[i][j]
            assert base is not None
            for kind, t, e in transitions(i, j):
                if kind == "u":
                    child = M[i + 1][j]
                    if child is None:
                        continue
                    d = shifted_term(e, [child], base)
                    R[i + 1][j] = _logaddexp(
                        R[i + 1][j], r + d
                    )
                else:
                    m_in, m_out = M[i + 1][t], M[t + 1][j]
                    if m_in is None or m_out is None:
                        continue
                    d = shifted_term(e, [m_in, m_out], base)
                    R[i + 1][t] = _logaddexp(
                        R[i + 1][t], r + d + Q[t + 1][j]
                    )
                    R[t + 1][j] = _logaddexp(
                        R[t + 1][j], r + d + Q[i + 1][t]
                    )

    # ------------------------------------------------------------------
    # Pass 4: event marginals.  The interval decomposition is a unique tree
    # per structure, and each position is decided exactly once (at the node
    # where it is the leftmost base), so the edge weights partition Z by the
    # event "i unpaired" / "i paired with t".
    # ------------------------------------------------------------------
    unpaired_logs: List[float] = [_NEG_INF] * n
    pair_logs: Dict[Tuple[int, int], float] = {}
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            r = R[i][j]
            if r == _NEG_INF:
                continue
            base = M[i][j]
            assert base is not None
            for kind, t, e in transitions(i, j):
                if kind == "u":
                    child = M[i + 1][j]
                    if child is None:
                        continue
                    d = shifted_term(e, [child], base)
                    unpaired_logs[i] = _logaddexp(
                        unpaired_logs[i], r + d + Q[i + 1][j]
                    )
                else:
                    m_in, m_out = M[i + 1][t], M[t + 1][j]
                    if m_in is None or m_out is None:
                        continue
                    d = shifted_term(e, [m_in, m_out], base)
                    term = r + d + Q[i + 1][t] + Q[t + 1][j]
                    pair_logs[(i, t)] = _logaddexp(
                        pair_logs.get((i, t), _NEG_INF), term
                    )

    # Probabilities are event shifted-weight over total shifted-weight;
    # every exponent argument is <= 0 (only underflow, never overflow).
    total_shifted = Q[0][n]
    unpaired_prob = tuple(
        math.exp(unpaired_logs[k] - total_shifted) for k in range(n)
    )
    pair_prob: Dict[Tuple[int, int], float] = {}
    for key in sorted(pair_logs):
        p = math.exp(pair_logs[key] - total_shifted)
        if p > 0.0:
            pair_prob[key] = p
    # Forced pairs occur in every admissible structure: pin exactly 1.
    for k in range(n):
        if k < partner[k]:
            pair_prob[(k, partner[k])] = 1.0

    try:
        z_value = math.exp(ln_z)
    except OverflowError:
        z_value = math.inf

    return {
        "partition_function": z_value,
        "ln_partition_function": ln_z,
        "free_energy": -temp * ln_z,
        "temperature": temp,
        "min_energy": min_energy,
        "unpaired_probabilities": unpaired_prob,
        "pair_probabilities": pair_prob,
        "pair_energies": energy_table,
        "unpaired_energy": int(e_up),
    }
