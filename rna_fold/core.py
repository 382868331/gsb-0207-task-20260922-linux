"""RNA secondary structure prediction with forced-pair constraints.

Exact Nussinov-style inside/outside dynamic programming over intervals.
The algorithm is fully deterministic and exact: no sampling, no heuristics,
no search budget.  The only source of "no solution" is a set of mutually
inconsistent hard constraints, which is detected up front and reported as
``ConstraintError``.

Model (fixed by the task contract, not a full thermodynamic model):

* sequence length 0..120, alphabet ``A``, ``C``, ``G``, ``U`` (uppercase);
* allowed pairs: AU/UA (score 2), GC/CG (score 3), GU/UG (score 1);
* a pair (i, j) requires at least 3 unpaired bases between i and j
  (i.e. ``j - i >= 4``);
* pairs never cross (no pseudoknots) and every position pairs at most once;
* hard constraints: positions forced to stay unpaired and forced pairs.
"""

from dataclasses import dataclass

MOD = 1_000_000_007
"""Modulus used for the optimal-structure count."""

MAX_LENGTH = 120
"""Maximum accepted sequence length."""

MIN_LOOP = 3
"""Minimum number of unpaired bases between the two ends of a pair."""

_VALID_BASES = frozenset("ACGU")

_PAIR_SCORES = {
    ("A", "U"): 2, ("U", "A"): 2,
    ("G", "C"): 3, ("C", "G"): 3,
    ("G", "U"): 1, ("U", "G"): 1,
}


class FoldError(Exception):
    """Base class for every error raised by :func:`fold`."""


class SequenceError(FoldError, ValueError):
    """The RNA sequence itself is invalid (type, length or alphabet)."""


class ConstraintError(FoldError, ValueError):
    """Constraints are malformed, out of range or mutually inconsistent."""


@dataclass(frozen=True)
class Constraints:
    """Hard constraints for :func:`fold`.

    ``forced_pairs``: iterable of ``(i, j)`` tuples, 0-based, ``i < j``.
    ``unpaired``: iterable of 0-based positions that must stay unpaired.
    """

    forced_pairs: frozenset = frozenset()
    unpaired: frozenset = frozenset()


@dataclass(frozen=True)
class FoldResult:
    """Result of :func:`fold`.

    ``score``: maximum total pair score (exact integer).
    ``structure``: lexicographically smallest dot-bracket string among all
        optimal structures (ASCII order: ``'(' < ')' < '.'``), 0-based.
    ``count``: number of optimal structures, modulo ``MOD``.
    ``pairs``: frozenset of ``(i, j)`` pairs (0-based, ``i < j``) that occur
        in at least one optimal structure.
    """

    score: int
    structure: str
    count: int
    pairs: frozenset


def _validate_sequence(sequence):
    if not isinstance(sequence, str):
        raise SequenceError(
            f"sequence: expected str, got {type(sequence).__name__}"
        )
    if len(sequence) > MAX_LENGTH:
        raise SequenceError(
            f"sequence: length {len(sequence)} exceeds the maximum of "
            f"{MAX_LENGTH}"
        )
    for i, ch in enumerate(sequence):
        if ch not in _VALID_BASES:
            raise SequenceError(
                f"sequence: invalid base {ch!r} at position {i}; "
                "allowed bases are A, C, G, U (uppercase)"
            )
    return sequence


def _as_position(value, what, n):
    # bool is a subclass of int and is rejected explicitly; NaN and Infinity
    # are floats and are rejected by the isinstance check together with every
    # other non-int type.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConstraintError(
            f"{what}: expected an integer position, got {value!r}"
        )
    if not 0 <= value < n:
        raise ConstraintError(
            f"{what}: position {value} out of range [0, {n})"
        )
    return value


def _validate_constraints(constraints, seq, n):
    """Return ``(forced_pairs, unpaired)`` or raise ``ConstraintError``."""
    if constraints is None:
        return [], set()
    if isinstance(constraints, Constraints):
        raw_pairs = constraints.forced_pairs
        raw_unpaired = constraints.unpaired
    elif isinstance(constraints, dict):
        unknown = set(constraints) - {"forced_pairs", "unpaired"}
        if unknown:
            raise ConstraintError(
                f"constraints: unknown keys {sorted(unknown)!r}; supported "
                "keys are 'forced_pairs' and 'unpaired'"
            )
        raw_pairs = constraints.get("forced_pairs") or ()
        raw_unpaired = constraints.get("unpaired") or ()
    else:
        raise ConstraintError(
            "constraints: expected None, a Constraints instance or a dict, "
            f"got {type(constraints).__name__}"
        )

    pairs = []
    owner = {}  # position -> index of the forced pair already using it
    for idx, entry in enumerate(raw_pairs):
        what = f"constraints.forced_pairs[{idx}]"
        try:
            a, b = entry
        except (TypeError, ValueError):
            raise ConstraintError(
                f"{what}: expected an (i, j) pair, got {entry!r}"
            ) from None
        a = _as_position(a, f"{what}[0]", n)
        b = _as_position(b, f"{what}[1]", n)
        if a == b:
            raise ConstraintError(
                f"{what}: a pair needs two distinct positions, got ({a}, {b})"
            )
        if a > b:
            raise ConstraintError(f"{what}: require i < j, got ({a}, {b})")
        if b - a < MIN_LOOP + 1:
            raise ConstraintError(
                f"{what}: pair ({a}, {b}) leaves fewer than {MIN_LOOP} "
                "unpaired bases between the paired positions"
            )
        if (seq[a], seq[b]) not in _PAIR_SCORES:
            raise ConstraintError(
                f"{what}: bases {seq[a]}{seq[b]} at ({a}, {b}) cannot pair "
                "(allowed: AU, UA, GC, CG, GU, UG)"
            )
        for pos in (a, b):
            if pos in owner:
                raise ConstraintError(
                    f"{what}: position {pos} is already used by forced pair "
                    f"{pairs[owner[pos]]}"
                )
        owner[a] = owner[b] = idx
        pairs.append((a, b))

    unpaired = set()
    for idx, value in enumerate(raw_unpaired):
        unpaired.add(_as_position(value, f"constraints.unpaired[{idx}]", n))

    for a, b in pairs:
        for pos in (a, b):
            if pos in unpaired:
                raise ConstraintError(
                    f"constraints: forced pair ({a}, {b}) conflicts with "
                    f"forced-unpaired position {pos}"
                )

    # Crossing check: sorted by left endpoint, keep a stack of open pairs.
    stack = []
    for a, b in sorted(pairs):
        while stack and stack[-1][1] < a:
            stack.pop()
        if stack and stack[-1][1] < b:
            raise ConstraintError(
                f"constraints: forced pairs {stack[-1]} and {(a, b)} cross; "
                "crossing (pseudoknot) pairs are not allowed"
            )
        stack.append((a, b))

    return pairs, unpaired


def fold(sequence, constraints=None):
    """Predict the maximum-score secondary structure of ``sequence``.

    ``constraints`` is ``None``, a :class:`Constraints` instance, or a dict
    with optional keys ``forced_pairs`` and ``unpaired``.  Returns a
    :class:`FoldResult`.  Raises :class:`SequenceError` for an invalid
    sequence and :class:`ConstraintError` for malformed or mutually
    inconsistent constraints.
    """
    seq = _validate_sequence(sequence)
    n = len(seq)
    forced, unpaired = _validate_constraints(constraints, seq, n)

    mate = {}
    for a, b in forced:
        mate[a] = b
        mate[b] = a

    def w_forced(a, b):
        return _PAIR_SCORES[(seq[a], seq[b])]

    # --- Interval validity -------------------------------------------------
    # An interval [i, j) is usable iff it does not cut any forced pair,
    # i.e. it contains as many forced left endpoints as right endpoints.
    # left_eps[i] / right_eps[i]: number of forced left/right endpoints of
    # forced pairs at positions in [0, i) (exclusive prefix sums).
    left_eps = [0] * (n + 1)
    right_eps = [0] * (n + 1)
    for a, b in forced:
        left_eps[a + 1] += 1
        right_eps[b + 1] += 1
    for p in range(n):
        left_eps[p + 1] += left_eps[p]
        right_eps[p + 1] += right_eps[p]

    def valid(i, j):
        return left_eps[j] - left_eps[i] == right_eps[j] - right_eps[i]

    # --- Pair-score table for free positions (0 = cannot pair). ------------
    # Forced-pair endpoints and forced-unpaired positions may not form any
    # additional pair, so they are excluded here.
    wmat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        if i in unpaired or i in mate:
            continue
        row = wmat[i]
        si = seq[i]
        for k in range(i + MIN_LOOP + 1, n):
            if k not in unpaired and k not in mate:
                row[k] = _PAIR_SCORES.get((si, seq[k]), 0)

    # --- Inside DP -----------------------------------------------------------
    # best[i][j]: max score on the valid half-open interval [i, j) (-1 if the
    #   interval cuts a forced pair);
    # cnt[i][j]:  number of optimal structures on [i, j), modulo MOD;
    # lex[i][j]:  lexicographically smallest dot-bracket among them.
    best = [[-1] * (n + 1) for _ in range(n + 1)]
    cnt = [[0] * (n + 1) for _ in range(n + 1)]
    lex = [[""] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        best[i][i] = 0
        cnt[i][i] = 1

    for span in range(1, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            if not valid(i, j):
                continue
            mi = mate.get(i)
            if mi is not None:
                # i is the left endpoint of forced pair (i, mi); validity of
                # [i, j) guarantees mi < j.  The pair is mandatory.
                best[i][j] = w_forced(i, mi) + best[i + 1][mi] + best[mi + 1][j]
                cnt[i][j] = cnt[i + 1][mi] * cnt[mi + 1][j] % MOD
                lex[i][j] = "(" + lex[i + 1][mi] + ")" + lex[mi + 1][j]
                continue
            if i in unpaired:
                best[i][j] = best[i + 1][j]
                cnt[i][j] = cnt[i + 1][j]
                lex[i][j] = "." + lex[i + 1][j]
                continue
            # Free position i: unpaired, or paired with some free k.
            b_score = best[i + 1][j]
            b_count = cnt[i + 1][j]
            b_lex = "." + lex[i + 1][j]
            wi = wmat[i]
            for k in range(i + MIN_LOOP + 1, j):
                w = wi[k]
                if not w:
                    continue
                s_in = best[i + 1][k]
                s_right = best[k + 1][j]
                if s_in < 0 or s_right < 0:
                    continue  # pairing (i, k) would cut a forced pair
                cand = w + s_in + s_right
                if cand < b_score:
                    continue
                c = cnt[i + 1][k] * cnt[k + 1][j] % MOD
                if cand > b_score:
                    b_score = cand
                    b_count = c
                    b_lex = "(" + lex[i + 1][k] + ")" + lex[k + 1][j]
                else:
                    b_count = (b_count + c) % MOD
                    s = "(" + lex[i + 1][k] + ")" + lex[k + 1][j]
                    if s < b_lex:
                        b_lex = s
            best[i][j] = b_score
            cnt[i][j] = b_count
            lex[i][j] = b_lex

    # --- Outside DP ----------------------------------------------------------
    # out[i][j]: max score achievable on the complement of the valid interval
    # [i, j) (positions [0, i) and [j, n), where pairs may also enclose the
    # interval), with all forced pairs outside [i, j) mandatory.  -1 if the
    # interval cuts a forced pair.
    out = [[-1] * (n + 1) for _ in range(n + 1)]
    out[0][n] = 0
    for span in range(n - 1, -1, -1):
        for i in range(0, n - span + 1):
            j = i + span
            if not valid(i, j):
                continue
            cur = -1
            # Expand on the left: bring position p = i - 1 into the outside.
            if i > 0:
                p = i - 1
                mp = mate.get(p)
                if mp is None:
                    base = out[p][j]  # p stays unpaired
                    if base >= 0:
                        cur = max(cur, base)
                    if p not in unpaired:
                        wp = wmat[p]
                        # Pair (p, k) with k >= j encloses [i, j).
                        for k in range(j, n):
                            w = wp[k]
                            if not w:
                                continue
                            s_in = best[j][k]
                            s_out = out[p][k + 1]
                            if s_in >= 0 and s_out >= 0:
                                cur = max(cur, w + s_in + s_out)
                        # Pair (k, p) with k < p lies left of [i, j).
                        for k in range(0, p):
                            w = wmat[k][p]
                            if not w:
                                continue
                            s_in = best[k + 1][p]
                            s_out = out[k][j]
                            if s_in >= 0 and s_out >= 0:
                                cur = max(cur, w + s_in + s_out)
                elif mp > p:
                    # Forced pair (p, mp) encloses [i, j) (validity => mp >= j).
                    s_in = best[j][mp]
                    s_out = out[p][mp + 1]
                    if s_in >= 0 and s_out >= 0:
                        cur = max(cur, w_forced(p, mp) + s_in + s_out)
                else:
                    # Forced pair (mp, p) lies left of [i, j).
                    s_in = best[mp + 1][p]
                    s_out = out[mp][j]
                    if s_in >= 0 and s_out >= 0:
                        cur = max(cur, w_forced(mp, p) + s_in + s_out)
            # Expand on the right: bring position q = j into the outside.
            if j < n:
                q = j
                mq = mate.get(q)
                if mq is None:
                    base = out[i][q + 1]  # q stays unpaired
                    if base >= 0:
                        cur = max(cur, base)
                    if q not in unpaired:
                        wq = wmat[q]
                        # Pair (k, q) with k < i encloses [i, j).
                        for k in range(0, i):
                            w = wmat[k][q]
                            if not w:
                                continue
                            s_in = best[k + 1][i]
                            s_out = out[k][q + 1]
                            if s_in >= 0 and s_out >= 0:
                                cur = max(cur, w + s_in + s_out)
                        # Pair (q, k) with k > q lies right of [i, j).
                        for k in range(q + 1, n):
                            w = wq[k]
                            if not w:
                                continue
                            s_in = best[q + 1][k]
                            s_out = out[i][k + 1]
                            if s_in >= 0 and s_out >= 0:
                                cur = max(cur, w + s_in + s_out)
                elif mq > q:
                    # Forced pair (q, mq) lies right of [i, j).
                    s_in = best[q + 1][mq]
                    s_out = out[i][mq + 1]
                    if s_in >= 0 and s_out >= 0:
                        cur = max(cur, w_forced(q, mq) + s_in + s_out)
                else:
                    # Forced pair (mq, q) encloses [i, j) (validity => mq < i).
                    s_in = best[mq + 1][i]
                    s_out = out[mq][q + 1]
                    if s_in >= 0 and s_out >= 0:
                        cur = max(cur, w_forced(mq, q) + s_in + s_out)
            out[i][j] = cur

    # --- Assemble the result -------------------------------------------------
    target = best[0][n]

    # A free pair (i, k) occurs in some optimal structure iff
    # w(i,k) + best[i+1][k] + out[i][k+1] == best[0][n]: given the pair, its
    # interior and its complement are independent.  Forced pairs occur in
    # every structure by construction.
    opt_pairs = set(forced)
    for i in range(n):
        wi = wmat[i]
        for k in range(i + MIN_LOOP + 1, n):
            w = wi[k]
            if not w:
                continue
            s_in = best[i + 1][k]
            s_out = out[i][k + 1]
            if s_in >= 0 and s_out >= 0 and w + s_in + s_out == target:
                opt_pairs.add((i, k))

    return FoldResult(
        score=target,
        structure=lex[0][n],
        count=cnt[0][n],
        pairs=frozenset(opt_pairs),
    )
