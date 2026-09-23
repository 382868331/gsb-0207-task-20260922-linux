"""Independent brute-force reference for tests.

Enumerates every non-crossing matching of the sequence (positions paired at
most once, at least 3 unpaired bases inside every pair) by plain recursion
and scores each one directly.  It never imports or calls ``rna_fold``; the
two implementations share nothing but the written task contract.  Only
usable for short sequences (the test suite keeps lengths <= 14).
"""

PAIR_SCORE = {
    ("A", "U"): 2, ("U", "A"): 2,
    ("G", "C"): 3, ("C", "G"): 3,
    ("G", "U"): 1, ("U", "G"): 1,
}


def brute_force_fold(seq, forced_pairs=(), unpaired=()):
    """Return (score, count, lexicographically smallest structure, pairs).

    ``pairs`` is the frozenset of (i, j) pairs occurring in at least one
    optimal structure.  Only call with constraint sets that are already
    known to be consistent (the reference does no validation).
    """
    n = len(seq)
    unpaired = frozenset(unpaired)
    mate = {}
    for a, b in forced_pairs:
        mate[a] = b
        mate[b] = a

    best_score = None
    best_count = 0
    best_lex = None
    best_pairs = set()

    def visit(pairs):
        nonlocal best_score, best_count, best_lex, best_pairs
        score = 0
        chars = ["."] * n
        for a, b in pairs:
            score += PAIR_SCORE[(seq[a], seq[b])]
            chars[a] = "("
            chars[b] = ")"
        s = "".join(chars)
        if best_score is None or score > best_score:
            best_score = score
            best_count = 1
            best_lex = s
            best_pairs = set(pairs)
        elif score == best_score:
            best_count += 1
            if s < best_lex:
                best_lex = s
            best_pairs.update(pairs)

    def rec(i, stack, pairs):
        # stack holds the right endpoints of currently open pairs,
        # innermost last; a new pair must close before the innermost one.
        if i == n:
            visit(pairs)
            return
        if stack and stack[-1] == i:
            # Position i is the right endpoint of the innermost open pair.
            rec(i + 1, stack[:-1], pairs)
            return
        limit = stack[-1] if stack else n
        if i in unpaired:
            rec(i + 1, stack, pairs)
            return
        j = mate.get(i)
        if j is not None:
            # Forced pair: it must be taken (its right endpoint is pushed
            # here and popped when the walk reaches it).
            if j < limit:
                rec(i + 1, stack + [j], pairs + [(i, j)])
            return
        # Option 1: i stays unpaired.
        rec(i + 1, stack, pairs)
        # Option 2: pair i with some k inside the current enclosure.
        for k in range(i + 4, limit):
            if k in unpaired or k in mate:
                continue
            if (seq[i], seq[k]) in PAIR_SCORE:
                rec(i + 1, stack + [k], pairs + [(i, k)])

    rec(0, [], [])
    return best_score, best_count, best_lex, frozenset(best_pairs)


def pairs_of(dotbracket):
    """Parse a dot-bracket string into a set of (i, j) pairs."""
    stack = []
    pairs = set()
    for i, ch in enumerate(dotbracket):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            pairs.add((stack.pop(), i))
    assert not stack
    return pairs
