"""Test suite for the rna_fold package.

Correctness is anchored by an independent brute-force reference
(``reference_impl.py``) that exhaustively enumerates every non-crossing
matching for sequences of length <= 14 without calling the tested core.
"""

import random
import unittest

from reference_impl import brute_force_fold, pairs_of

from rna_fold import (
    MAX_LENGTH,
    MOD,
    ConstraintError,
    Constraints,
    FoldResult,
    SequenceError,
    fold,
)


def gen_sequences(n):
    """Deterministic test sequences of length n (varied alphabets)."""
    seqs = []
    for t, alphabet in enumerate(("ACGU", "GC", "GU", "AC")):
        rng = random.Random(20260923 + 31 * n + t)
        seqs.append("".join(rng.choice(alphabet) for _ in range(n)))
    if n >= 8:
        flank = n // 4
        seqs.append("G" * flank + "A" * (n - 2 * flank) + "C" * flank)
    return seqs


def assert_matches_reference(testcase, seq, result, expected):
    score, count, lexmin, pairs = expected
    testcase.assertEqual(result.score, score, f"score for {seq!r}")
    testcase.assertEqual(result.count, count % MOD, f"count for {seq!r}")
    testcase.assertEqual(result.structure, lexmin, f"structure for {seq!r}")
    testcase.assertEqual(result.pairs, pairs, f"pairs for {seq!r}")


class SequenceValidationTest(unittest.TestCase):
    def test_empty_sequence(self):
        result = fold("")
        self.assertEqual(result, FoldResult(0, "", 1, frozenset()))

    def test_sequence_type_rejected(self):
        for bad in (None, 123, 4.5, ["A", "C"], b"ACGU"):
            with self.subTest(bad=bad):
                with self.assertRaises(SequenceError):
                    fold(bad)

    def test_invalid_characters(self):
        for seq, pos in (("ACGUX", 4), ("acgu", 0), ("ACG N", 3), ("ACGU1", 4)):
            with self.subTest(seq=seq):
                with self.assertRaises(SequenceError) as ctx:
                    fold(seq)
                self.assertIn(f"position {pos}", str(ctx.exception))

    def test_length_limit(self):
        with self.assertRaises(SequenceError) as ctx:
            fold("A" * (MAX_LENGTH + 1))
        self.assertIn(str(MAX_LENGTH), str(ctx.exception))
        result = fold("A" * MAX_LENGTH)  # exactly at the limit: accepted
        self.assertEqual(result.score, 0)
        self.assertEqual(result.structure, "." * MAX_LENGTH)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.pairs, frozenset())


class BasicModelTest(unittest.TestCase):
    def test_pair_scores(self):
        self.assertEqual(fold("GAAAAC").score, 3)   # GC
        self.assertEqual(fold("ACCCCU").score, 2)   # AU
        self.assertEqual(fold("GAAAU").score, 1)    # GU wobble
        self.assertEqual(fold("UAAAG").score, 1)    # UG wobble

    def test_gu_wobble_structure(self):
        result = fold("GAAAU")
        self.assertEqual(result.structure, "(...)")
        self.assertEqual(result.count, 1)
        self.assertEqual(result.pairs, frozenset({(0, 4)}))

    def test_min_loop_enforced(self):
        self.assertEqual(fold("GAAAC").score, 3)   # span 4: allowed
        self.assertEqual(fold("GAAC").score, 0)    # span 3: forbidden
        self.assertEqual(fold("GAAC").structure, "....")

    def test_no_pairs_possible(self):
        result = fold("AAAAAAAA")
        self.assertEqual((result.score, result.count), (0, 1))
        self.assertEqual(result.structure, "........")
        self.assertEqual(result.pairs, frozenset())

    def test_tied_optima(self):
        # (0, 4) and (0, 8) are both CG pairs scoring 3 and cannot combine.
        result = fold("CAAAGAAAG")
        self.assertEqual(result.score, 3)
        self.assertEqual(result.count, 2)
        # ')' (0x29) sorts before '.' (0x2E), so the earlier closing
        # bracket wins the lexicographic comparison.
        self.assertEqual(result.structure, "(...)....")
        self.assertEqual(result.pairs, frozenset({(0, 4), (0, 8)}))

    def test_independent_pairs_combine(self):
        result = fold("GAAACGAAAC")
        self.assertEqual(result.score, 6)
        self.assertEqual(result.structure, "(...)(...)")
        self.assertEqual(result.count, 1)
        self.assertEqual(result.pairs, frozenset({(0, 4), (5, 9)}))


class ConstraintTest(unittest.TestCase):
    def test_forced_pair_applied(self):
        result = fold("GAAAC", {"forced_pairs": [(0, 4)]})
        self.assertEqual(result.structure, "(...)")
        self.assertEqual(result.score, 3)
        self.assertEqual(result.pairs, frozenset({(0, 4)}))

    def test_forced_pair_can_lower_score(self):
        # Unconstrained optimum is 6 via (0,4)+(5,9); forcing (0,9) blocks it.
        result = fold("GAAACGAAAC", {"forced_pairs": [(0, 9)]})
        self.assertEqual(result.score, 3)
        self.assertEqual(result.structure, "(........)")
        self.assertEqual(result.count, 1)
        self.assertEqual(result.pairs, frozenset({(0, 9)}))

    def test_forced_unpaired_respected(self):
        self.assertEqual(fold("GAAAC", {"unpaired": {0}}).score, 0)
        self.assertEqual(fold("GAAAC", {"unpaired": {4}}).score, 0)
        result = fold("GAAACGAAAC", {"unpaired": {0, 9}})
        self.assertEqual(result.score, 0)

    def test_constraints_dataclass(self):
        result = fold("GAAAC", Constraints(forced_pairs=frozenset({(0, 4)}),
                                           unpaired=frozenset({1})))
        self.assertEqual(result.structure, "(...)")

    def test_crossing_forced_pairs_rejected(self):
        seq = "GAAAC" * 4  # G at 0,5,10,15 and C at 4,9,14,19
        with self.assertRaises(ConstraintError) as ctx:
            fold(seq, {"forced_pairs": [(0, 9), (5, 14)]})
        self.assertIn("cross", str(ctx.exception))

    def test_nested_forced_pairs_accepted(self):
        seq = "G" + "CAAAG" + "AAAA" + "C"  # (0,10) GC outer, (1,5) CG inner
        result = fold(seq, {"forced_pairs": [(0, 10), (1, 5)]})
        self.assertEqual(result.structure, "((...)....)")
        self.assertEqual(result.score, 6)

    def test_forced_pair_conflicts_with_unpaired(self):
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [(0, 4)], "unpaired": {4}})

    def test_forced_pair_invalid_bases(self):
        with self.assertRaises(ConstraintError) as ctx:
            fold("AAAAAA", {"forced_pairs": [(0, 5)]})
        self.assertIn("cannot pair", str(ctx.exception))

    def test_forced_pair_min_loop(self):
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [(0, 3)]})

    def test_forced_pair_shared_position(self):
        with self.assertRaises(ConstraintError) as ctx:
            fold("CAAAGAAAC", {"forced_pairs": [(0, 4), (4, 8)]})
        self.assertIn("already used", str(ctx.exception))

    def test_forced_pair_duplicate(self):
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [(0, 4), (0, 4)]})

    def test_forced_pair_reversed(self):
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [(4, 0)]})

    def test_position_type_errors(self):
        bad_positions = [True, 4.0, float("nan"), float("inf"), None, "4"]
        for bad in bad_positions:
            with self.subTest(bad=bad):
                with self.assertRaises(ConstraintError):
                    fold("GAAAC", {"forced_pairs": [(0, bad)]})
                with self.assertRaises(ConstraintError):
                    fold("GAAAC", {"unpaired": [bad]})

    def test_position_out_of_range(self):
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [(0, 5)]})
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [(-1, 4)]})
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"unpaired": [5]})

    def test_malformed_constraints(self):
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"bogus": 1})
        with self.assertRaises(ConstraintError):
            fold("GAAAC", 42)
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [(0, 1, 2)]})
        with self.assertRaises(ConstraintError):
            fold("GAAAC", {"forced_pairs": [7]})


class BruteForceComparisonTest(unittest.TestCase):
    """Exhaustive comparison against the independent reference (n <= 14)."""

    def test_unconstrained(self):
        for n in range(0, 15):
            for seq in gen_sequences(n):
                with self.subTest(n=n, seq=seq):
                    expected = brute_force_fold(seq)
                    assert_matches_reference(self, seq, fold(seq), expected)

    def test_forced_unpaired(self):
        rng = random.Random(777)
        for n in range(4, 13):
            for seq in gen_sequences(n):
                unpaired = frozenset(i for i in range(n) if rng.random() < 0.2)
                with self.subTest(n=n, seq=seq, unpaired=unpaired):
                    expected = brute_force_fold(seq, unpaired=unpaired)
                    result = fold(seq, {"unpaired": unpaired})
                    assert_matches_reference(self, seq, result, expected)

    def test_forced_pairs(self):
        for n in range(5, 13):
            for seq in gen_sequences(n):
                _, _, lexmin, _ = brute_force_fold(seq)
                all_pairs = sorted(pairs_of(lexmin))
                if not all_pairs:
                    continue
                for forced in (all_pairs, all_pairs[::2]):
                    forced = frozenset(forced)
                    with self.subTest(n=n, seq=seq, forced=forced):
                        expected = brute_force_fold(seq, forced_pairs=forced)
                        result = fold(seq, {"forced_pairs": forced})
                        assert_matches_reference(self, seq, result, expected)


class LargeInputTest(unittest.TestCase):
    """Self-consistency checks at the interface boundary (n = 120)."""

    @staticmethod
    def check_consistent(seq, result, forced=frozenset(), unpaired=frozenset()):
        assert len(result.structure) == len(seq)
        assert set(result.structure) <= {"(", ")", "."}
        pairs = pairs_of(result.structure)
        scores = {"AU": 2, "UA": 2, "GC": 3, "CG": 3, "GU": 1, "UG": 1}
        total = 0
        for a, b in pairs:
            assert b - a >= 4, (a, b)
            assert seq[a] + seq[b] in scores, (a, b)
            assert a not in unpaired and b not in unpaired
            total += scores[seq[a] + seq[b]]
        assert total == result.score, (total, result.score)
        assert forced <= pairs
        assert pairs <= result.pairs
        for a, b in result.pairs:
            assert 0 <= a < b < len(seq)
            assert b - a >= 4
            assert seq[a] + seq[b] in scores
        assert 1 <= result.count < MOD

    def test_max_length_unconstrained(self):
        rng = random.Random(120)
        seq = "".join(rng.choice("ACGU") for _ in range(MAX_LENGTH))
        self.check_consistent(seq, fold(seq))

    def test_max_length_constrained(self):
        rng = random.Random(121)
        seq = "".join(rng.choice("ACGU") for _ in range(MAX_LENGTH))
        first = fold(seq)
        structure_pairs = pairs_of(first.structure)
        forced = frozenset(sorted(structure_pairs)[::2])
        unpaired = frozenset(
            i for i, ch in enumerate(first.structure) if ch == "."
        )
        result = fold(seq, Constraints(forced_pairs=forced,
                                       unpaired=unpaired))
        self.check_consistent(seq, result, forced=forced, unpaired=unpaired)
        # Forcing every unpaired position of an optimum pins the structure.
        self.assertEqual(result.structure, first.structure)
        self.assertEqual(result.score, first.score)


if __name__ == "__main__":
    unittest.main()
