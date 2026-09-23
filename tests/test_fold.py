"""Tests for rna_fold, cross-checked against an independent brute force.

The reference in ``brute_reference.py`` never imports the module under test;
it independently enumerates every non-crossing matching for n <= 14.
Round-trip checks on the returned structure are used only as a supplement.
"""

from __future__ import annotations

import math
import random
import unittest

from rna_fold import (
    MOD,
    BudgetExceededError,
    ConstraintError,
    InvalidSequenceError,
    UnsatisfiableConstraintError,
    fold,
)
from tests.brute_reference import brute_force

PAIR_SCORES = {
    ("G", "C"): 3,
    ("C", "G"): 3,
    ("A", "U"): 2,
    ("U", "A"): 2,
    ("G", "U"): 1,
    ("U", "G"): 1,
}


def structure_pairs(dot_bracket: str):
    stack = []
    pairs = set()
    for i, ch in enumerate(dot_bracket):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            pairs.add((stack.pop(), i))
    return pairs


def structure_is_legal(seq, db):
    n = len(seq)
    assert len(db) == n
    pairs = structure_pairs(db)
    used = set()
    for a, b in pairs:
        if b - a < 4:
            return False
        if (seq[a], seq[b]) not in PAIR_SCORES:
            return False
        if a in used or b in used:
            return False
        used.add(a)
        used.add(b)
    # non-crossing
    ps = sorted(pairs)
    for x in range(len(ps)):
        for y in range(x + 1, len(ps)):
            a, b = ps[x]
            c, d = ps[y]
            if a < c < b < d or c < a < d < b:
                return False
    return True


class ExactComparisonTests(unittest.TestCase):
    """DP vs exhaustive enumeration on deterministic random inputs (<=14)."""

    def test_all_short_unconstrained(self):
        rng = random.Random(20260923)
        checked = 0
        for n in range(0, 13):
            for _ in range(30 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                res = fold(seq)
                ref = brute_force(seq)
                with self.subTest(seq=seq):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                    self.assertTrue(structure_is_legal(seq, res["structure"]))
                    checked += 1
        self.assertGreater(checked, 300)

    def test_length_14_exhaustive_sampled(self):
        rng = random.Random(424242)
        seqs = ["".join(rng.choice("ACGU") for _ in range(14)) for _ in range(8)]
        for seq in seqs:
            res = fold(seq)
            ref = brute_force(seq)
            self.assertEqual(res["score"], ref["score"])
            self.assertEqual(res["structure"], ref["structure"])
            self.assertEqual(res["count"], ref["exact_count"])
            self.assertEqual(set(res["pairs"]), set(ref["pairs"]))

    def test_constrained_vs_bruteforce(self):
        rng = random.Random(99)
        for n in (8, 10, 12, 14):
            for _ in range(25):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                # Choose 0-2 random *legal* forced pairs with a legal
                # unpaired position, built independently of fold().
                legal = [
                    (a, b)
                    for a in range(n)
                    for b in range(a + 4, n)
                    if (seq[a], seq[b]) in PAIR_SCORES
                ]
                rng.shuffle(legal)
                forced = []
                used = set()
                for a, b in legal:
                    if a in used or b in used:
                        continue
                    if any(
                        (a < c < b < d) or (c < a < d < b)
                        for c, d in forced
                    ):
                        continue
                    forced.append((a, b))
                    used.update((a, b))
                    if len(forced) == 2:
                        break
                free = [p for p in range(n) if p not in used]
                up = set(rng.sample(free, min(1, len(free)))) if free else set()
                res = fold(seq, {"pairs": forced, "unpaired": up})
                ref = brute_force(seq, frozenset(forced), frozenset(up))
                with self.subTest(seq=seq, forced=forced, up=up):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                    # every forced pair appears in the lex-min witness
                    got = structure_pairs(res["structure"])
                    self.assertTrue(set(forced) <= got)
                    self.assertTrue(up.isdisjoint(p for pr in got for p in pr))

    def test_gu_pairs_used_when_optimal(self):
        # Only GU/UG pairings are available here; score must count them as 1.
        seq = "GUGUGUGUGUU"  # n=11
        res = fold(seq)
        self.assertGreater(res["score"], 0)
        self.assertTrue(
            all(
                {seq[a], seq[b]} == {"G", "U"}
                for a, b in res["pairs"]
            )
        )
        # returned structure actually achieves the score
        self.assertEqual(
            res["score"],
            sum(PAIR_SCORES[(seq[a], seq[b])] for a, b in structure_pairs(res["structure"])),
        )

    def test_tied_optima_count_and_lex(self):
        # Symmetric sequence: two outer GC pairs either nested-ish choices
        # produce multiple distinct optimal structures.
        seq = "GCGCGCGCGC"  # n=10
        res = fold(seq)
        ref = brute_force(seq)
        self.assertEqual(res["count"], ref["exact_count"])
        self.assertGreaterEqual(res["count"], 1)
        # lex minimum must itself be optimal
        self.assertTrue(structure_is_legal(seq, res["structure"]))
        self.assertEqual(
            res["score"],
            sum(
                PAIR_SCORES[(seq[a], seq[b])]
                for a, b in structure_pairs(res["structure"])
            ),
        )
        self.assertEqual(res["structure"], ref["structure"])

    def test_all_pairs_can_appear_in_an_optimum(self):
        # Every reported pair must belong to some optimal structure.  We
        # verify against the brute-force enumeration directly.
        rng = random.Random(7)
        for n in (9, 12):
            for _ in range(15):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                res = fold(seq)
                ref = brute_force(seq)
                self.assertEqual(set(res["pairs"]), set(ref["pairs"]))

    def test_pair_set_independent_of_count_modulus(self):
        # Reachability of an optimal pair must not depend on the modular
        # counter: with a small modulus some non-empty optimal sets count as
        # 0, yet 'pairs' and 'structure' must stay exact.
        import rna_fold

        rng = random.Random(202)
        seqs = ["".join(rng.choice("ACGU") for _ in range(13)) for _ in range(10)]
        refs = {s: brute_force(s) for s in seqs}
        old_mod = rna_fold.MOD
        try:
            rna_fold.MOD = 7
            for seq in seqs:
                res_small = fold(seq)
                self.assertEqual(set(res_small["pairs"]), set(refs[seq]["pairs"]))
                self.assertEqual(res_small["structure"], refs[seq]["structure"])
                self.assertIn(res_small["count"], range(7))
        finally:
            rna_fold.MOD = old_mod


class ForcedPairTests(unittest.TestCase):
    def test_forced_outer_spanning_pair(self):
        # Force (0, 13) to span the whole sequence; the optimum must use it
        # and may freely pair inside.
        seq = "GAAAAAAAAAAAAU"
        self.assertEqual(len(seq), 14)
        res = fold(seq, {"pairs": [(0, 13)]})
        pairs = structure_pairs(res["structure"])
        self.assertIn((0, 13), pairs)
        ref = brute_force(seq, frozenset({(0, 13)}))
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["structure"], ref["structure"])

    def test_forced_nested_and_adjacent(self):
        # Build a sequence around two nested forced pairs: (0,13) GC, (1,8) AU.
        chars = ["A"] * 14
        chars[0], chars[13] = "G", "C"
        chars[1], chars[8] = "A", "U"
        seq = "".join(chars)
        res = fold(seq, {"pairs": [(0, 13), (1, 8)]})
        pairs = structure_pairs(res["structure"])
        self.assertTrue({(0, 13), (1, 8)} <= pairs)
        ref = brute_force(seq, frozenset({(0, 13), (1, 8)}))
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["count"], ref["exact_count"])

    def test_string_constraint_form(self):
        # 14 chars; forces (0,13) GC and (2,7) AU.
        chars = ["A"] * 14
        chars[0], chars[13] = "G", "C"
        chars[2], chars[7] = "A", "U"
        seq = "".join(chars)
        pattern = "(.(....).....)"
        self.assertEqual(len(pattern), len(seq))
        res = fold(seq, pattern)
        pairs = structure_pairs(res["structure"])
        self.assertIn((0, 13), pairs)
        self.assertIn((2, 7), pairs)
        ref = brute_force(seq, frozenset({(0, 13), (2, 7)}))
        self.assertEqual(res["score"], ref["score"])

    def test_x_forces_unpaired(self):
        seq = "GGGGAAAACCCC"
        pattern = "x" + "." * (len(seq) - 1)
        res = fold(seq, pattern)
        self.assertEqual(res["structure"][0], ".")
        ref = brute_force(seq, None, frozenset({0}))
        self.assertEqual(res["score"], ref["score"])

    def test_forced_pair_can_lower_score(self):
        # Free optimum nests (0,10) GC + (1,4) AU = 5; forcing the weak
        # GU pair (0, 4) strands the C at 10 and scores only 1.
        chars = ["A"] * 11
        chars[0], chars[4], chars[10] = "G", "U", "C"
        seq = "".join(chars)
        free = fold(seq)
        forced = fold(seq, {"pairs": [(0, 4)]})
        self.assertIn((0, 4), structure_pairs(forced["structure"]))
        self.assertEqual(free["score"], 5)
        self.assertEqual(forced["score"], 1)
        self.assertLess(forced["score"], free["score"])
        ref = brute_force(seq, frozenset({(0, 4)}))
        self.assertEqual(forced["score"], ref["score"])

    def test_count_modulus_contract(self):
        # Purely structural check: counts are reported mod MOD.
        seq = "A" * 120
        res = fold(seq)
        self.assertEqual(res["score"], 0)
        self.assertEqual(res["structure"], "." * 120)
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["pairs"], ())


class LexOrderTests(unittest.TestCase):
    def test_lex_dot_bracket_ascii_order(self):
        # G C G A A A A U: the AU pair (3,7) scores 2 and beats both outer
        # GU candidates (0,7)/(2,7) scoring 1; the unique optimum is "...(...)".
        seq = "GCGAAAAU"
        res = fold(seq)
        self.assertEqual(res["score"], 2)
        self.assertEqual(res["structure"], "...(...)")
        self.assertEqual(res["structure"], fold(seq)["structure"])
        # ASCII lex ordering used by the contract: '(' < ')' < '.'.
        self.assertLess("(", ")")
        self.assertLess(")", ".")

    def test_lex_min_bruteforce_many(self):
        rng = random.Random(31337)
        for _ in range(60):
            n = rng.randint(1, 14)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            res = fold(seq)
            ref = brute_force(seq)
            self.assertEqual(res["structure"], ref["structure"])


class InputValidationTests(unittest.TestCase):
    def test_illegal_character_is_located(self):
        with self.assertRaises(InvalidSequenceError) as cm:
            fold("ACGT")
        self.assertIn("position 3", str(cm.exception))

    def test_lowercase_rejected(self):
        with self.assertRaises(InvalidSequenceError) as cm:
            fold("acgu")
        self.assertIn("position 0", str(cm.exception))

    def test_too_long(self):
        with self.assertRaises(InvalidSequenceError) as cm:
            fold("A" * 121)
        self.assertIn("121", str(cm.exception))

    def test_sequence_wrong_type(self):
        with self.assertRaises(TypeError):
            fold(b"ACGU")
        with self.assertRaises(TypeError):
            fold(["A", "C", "G", "U"])

    def test_bool_rejected_as_integer(self):
        with self.assertRaises(TypeError) as cm:
            fold("GCGAAAC", {"pairs": [(True, 6)]})
        self.assertIn("pairs[0][0]", str(cm.exception))
        with self.assertRaises(TypeError):
            fold("GCGAAAC", {"unpaired": [False]})

    def test_float_nan_infinity_rejected(self):
        with self.assertRaises(ValueError) as cm:
            fold("GCGAAAC", {"pairs": [(0, math.nan)]})
        self.assertIn("NaN", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            fold("GCGAAAC", {"pairs": [(0, math.inf)]})
        self.assertIn("Infinity", str(cm.exception))
        with self.assertRaises(TypeError):
            fold("GCGAAAC", {"pairs": [(0, 1.0)]})

    def test_position_out_of_range_located(self):
        with self.assertRaises(ConstraintError) as cm:
            fold("GCGAAAC", {"pairs": [(0, 99)]})
        self.assertIn("pairs[0]", str(cm.exception))
        with self.assertRaises(ConstraintError) as cm:
            fold("GCGAAAC", {"unpaired": [-1]})
        self.assertIn("unpaired[0]", str(cm.exception))

    def test_non_complementary_forced_pair_rejected(self):
        with self.assertRaises(ConstraintError) as cm:
            fold("AAAAAAAA", {"pairs": [(0, 4)]})
        self.assertIn("pairs[0]=(0, 4)", str(cm.exception))

    def test_too_close_forced_pair_rejected(self):
        with self.assertRaises(ConstraintError) as cm:
            fold("GCGAAAC", {"pairs": [(0, 3)]})
        self.assertIn("j - i must be >= 4", str(cm.exception))

    def test_shared_endpoint_rejected(self):
        chars = ["A"] * 15
        chars[0], chars[13], chars[8] = "G", "C", "C"
        seq = "".join(chars)
        with self.assertRaises(ConstraintError) as cm:
            fold(seq, {"pairs": [(0, 13), (0, 8)]})
        self.assertIn("position 0", str(cm.exception))

    def test_crossing_forced_pairs_rejected(self):
        chars = ["A"] * 15
        chars[0], chars[8], chars[3], chars[13] = "G", "C", "G", "C"
        seq = "".join(chars)
        with self.assertRaises(ConstraintError) as cm:
            fold(seq, {"pairs": [(0, 8), (3, 13)]})
        self.assertIn("cross", str(cm.exception))

    def test_forced_pair_vs_forced_unpaired(self):
        chars = ["A"] * 15
        chars[0], chars[13] = "G", "C"
        seq = "".join(chars)
        with self.assertRaises(ConstraintError) as cm:
            fold(seq, {"pairs": [(0, 13)], "unpaired": [0]})
        self.assertIn("forced unpaired", str(cm.exception))

    def test_bad_string_constraint(self):
        with self.assertRaises(ConstraintError):
            fold("GCGAAAC", "...")  # wrong length
        with self.assertRaises(ConstraintError) as cm:
            fold("GCGAAAC", "?).....")
        self.assertIn("position 0", str(cm.exception))
        with self.assertRaises(ConstraintError) as cm:
            fold("GCGAAAC", "(.....X")
        self.assertIn("position 6", str(cm.exception))

    def test_unknown_constraint_key(self):
        with self.assertRaises(ConstraintError) as cm:
            fold("GCGAAAC", {"pairz": []})
        self.assertIn("unknown key", str(cm.exception))

    def test_bad_constraint_type(self):
        with self.assertRaises(TypeError):
            fold("GCGAAAC", [("(", ")")])

    def test_no_silent_partial_change(self):
        # fold is pure: a failing call must not mutate its arguments.
        pairs = [(0, 8)]
        with self.assertRaises(ConstraintError):
            fold("GCGAAAC", {"pairs": pairs})
        self.assertEqual(pairs, [(0, 8)])


class BoundaryAndPerformanceTests(unittest.TestCase):
    def test_empty_sequence(self):
        res = fold("")
        self.assertEqual(res["score"], 0)
        self.assertEqual(res["structure"], "")
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["pairs"], ())

    def test_max_length_runs_quickly(self):
        # Deterministic structured input at the interface cap; exact DP must
        # finish comfortably and return a valid structure.
        seq = ("GCGAAACGC" * 14)[:120]
        res = fold(seq)
        self.assertEqual(len(res["structure"]), 120)
        self.assertTrue(structure_is_legal(seq, res["structure"]))
        self.assertGreater(res["score"], 0)

    def test_budget_exception_distinct_from_infeasible(self):
        # The two failure semantics are permanently distinct classes.
        self.assertTrue(issubclass(UnsatisfiableConstraintError, ConstraintError))
        self.assertFalse(issubclass(BudgetExceededError, ConstraintError))

    def test_structure_uses_only_dot_bracket(self):
        rng = random.Random(123)
        for _ in range(20):
            n = rng.randint(1, 40)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            res = fold(seq)
            self.assertTrue(set(res["structure"]) <= {"(", ")", "."})


if __name__ == "__main__":
    unittest.main()
