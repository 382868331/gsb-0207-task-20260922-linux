"""Tests for per-position unpaired scores and the adjacent stack bonus.

All exact values are cross-checked against the independent brute-force
reference in ``tests/brute_reference.py`` (which never imports the module
under test).
"""

from __future__ import annotations

import random
import unittest

from rna_fold import (
    ConstraintError,
    fold,
    score_structure,
)
from tests.brute_reference import brute_force
from tests.test_fold import structure_pairs


def _score_by_pairs(seq, db, up, stack):
    pair_scores = {
        ("G", "C"): 3, ("C", "G"): 3,
        ("A", "U"): 2, ("U", "A"): 2,
        ("G", "U"): 1, ("U", "G"): 1,
    }
    total = sum(up[i] for i, ch in enumerate(db) if ch == ".")
    partner = {}
    for a, b in structure_pairs(db):
        total += pair_scores[(seq[a], seq[b])]
        partner[a] = b
        partner[b] = a
    for a, b in partner.items():
        if a < b and partner.get(b - 1) == a + 1:
            total += stack
    return total


class BruteForceScoringTests(unittest.TestCase):
    def test_unpaired_scores_exhaustive_small(self):
        rng = random.Random(11)
        checked = 0
        for n in range(0, 12):
            for _ in range(25 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                up = [rng.randint(-3, 3) for _ in range(n)]
                res = fold(seq, unpaired_scores=up)
                ref = brute_force(seq, unpaired_scores=up)
                with self.subTest(seq=seq, up=up):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                    self.assertEqual(
                        _score_by_pairs(seq, res["structure"], up, 0),
                        ref["score"],
                    )
                checked += 1
        self.assertGreater(checked, 200)

    def test_stack_scores_exhaustive_small(self):
        rng = random.Random(22)
        checked = 0
        for n in range(0, 13):
            for _ in range(25 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                stack = rng.choice([-5, -3, -1, 0, 1, 2, 4])
                res = fold(seq, stack_score=stack)
                ref = brute_force(seq, stack_score=stack)
                with self.subTest(seq=seq, stack=stack):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                checked += 1
        self.assertGreater(checked, 200)

    def test_combined_terms_exhaustive(self):
        rng = random.Random(33)
        for n in (8, 10, 12, 14):
            for _ in range(30):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                up = [rng.randint(-4, 4) for _ in range(n)]
                stack = rng.randint(-4, 4)
                res = fold(seq, unpaired_scores=up, stack_score=stack)
                ref = brute_force(seq, unpaired_scores=up, stack_score=stack)
                with self.subTest(seq=seq, up=up, stack=stack):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))

    def test_combined_terms_with_forced_constraints(self):
        rng = random.Random(44)
        pair_scores = {
            ("G", "C"), ("C", "G"), ("A", "U"),
            ("U", "A"), ("G", "U"), ("U", "G"),
        }
        for n in (8, 10, 12):
            for _ in range(30):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                legal = [
                    (a, b)
                    for a in range(n)
                    for b in range(a + 4, n)
                    if (seq[a], seq[b]) in pair_scores
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
                up_constraint = set(rng.sample(free, 1)) if free else set()
                up_scores = [rng.randint(-4, 4) for _ in range(n)]
                stack = rng.randint(-4, 4)
                constraints = {
                    "pairs": forced,
                    "unpaired": up_constraint,
                }
                res = fold(
                    seq, constraints,
                    unpaired_scores=up_scores, stack_score=stack,
                )
                ref = brute_force(
                    seq, frozenset(forced), frozenset(up_constraint),
                    unpaired_scores=up_scores, stack_score=stack,
                )
                with self.subTest(seq=seq, forced=forced):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))


class NegativeStackTests(unittest.TestCase):
    def test_negative_stack_changes_optimum(self):
        # GGAAACC (n=7): the only two-pair structure is the exact stack
        # (0,6) GC + (1,5) GC = 6; alternatives are lone GC pairs scoring 3.
        # A -4 stack turns the chain into 2, so a lone pair wins instead.
        seq = "GGAAACC"
        free = fold(seq)
        neg = fold(seq, stack_score=-4)
        self.assertEqual(free["score"], 6)
        self.assertEqual(neg["score"], 3)
        ref_neg = brute_force(seq, stack_score=-4)
        self.assertEqual(neg["score"], ref_neg["score"])
        # A negative stack must remove every adjacent stack from the optimum.
        pairs = structure_pairs(neg["structure"])
        for a, b in pairs:
            self.assertNotIn((a + 1, b - 1), pairs)

    def test_negative_stack_with_forced_stacked_pair_still_pays(self):
        chars = ["A"] * 9
        chars[0], chars[8] = "G", "C"
        chars[1], chars[7] = "A", "U"
        seq = "".join(chars)
        res = fold(
            seq,
            {"pairs": [(0, 8), (1, 7)]},
            stack_score=-10,
        )
        pairs = structure_pairs(res["structure"])
        self.assertIn((0, 8), pairs)
        self.assertIn((1, 7), pairs)
        ref = brute_force(
            seq, frozenset({(0, 8), (1, 7)}), stack_score=-10
        )
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["score"], 3 + 2 - 10)


class UnpairedPayoffTests(unittest.TestCase):
    def test_unpaired_bonus_beats_pairing(self):
        # GCGAAAC (n=7): (0,6) GC=3 and (2,6) GC=3 cannot coexist (share 6),
        # so the free optimum is one GC pair -> score 3.
        seq = "GCGAAAC"
        free = fold(seq)
        self.assertEqual(free["score"], 3)
        up = [10] + [0] * 6
        paid = fold(seq, unpaired_scores=up)
        self.assertEqual(paid["structure"][0], ".")
        self.assertTrue(all(a != 0 and b != 0 for a, b in paid["pairs"]))
        ref = brute_force(seq, unpaired_scores=up)
        self.assertEqual(paid["score"], ref["score"])
        self.assertEqual(paid["score"], 10 + 3)

    def test_negative_unpaired_cost_prefers_pairing(self):
        seq = "GCGCGCGCGC"  # n=10
        up = [-2] * 10
        res = fold(seq, unpaired_scores=up)
        ref = brute_force(seq, unpaired_scores=up)
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["structure"], ref["structure"])
        dots = res["structure"].count(".")
        pair_count = len(structure_pairs(res["structure"]))
        self.assertEqual(dots, 10 - 2 * pair_count)


class DecompositionCountTests(unittest.TestCase):
    def test_structure_counted_once_despite_multiple_decompositions(self):
        seq = "GCGCGCGCGC"  # n=10
        for stack in (-3, 0, 3):
            with self.subTest(stack=stack):
                res = fold(seq, stack_score=stack)
                ref = brute_force(seq, stack_score=stack)
                self.assertEqual(res["count"], ref["exact_count"])
        res = fold("A" * 10, unpaired_scores=[1] * 10)
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["score"], 10)

    def test_counts_match_exact_enumeration_with_terms(self):
        rng = random.Random(55)
        for _ in range(120):
            n = rng.randint(1, 14)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            up = [rng.randint(-2, 2) for _ in range(n)]
            stack = rng.choice([-3, -1, 0, 1, 3])
            res = fold(seq, unpaired_scores=up, stack_score=stack)
            ref = brute_force(seq, unpaired_scores=up, stack_score=stack)
            self.assertEqual(res["count"], ref["exact_count"])
            self.assertEqual(res["score"], ref["score"])
            self.assertEqual(set(res["pairs"]), set(ref["pairs"]))


class StackChangesOptimumDemoTests(unittest.TestCase):
    def test_stack_reward_changes_structure(self):
        # GGAACUU (n=7): the lone GC pair (0,4) scores 3 ("(...).."); the
        # directly nested two-GU stack (0,6)+(1,5) scores only 1+1=2 by
        # itself but exactly one stack bonus makes 2+3=5, flipping the
        # optimum to "((...))".
        seq = "GGAACUU"
        no_stack = fold(seq, stack_score=0)
        bonus = fold(seq, stack_score=3)
        self.assertEqual(no_stack["structure"], "(...)..")
        self.assertEqual(no_stack["score"], 3)
        self.assertEqual(bonus["structure"], "((...))")
        self.assertEqual(bonus["score"], 1 + 1 + 3)
        ref = brute_force(seq, stack_score=3)
        self.assertEqual(bonus["score"], ref["score"])
        self.assertEqual(bonus["structure"], ref["structure"])

        # Negative stack makes the outer pair unattractive again.
        neg = fold(seq, stack_score=-2)
        ref_neg = brute_force(seq, stack_score=-2)
        self.assertEqual(neg["score"], ref_neg["score"])
        self.assertEqual(neg["structure"], "(...)..")


class ScoreStructureHelperTests(unittest.TestCase):
    def test_score_structure_matches_fold(self):
        rng = random.Random(66)
        for _ in range(50):
            n = rng.randint(1, 14)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            up = [rng.randint(-3, 3) for _ in range(n)]
            stack = rng.randint(-3, 3)
            res = fold(seq, unpaired_scores=up, stack_score=stack)
            self.assertEqual(
                score_structure(
                    seq, res["structure"],
                    unpaired_scores=up, stack_score=stack,
                ),
                res["score"],
            )

    def test_score_structure_stacks_counted_once(self):
        seq = "GCGCGCGC"  # n=8: ((....)) -> two pairs, one stack
        self.assertEqual(
            score_structure(seq, "((....))", stack_score=5),
            3 + 3 + 5,
        )
        self.assertEqual(score_structure(seq, "((....))"), 6)

    def test_score_structure_unpaired_scores(self):
        seq = "GCGAAAC"
        self.assertEqual(
            score_structure(seq, "(.....)", unpaired_scores=[0, 2, 0, 1, 1, 1, 0]),
            3 + 2 + 1 + 1 + 1,
        )

    def test_score_structure_rejects_bad_input(self):
        with self.assertRaises(ConstraintError):
            score_structure("GCGAAAC", "((...))")  # distance < 4
        with self.assertRaises(ConstraintError):
            score_structure("AAAAAAA", "(.....)")  # AA not a pair
        with self.assertRaises(ConstraintError):
            score_structure("GCGAAAC", "(....")  # unbalanced

    def test_scoring_parameter_validation(self):
        with self.assertRaises(ConstraintError):
            fold("ACGUACGU", unpaired_scores=[1, 2, 3])  # wrong length
        with self.assertRaises(TypeError):
            fold("ACGUACGU", unpaired_scores="00000000")
        with self.assertRaises(TypeError):
            fold("ACGUACGU", stack_score=True)
        with self.assertRaises(TypeError):
            fold("ACGUACGU", unpaired_scores=[0] * 7 + [1.5])

    def test_defaults_are_zero(self):
        seq = "GCGAAACGC"
        self.assertEqual(fold(seq)["score"], fold(seq, unpaired_scores=[0] * 9, stack_score=0)["score"])


if __name__ == "__main__":
    unittest.main()
