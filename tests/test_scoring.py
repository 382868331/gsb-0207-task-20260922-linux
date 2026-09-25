"""Tests for unpaired-position scores and the adjacent stacking bonus.

All expected values come from the independent exhaustive reference in
tests/brute_reference.py (which never imports the module under test) or
are computed by hand from first principles.
"""

from __future__ import annotations

import math
import random
import unittest

from rna_fold import fold
from tests.brute_reference import brute_force
from tests.test_fold import structure_is_legal, structure_pairs


def manual_score(seq, db, unpaired_scores=None, stack_score=0):
    """Score a dot-bracket structure directly from its definition."""
    pairs = structure_pairs(db)
    pair_scores = {
        ("G", "C"): 3, ("C", "G"): 3,
        ("A", "U"): 2, ("U", "A"): 2,
        ("G", "U"): 1, ("U", "G"): 1,
    }
    ups = unpaired_scores or [0] * len(seq)
    paired = {p for pr in pairs for p in pr}
    total = sum(pair_scores[(seq[a], seq[b])] for a, b in pairs)
    total += sum(ups[p] for p in range(len(seq)) if p not in paired)
    total += stack_score * sum(
        1 for a, b in pairs if (a + 1, b - 1) in pairs
    )
    return total


class RandomCrossCheckTests(unittest.TestCase):
    """DP vs exhaustive enumeration with the new scoring terms active."""

    def test_random_unpaired_and_stack_scores(self):
        rng = random.Random(20260925)
        checked = 0
        for n in range(0, 13):
            for _ in range(12 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                ups = [rng.randint(-3, 4) for _ in range(n)]
                stack = rng.randint(-4, 5)
                res = fold(seq, unpaired_scores=ups, stack_score=stack)
                ref = brute_force(
                    seq, unpaired_scores=ups, stack_score=stack
                )
                with self.subTest(seq=seq, ups=ups, stack=stack):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                    self.assertTrue(
                        structure_is_legal(seq, res["structure"])
                    )
                    # The returned witness achieves the reported score.
                    self.assertEqual(
                        res["score"],
                        manual_score(seq, res["structure"], ups, stack),
                    )
                    checked += 1
        self.assertGreater(checked, 130)

    def test_random_scores_with_constraints(self):
        pair_scores = {
            ("G", "C"), ("C", "G"), ("A", "U"),
            ("U", "A"), ("G", "U"), ("U", "G"),
        }
        rng = random.Random(777)
        for n in (8, 10, 12, 14):
            for _ in range(15):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                ups = [rng.randint(-2, 3) for _ in range(n)]
                stack = rng.randint(-3, 4)
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
                res = fold(
                    seq,
                    {"pairs": forced},
                    unpaired_scores=ups,
                    stack_score=stack,
                )
                ref = brute_force(
                    seq,
                    frozenset(forced),
                    unpaired_scores=ups,
                    stack_score=stack,
                )
                with self.subTest(seq=seq, forced=forced, stack=stack):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))

    def test_mapping_form_matches_sequence_form(self):
        rng = random.Random(55)
        for _ in range(20):
            n = rng.randint(1, 14)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            ups = [rng.randint(-2, 3) for _ in range(n)]
            sparse = {p: s for p, s in enumerate(ups) if s != 0}
            stack = rng.randint(-2, 3)
            as_list = fold(seq, unpaired_scores=ups, stack_score=stack)
            as_map = fold(seq, unpaired_scores=sparse, stack_score=stack)
            self.assertEqual(as_list, as_map)


class TargetedScoringTests(unittest.TestCase):
    def test_negative_stack_penalty(self):
        # The stacked structure ((.((...)))) wins with a bonus; a penalty
        # must make the DP prefer something else.  Verified against
        # exhaustive enumeration.
        seq = "GUGAUCCUAUGC"
        bonus = fold(seq, stack_score=3)
        penal = fold(seq, stack_score=-3)
        ref = brute_force(seq, stack_score=-3)
        self.assertEqual(penal["score"], ref["score"])
        self.assertEqual(penal["structure"], ref["structure"])
        self.assertEqual(penal["count"], ref["exact_count"])
        self.assertNotEqual(bonus["structure"], penal["structure"])
        # manual check: the penalised optimum contains no adjacent stack
        penal_pairs = structure_pairs(penal["structure"])
        for a, b in penal_pairs:
            self.assertNotIn((a + 1, b - 1), penal_pairs)

    def test_unpaired_score_outweighs_pairing(self):
        # GC pair is worth 3; paying 5 for each unpaired position makes the
        # all-unpaired structure the unique optimum.
        seq = "GCGAAACGC"
        ups = [5] * len(seq)
        res = fold(seq, unpaired_scores=ups)
        self.assertEqual(res["structure"], "." * len(seq))
        self.assertEqual(res["score"], 5 * len(seq))
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["pairs"], ())
        ref = brute_force(seq, unpaired_scores=ups)
        self.assertEqual(res["score"], ref["score"])

    def test_negative_unpaired_score_forces_pairing(self):
        # Penalising unpaired positions makes a weak GU pair worthwhile.
        seq = "GAAAU"
        # With +1 per unpaired position the weak GU pair (score 1) loses:
        # "(...)" scores 1 + 3 = 4, but "....." scores 5.
        paid = fold(seq, unpaired_scores=[1] * len(seq))
        self.assertEqual(paid["structure"], ".....")
        self.assertEqual(paid["score"], 5)
        # With -2 per unpaired position the same pair becomes worthwhile.
        res = fold(seq, unpaired_scores=[-2] * len(seq))
        self.assertEqual(res["structure"], "(...)")
        self.assertEqual(res["score"], 1 - 2 * 3)  # GU=1, 3 unpaired left
        ref = brute_force(seq, unpaired_scores=[-2] * len(seq))
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["count"], ref["exact_count"])

    def test_forced_pair_with_new_scores(self):
        # A forced pair interacts with unpaired scores and the stack bonus.
        seq = "GUGAUCCUAUGC"
        res = fold(
            seq,
            {"pairs": [(0, 11)]},
            unpaired_scores=[1] * len(seq),
            stack_score=2,
        )
        ref = brute_force(
            seq,
            frozenset({(0, 11)}),
            unpaired_scores=[1] * len(seq),
            stack_score=2,
        )
        self.assertIn((0, 11), structure_pairs(res["structure"]))
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["structure"], ref["structure"])
        self.assertEqual(res["count"], ref["exact_count"])

    def test_structure_counted_once_across_decompositions(self):
        # Every structure must contribute exactly 1 to the count, however
        # many DP decomposition paths lead to equal scores.  The exhaustive
        # reference counts each enumerated structure once; exact agreement
        # on tied optima proves the DP does the same.
        rng = random.Random(31415)
        for _ in range(40):
            n = rng.randint(4, 13)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            ups = [rng.randint(-1, 1) for _ in range(n)]
            stack = rng.randint(-1, 2)
            res = fold(seq, unpaired_scores=ups, stack_score=stack)
            ref = brute_force(seq, unpaired_scores=ups, stack_score=stack)
            with self.subTest(seq=seq):
                self.assertEqual(res["count"], ref["exact_count"])

    def test_stack_bonus_changes_optimum(self):
        # Demonstration case: the same sequence folds differently once the
        # adjacent stacking bonus is enabled.
        seq = "GUGAUCCUAUGC"
        base = fold(seq)
        boosted = fold(seq, stack_score=3)
        self.assertEqual(base["structure"], "(((...).)..)")
        self.assertEqual(base["score"], 8)
        self.assertEqual(boosted["structure"], "((.((...))))")
        self.assertEqual(boosted["score"], 14)  # 8 pair score + 2 stacks
        # both optima verified independently
        self.assertEqual(base["score"], brute_force(seq)["score"])
        self.assertEqual(
            boosted["score"], brute_force(seq, stack_score=3)["score"]
        )

    def test_stack_applies_only_to_directly_adjacent_nesting(self):
        # Directly nested: (0,7) GC + (1,6) GC -> 6 + bonus.
        seq2 = "GGAAAACC"
        res = fold(seq2, stack_score=4)
        self.assertEqual(res["structure"], "((....))")
        self.assertEqual(res["score"], 6 + 4)
        # Non-adjacent nesting ((.(....))) gets no bonus: pairs (0,9),(2,8).
        seq3 = "GCAGAAAACG"
        res3 = fold(seq3, stack_score=4)
        pairs3 = structure_pairs(res3["structure"])
        if (0, 9) in pairs3 and (2, 8) in pairs3:
            expected = manual_score(seq3, res3["structure"], stack_score=4)
            self.assertEqual(res3["score"], expected)
        self.assertEqual(
            res3["score"], brute_force(seq3, stack_score=4)["score"]
        )


class ScoringValidationTests(unittest.TestCase):
    def test_unpaired_scores_wrong_length(self):
        with self.assertRaises(ValueError) as cm:
            fold("GCGAAAC", unpaired_scores=[1, 2])
        self.assertIn("7", str(cm.exception))

    def test_unpaired_scores_position_out_of_range(self):
        with self.assertRaises(ValueError) as cm:
            fold("GCGAAAC", unpaired_scores={7: 1})
        self.assertIn("out of range", str(cm.exception))

    def test_unpaired_scores_rejects_bool_nan_inf(self):
        with self.assertRaises(TypeError):
            fold("GCGAAAC", unpaired_scores=[True] * 7)
        with self.assertRaises(ValueError) as cm:
            fold("GCGAAAC", unpaired_scores={0: math.nan})
        self.assertIn("NaN", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            fold("GCGAAAC", unpaired_scores={0: math.inf})
        self.assertIn("Infinity", str(cm.exception))
        with self.assertRaises(TypeError):
            fold("GCGAAAC", unpaired_scores=[1.5] * 7)

    def test_unpaired_scores_bad_type(self):
        with self.assertRaises(TypeError):
            fold("GCGAAAC", unpaired_scores="1111111")
        with self.assertRaises(TypeError):
            fold("GCGAAAC", unpaired_scores=5)

    def test_stack_score_validation(self):
        with self.assertRaises(TypeError):
            fold("GCGAAAC", stack_score=True)
        with self.assertRaises(ValueError) as cm:
            fold("GCGAAAC", stack_score=math.nan)
        self.assertIn("NaN", str(cm.exception))
        with self.assertRaises(TypeError):
            fold("GCGAAAC", stack_score=1.5)

    def test_defaults_match_legacy_behaviour(self):
        # With the new arguments left at their defaults the result is
        # identical to a call without them (all new scores are zero).
        rng = random.Random(8)
        for _ in range(20):
            n = rng.randint(0, 14)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            self.assertEqual(
                fold(seq), fold(seq, unpaired_scores=None, stack_score=0)
            )
            self.assertEqual(
                fold(seq), fold(seq, unpaired_scores=[0] * n, stack_score=0)
            )


if __name__ == "__main__":
    unittest.main()
