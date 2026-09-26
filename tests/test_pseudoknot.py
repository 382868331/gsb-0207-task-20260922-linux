"""Tests for the optional single H-type pseudoknot mode of rna_fold.fold.

Cross-checked against tests/brute_pk_reference.py, an independent
enumeration that never imports the module under test.
"""

from __future__ import annotations

import random
import unittest

from rna_fold import (
    MAX_PSEUDOKNOT_SEQUENCE_LENGTH,
    ConstraintError,
    InvalidSequenceError,
    fold,
)
from tests.brute_pk_reference import (
    PAIR_SCORES,
    brute_force_pk,
    topology_has_single_h_pseudoknot,
)


def structure_pairs(db: str):
    """Parse a dot-bracket string with () and [] into a set of pairs."""
    stacks = {"(": [], "[": []}
    close = {")": "(", "]": "["}
    pairs = set()
    for i, ch in enumerate(db):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in close:
            pairs.add((stacks[close[ch]].pop(), i))
    return pairs


def structure_is_admissible(seq, db):
    """Legal pairs, each position used once, topology non-crossing or 1 H-pk."""
    n = len(seq)
    if len(db) != n:
        return False
    pairs = structure_pairs(db)
    used = set()
    for a, b in pairs:
        if b - a < 4 or (seq[a], seq[b]) not in PAIR_SCORES:
            return False
        if a in used or b in used:
            return False
        used.add(a)
        used.add(b)
    return topology_has_single_h_pseudoknot(frozenset(pairs)) is not None


class PseudoknotComparisonTests(unittest.TestCase):
    """fold(pseudoknot_penalty=...) vs independent enumeration, n <= 12."""

    def test_random_short_sequences(self):
        rng = random.Random(20260926)
        checked = 0
        for n in range(0, 13):
            for _ in range(12 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                penalty = rng.choice([0, 1, 2, 3])
                res = fold(seq, pseudoknot_penalty=penalty)
                ref = brute_force_pk(seq, penalty)
                with self.subTest(seq=seq, penalty=penalty):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                    self.assertTrue(
                        structure_is_admissible(seq, res["structure"])
                    )
                    # 'pseudoknot' marks the stems of the reported structure.
                    if "[" in res["structure"]:
                        self.assertIsNotNone(res["pseudoknot"])
                        stems = (
                            set(res["pseudoknot"]["stem1"])
                            | set(res["pseudoknot"]["stem2"])
                        )
                        self.assertTrue(
                            stems <= structure_pairs(res["structure"])
                        )
                    else:
                        self.assertIsNone(res["pseudoknot"])
                checked += 1
        self.assertGreater(checked, 100)

    def test_constrained_vs_reference(self):
        rng = random.Random(31517)
        for n in (10, 12):
            for _ in range(15):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
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
                    break
                free = [p for p in range(n) if p not in used]
                up = set(rng.sample(free, min(1, len(free)))) if free else set()
                penalty = rng.choice([0, 2])
                res = fold(
                    seq,
                    {"pairs": forced, "unpaired": up},
                    pseudoknot_penalty=penalty,
                )
                ref = brute_force_pk(
                    seq, penalty, frozenset(forced), frozenset(up)
                )
                with self.subTest(seq=seq, forced=forced, up=up):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                    got = structure_pairs(res["structure"])
                    self.assertTrue(set(forced) <= got)
                    self.assertTrue(
                        up.isdisjoint(p for pr in got for p in pr)
                    )


class PseudoknotSemanticsTests(unittest.TestCase):
    # Two GC stems whose only high-scoring arrangement crosses (i<k<j<l).
    PK_SEQ = "GGGAAGGGACCCAACCC"

    def test_h_pseudoknot_found_and_marked(self):
        res = fold(self.PK_SEQ, pseudoknot_penalty=0)
        base = fold(self.PK_SEQ)
        self.assertGreater(res["score"], base["score"])
        pk = res["pseudoknot"]
        self.assertIsNotNone(pk)
        s1, s2 = pk["stem1"], pk["stem2"]
        # consecutive nested stems
        for stem in (s1, s2):
            for x in range(len(stem) - 1):
                self.assertEqual(stem[x + 1][0], stem[x][0] + 1)
                self.assertEqual(stem[x + 1][1], stem[x][1] - 1)
        # interleaved endpoints i < k < j < l
        self.assertLess(s1[0][0], s2[0][0])
        self.assertLess(s2[0][0], s1[0][1])
        self.assertLess(s1[0][1], s2[0][1])
        self.assertEqual(pk["penalty"], 0)
        # structure uses [] for the second stem
        self.assertIn("[", res["structure"])
        self.assertEqual(
            structure_pairs(res["structure"]),
            set(res["pairs"]),
        )

    def test_penalty_falls_back_to_pseudoknot_free(self):
        crossed = fold(self.PK_SEQ, pseudoknot_penalty=0)
        base = fold(self.PK_SEQ)
        huge = fold(self.PK_SEQ, pseudoknot_penalty=10**6)
        # A large penalty makes the pseudoknot structure lose: the optimum
        # falls back to the plain non-crossing fold.
        self.assertEqual(huge["score"], base["score"])
        self.assertEqual(huge["structure"], base["structure"])
        self.assertIsNone(huge["pseudoknot"])
        self.assertGreater(crossed["score"], huge["score"])

    def test_no_pseudoknot_stays_candidate_at_zero_penalty(self):
        # A sequence with no crossing opportunity must fold exactly as
        # before even with the switch on.
        seq = "GGAAAACC"
        res = fold(seq, pseudoknot_penalty=0)
        base = fold(seq)
        self.assertEqual(res["score"], base["score"])
        self.assertIsNone(res["pseudoknot"])
        self.assertNotIn("[", res["structure"])

    def test_no_illegal_second_pseudoknot(self):
        # Crafted sequences with two crossing-pair sites where an
        # unconstrained matcher could be tempted into two pseudoknots.
        # The independent reference admits at most one pseudoknot by
        # construction, so full agreement proves none was taken.
        crafty = [
            "GCAAACGGCAAACG",  # sites (0,5)x(1,6) and (7,12)x(8,13)
            "GGAUUAACCAAA",  # GC stem + UA stem crossing site
            "GGAGGAACCACC",  # two GC stems whose only union crosses
        ]
        for seq in crafty:
            for penalty in (0, 2):
                res = fold(seq, pseudoknot_penalty=penalty)
                ref = brute_force_pk(seq, penalty)
                with self.subTest(seq=seq, penalty=penalty):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
                    pairs = structure_pairs(res["structure"])
                    self.assertIsNotNone(
                        topology_has_single_h_pseudoknot(frozenset(pairs))
                    )

    def test_forced_crossing_pairs_still_rejected(self):
        # Forced pairs may not cross each other, even in pseudoknot mode.
        with self.assertRaises(ConstraintError):
            fold(
                self.PK_SEQ,
                {"pairs": [(0, 9), (4, 13)]},
                pseudoknot_penalty=0,
            )

    def test_forced_pair_inside_pseudoknot(self):
        # Forcing one stem pair of the natural pseudoknot keeps it optimal.
        res = fold(
            self.PK_SEQ,
            {"pairs": [(0, 11)]},
            pseudoknot_penalty=0,
        )
        self.assertIn((0, 11), structure_pairs(res["structure"]))
        self.assertIsNotNone(res["pseudoknot"])

    def test_forced_unpaired_blocks_stem(self):
        # Forcing position 1 unpaired kills the (1,10) stem pair, so the
        # length-3 stem1 is impossible; the reference agrees on the optimum.
        res = fold(
            self.PK_SEQ, {"unpaired": [1]}, pseudoknot_penalty=0
        )
        ref = brute_force_pk(self.PK_SEQ, 0, None, frozenset({1}))
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["count"], ref["exact_count"])

    def test_length_limit(self):
        seq = "A" * (MAX_PSEUDOKNOT_SEQUENCE_LENGTH + 1)
        with self.assertRaises(InvalidSequenceError):
            fold(seq, pseudoknot_penalty=0)
        # Without the switch the classic limit (120) still applies.
        fold("A" * (MAX_PSEUDOKNOT_SEQUENCE_LENGTH + 1))

    def test_penalty_validation(self):
        with self.assertRaises(TypeError):
            fold("ACGU", pseudoknot_penalty=True)
        with self.assertRaises(ValueError):
            fold("ACGU", pseudoknot_penalty=float("nan"))
        with self.assertRaises(TypeError):
            fold("ACGU", pseudoknot_penalty=1.5)

    def test_old_interface_unchanged(self):
        # The classic call keeps its exact 4-key result shape.
        res = fold("GGCGAUUAGCCGAUU")
        self.assertEqual(
            set(res), {"score", "structure", "count", "pairs"}
        )


if __name__ == "__main__":
    unittest.main()
