"""Tests for the optional single-H-type pseudoknot mode.

The production DP is cross-checked against ``brute_pseudoknot.py``, which
enumerates matchings from scratch and independently recognises the H
topology.
"""

from __future__ import annotations

import random
import unittest

from rna_fold import (
    MAX_PSEUDOKNOT_LENGTH,
    ConstraintError,
    InvalidSequenceError,
    UnsatisfiableConstraintError,
    fold,
)
from tests.brute_pseudoknot import brute_force_pk

PAIR_SCORES = {
    ("G", "C"): 3,
    ("C", "G"): 3,
    ("A", "U"): 2,
    ("U", "A"): 2,
    ("G", "U"): 1,
    ("U", "G"): 1,
}


def parse_extended(db):
    stacks = {"(": [], "[": []}
    close = {")": "(", "]": "["}
    pairs = set()
    stem1, stem2 = set(), set()
    for i, ch in enumerate(db):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in close:
            j = stacks[close[ch]].pop()
            pairs.add((j, i))
            (stem1 if ch == ")" else stem2).add((j, i))
    return pairs, stem1, stem2


class PseudoknotComparisonTests(unittest.TestCase):
    def test_unconstrained_vs_bruteforce(self):
        rng = random.Random(20260926)
        checked = 0
        for n in range(0, 13):
            for _ in range(20 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                for penalty in (0, 2):
                    res = fold(
                        seq,
                        allow_pseudoknot=True,
                        pseudoknot_penalty=penalty,
                    )
                    ref = brute_force_pk(seq, penalty=penalty)
                    with self.subTest(seq=seq, penalty=penalty):
                        self.assertEqual(res["score"], ref["score"])
                        self.assertEqual(res["structure"], ref["structure"])
                        self.assertEqual(res["count"], ref["exact_count"])
                        self.assertEqual(
                            set(res["pairs"]), set(ref["pairs"])
                        )
                        checked += 1
        self.assertGreater(checked, 200)

    def test_constrained_vs_bruteforce(self):
        rng = random.Random(77)
        for n in (10, 12, 14):
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
                    forced.append((a, b))
                    used.update((a, b))
                    if len(forced) == 2:
                        break
                penalty = rng.choice((0, 1, 5))
                res = fold(
                    seq,
                    {"pairs": forced},
                    allow_pseudoknot=True,
                    pseudoknot_penalty=penalty,
                )
                ref = brute_force_pk(
                    seq, frozenset(forced), penalty=penalty
                )
                if not ref["feasible"]:
                    continue
                with self.subTest(seq=seq, forced=forced, penalty=penalty):
                    self.assertEqual(res["score"], ref["score"])
                    self.assertEqual(res["structure"], ref["structure"])
                    self.assertEqual(res["count"], ref["exact_count"])
                    got, _, _ = parse_extended(res["structure"])
                    self.assertTrue(set(forced) <= got)


class PseudoknotTopologyTests(unittest.TestCase):
    def _pk_sequence(self):
        # stem1 (0,7),(1,6); stem2 (3,10),(4,9), n=12
        chars = ["A"] * 12
        assign = {
            0: "G", 1: "G", 7: "C", 6: "C",
            3: "G", 4: "G", 10: "C", 9: "C",
        }
        for x, ch in assign.items():
            chars[x] = ch
        return "".join(chars)

    def test_pseudoknot_found_and_marked(self):
        seq = self._pk_sequence()
        res = fold(seq, allow_pseudoknot=True)
        self.assertEqual(
            res["structure"], "((.[[.)).]]."
        )
        pairs, stem1, stem2 = parse_extended(res["structure"])
        self.assertEqual(
            stem1, {(0, 7), (1, 6)}
        )
        self.assertEqual(stem2, {(3, 10), (4, 9)})
        info = res["pseudoknot"]
        self.assertEqual(
            tuple(info["stem1"]), ((0, 7), (1, 6))
        )
        self.assertEqual(tuple(info["stem2"]), ((3, 10), (4, 9)))
        # score: four GC pairs = 12
        self.assertEqual(res["score"], 12)
        # endpoints used once
        positions = [p for pr in pairs for p in pr]
        self.assertEqual(len(positions), len(set(positions)))

    def test_penalty_reverts_to_pseudoknot_free(self):
        seq = self._pk_sequence()
        free = fold(seq)
        penalised = fold(
            seq, allow_pseudoknot=True, pseudoknot_penalty=5
        )
        self.assertIsNone(penalised["pseudoknot"])
        self.assertEqual(penalised["score"], free["score"])
        self.assertEqual(penalised["structure"], free["structure"])
        # small penalty still prefers the crossing (12 - 1 > 9)
        light = fold(seq, allow_pseudoknot=True, pseudoknot_penalty=1)
        self.assertIsNotNone(light["pseudoknot"])
        self.assertEqual(light["score"], 11)

    def test_no_pseudoknot_without_switch(self):
        seq = self._pk_sequence()
        res = fold(seq)
        self.assertIsNone(res["pseudoknot"])
        self.assertNotIn("[", res["structure"])

    def test_switch_off_does_not_force_crossing(self):
        # A sequence whose only crossing option is weak never crosses even
        # with the switch on; the pk-free optimum stays the answer.
        seq = "GGGGAAAACCCCAAA"[:14]
        res = fold(seq, allow_pseudoknot=True)
        ref = fold(seq)
        self.assertEqual(res["score"], ref["score"])
        self.assertIsNone(res["pseudoknot"])

    def test_forced_crossing_pair_admitted(self):
        seq = self._pk_sequence()
        res = fold(
            seq, {"pairs": [(0, 7), (3, 10)]}, allow_pseudoknot=True
        )
        pairs, _, _ = parse_extended(res["structure"])
        self.assertIn((0, 7), pairs)
        self.assertIn((3, 10), pairs)
        self.assertIsNotNone(res["pseudoknot"])

    def test_forced_second_pseudoknot_is_unsatisfiable(self):
        seq = self._pk_sequence()
        # (0,7)x(3,10) is the one H crossing; forcing an unrelated pair
        # that demands a second crossing network is infeasible.
        with self.assertRaises(UnsatisfiableConstraintError):
            fold(
                seq,
                {"pairs": [(0, 7), (3, 10), (1, 9)]},
                allow_pseudoknot=True,
            )

    def test_crossing_forbidden_without_switch(self):
        seq = self._pk_sequence()
        with self.assertRaises(ConstraintError):
            fold(seq, {"pairs": [(0, 7), (3, 10)]})

    def test_string_constraints_with_crossing_brackets(self):
        seq = self._pk_sequence()
        pattern = ["."] * 12
        for x, y in ((0, 7),):
            pattern[x], pattern[y] = "(", ")"
        for x, y in ((3, 10),):
            pattern[x], pattern[y] = "[", "]"
        res = fold(
            seq, "".join(pattern), allow_pseudoknot=True
        )
        pairs, _, _ = parse_extended(res["structure"])
        self.assertIn((0, 7), pairs)
        self.assertIn((3, 10), pairs)


    def test_forced_unpaired_with_pseudoknot(self):
        # Locking position 0 pushes the pseudoknot inside an enclosing
        # pair; the two crossing stems remain exactly one H core.
        seq = self._pk_sequence()
        res = fold(seq, {"unpaired": [0]}, allow_pseudoknot=True)
        ref = brute_force_pk(seq, forced_unpaired=frozenset({0}))
        self.assertEqual(res["score"], ref["score"])
        self.assertEqual(res["structure"], ref["structure"])
        self.assertEqual(res["count"], ref["exact_count"])
        self.assertEqual(set(res["pairs"]), set(ref["pairs"]))
        s1 = res["pseudoknot"]["stem1"]
        s2 = res["pseudoknot"]["stem2"]
        self.assertEqual(len(s1), len(s2))
        (i, j), (k, l) = s1[0], s2[0]
        self.assertTrue(i < k < j < l)

    def test_tie_between_pk_and_free_merges_count(self):
        # Tune the penalty so pk and pk-free scores coincide: both classes
        # contribute to count and pairs, lex decides the witness.
        seq = self._pk_sequence()
        free = fold(seq)
        pk0 = fold(seq, allow_pseudoknot=True)
        gap = pk0["score"] - free["score"]
        tied = fold(
            seq, allow_pseudoknot=True, pseudoknot_penalty=gap
        )
        ref = brute_force_pk(seq, penalty=gap)
        self.assertEqual(tied["score"], free["score"])
        self.assertEqual(tied["count"], ref["exact_count"])
        self.assertEqual(tied["structure"], ref["structure"])
        self.assertEqual(set(tied["pairs"]), set(ref["pairs"]))


class PseudoknotValidationTests(unittest.TestCase):
    def test_length_cap_is_24(self):
        fold("A" * 24, allow_pseudoknot=True)
        with self.assertRaises(InvalidSequenceError) as cm:
            fold("A" * 25, allow_pseudoknot=True)
        self.assertIn("25", str(cm.exception))
        # default mode still allows up to 120
        fold("A" * 25)

    def test_default_max_constant(self):
        from rna_fold import MAX_SEQUENCE_LENGTH

        self.assertEqual(MAX_SEQUENCE_LENGTH, 120)
        self.assertEqual(MAX_PSEUDOKNOT_LENGTH, 24)

    def test_bad_penalty(self):
        with self.assertRaises(ValueError):
            fold("ACGUACGUAC", allow_pseudoknot=True, pseudoknot_penalty=-1)
        with self.assertRaises(TypeError):
            fold("ACGUACGUAC", allow_pseudoknot=True, pseudoknot_penalty=1.5)
        with self.assertRaises(TypeError):
            fold("ACGUACGUAC", allow_pseudoknot="yes")

    def test_result_is_plain_dict_with_pseudoknot_key(self):
        res = fold("ACGUACGUAC")
        self.assertIsNone(res["pseudoknot"])
        self.assertEqual(set(res), {
            "score", "structure", "count", "pairs", "pseudoknot"
        })


if __name__ == "__main__":
    unittest.main()
