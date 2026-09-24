"""Tests for partition_fold (Boltzmann ensemble).

Expected values come from three independent sources, never from the
implementation under test:

* hand-computed two-/three-structure examples with closed-form exp weights;
* analytic identities (zero energies => every structure equally likely,
  Z equals the enumerated structure count; forced pair => probability 1;
  free_energy = -T ln Z);
* ``brute_partition`` in brute_reference.py, which directly enumerates
  every non-crossing matching and sums weights with plain arithmetic.
"""

from __future__ import annotations

import math
import random
import unittest

from rna_fold import (
    DEFAULT_PAIR_ENERGIES,
    MAX_PARTITION_SEQUENCE_LENGTH,
    ConstraintError,
    InvalidParameterError,
    InvalidSequenceError,
    fold,
    partition_fold,
)
from tests.brute_reference import brute_partition

PROB_TOL = 1e-9
LOG_TOL = 1e-9


def marginals_normalised(res, n):
    up = res["unpaired_probabilities"]
    pp = res["pair_probabilities"]
    for k in range(n):
        total = up[k] + sum(
            p for (a, b), p in pp.items() if a == k or b == k
        )
        if abs(total - 1.0) > 1e-10:
            return False, (k, total)
    return True, None


class HandComputedTests(unittest.TestCase):
    """Closed-form examples worked out by hand."""

    def test_no_pairs_possible(self):
        # AAAA admits exactly one structure (all unpaired), E = 0.
        res = partition_fold("AAAA")
        self.assertEqual(res["partition_function"], 1.0)
        self.assertEqual(res["ln_partition_function"], 0.0)
        self.assertEqual(res["free_energy"], 0.0)
        self.assertEqual(res["min_energy"], 0)
        self.assertEqual(res["unpaired_probabilities"], (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(res["pair_probabilities"], {})

    def test_empty_and_single(self):
        for seq in ("", "A", "G"):
            res = partition_fold(seq)
            self.assertEqual(res["ln_partition_function"], 0.0)
            self.assertEqual(res["partition_function"], 1.0)
            self.assertEqual(
                res["unpaired_probabilities"], (1.0,) * len(seq)
            )
            self.assertEqual(res["pair_probabilities"], {})

    def test_two_structure_exact_values(self):
        # GAAAC (n=5): structures are S0 = all unpaired (E=0, w=1) and
        # S1 = pair (0,4) GC (E=-3, w=e^3).  Z = 1 + e^3.
        e3 = math.exp(3.0)
        z = 1.0 + e3
        res = partition_fold("GAAAC", temperature=1.0)
        self.assertAlmostEqual(res["partition_function"], z, places=12)
        self.assertAlmostEqual(
            res["ln_partition_function"], math.log(z), places=12
        )
        self.assertAlmostEqual(res["free_energy"], -math.log(z), places=12)
        self.assertEqual(res["min_energy"], -3)
        self.assertAlmostEqual(
            res["pair_probabilities"][(0, 4)], e3 / z, places=12
        )
        # interior bases 1,2,3 are unpaired in every structure
        self.assertEqual(
            res["unpaired_probabilities"][1:4], (1.0, 1.0, 1.0)
        )
        # endpoints share their fate: P(unpaired) = 1 / Z
        self.assertAlmostEqual(
            res["unpaired_probabilities"][0], 1.0 / z, places=12
        )
        self.assertAlmostEqual(
            res["unpaired_probabilities"][4], 1.0 / z, places=12
        )

    def test_three_structure_exact_values(self):
        # GAAACC (n=6): legal GC pairs are (0,4) and (0,5); they share
        # endpoint 0 so exactly three structures exist:
        #   S0 all unpaired        E=0,  w=1
        #   S1 pair (0,4)          E=-3, w=e^3
        #   S2 pair (0,5)          E=-3, w=e^3
        e3 = math.exp(3.0)
        z = 1.0 + 2.0 * e3
        res = partition_fold("GAAACC")
        self.assertAlmostEqual(res["partition_function"], z, places=12)
        pp = res["pair_probabilities"]
        self.assertAlmostEqual(pp[(0, 4)], e3 / z, places=12)
        self.assertAlmostEqual(pp[(0, 5)], e3 / z, places=12)
        # position 4 is unpaired in S0 and S2: P = (1 + e^3)/Z
        self.assertAlmostEqual(
            res["unpaired_probabilities"][4], (1.0 + e3) / z, places=12
        )
        # position 0 is paired in both minimum structures
        self.assertAlmostEqual(
            res["unpaired_probabilities"][0], 1.0 / z, places=12
        )
        ok, bad = marginals_normalised(res, 6)
        self.assertTrue(ok, bad)

    def test_unpaired_energy_shifts_weights(self):
        # GAAAC with unpaired energy u=1:
        #   S0 all unpaired: E = 5, w = e^-5
        #   S1 pair (0,4):   E = 3*1 - 3 = 0, w = 1
        res = partition_fold("GAAAC", unpaired_energy=1)
        z = 1.0 + math.exp(-5.0)
        self.assertAlmostEqual(res["partition_function"], z, places=12)
        self.assertEqual(res["min_energy"], 0)
        self.assertAlmostEqual(
            res["pair_probabilities"][(0, 4)], 1.0 / z, places=12
        )
        self.assertAlmostEqual(
            res["unpaired_probabilities"][0], math.exp(-5.0) / z, places=12
        )

    def test_temperature_scales_weights(self):
        # Same two structures at T=2: Z = 1 + e^(3/2).
        res = partition_fold("GAAAC", temperature=2)
        z = 1.0 + math.exp(1.5)
        self.assertAlmostEqual(res["partition_function"], z, places=12)
        self.assertEqual(res["temperature"], 2.0)


class AnalyticRelationTests(unittest.TestCase):
    def test_zero_energies_uniform_over_structures(self):
        # Every pair energy 0 and unpaired energy 0 => every admissible
        # structure has weight 1, so Z is exactly the structure count.
        zero = {k: 0 for k in DEFAULT_PAIR_ENERGIES}
        rng = random.Random(5050)
        for n in (1, 5, 8, 11, 14):
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            res = partition_fold(seq, pair_energies=zero)
            ref = brute_partition(seq, pair_energies=zero)
            self.assertAlmostEqual(
                res["partition_function"],
                float(ref["structure_count"]),
                places=9,
            )
            self.assertAlmostEqual(
                res["ln_partition_function"],
                math.log(ref["structure_count"]),
                places=9,
            )
            for key, p in res["pair_probabilities"].items():
                self.assertAlmostEqual(p, ref["pair_probabilities"][key],
                                       places=9)

    def test_zero_energy_z_equals_structure_count_at_cap(self):
        # Analytic check at the n=20 cap via an INDEPENDENT exact-count
        # recurrence (plain integer counting -- no log DP, no call into the
        # partition code): with all energies zero, Z must equal the exact
        # number of admissible structures and marginals must be rational
        # counts/total.
        legal = {
            ("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"),
            ("G", "U"), ("U", "G"),
        }
        seq = "GGCCAAUUGGCCAAUUGGCC"  # n=20, many complementary options
        self.assertEqual(len(seq), MAX_PARTITION_SEQUENCE_LENGTH)
        n = len(seq)
        count = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            count[i][i] = 1
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length
                total = count[i + 1][j]
                for t in range(i + 4, j):
                    if (seq[i], seq[t]) in legal:
                        total += count[i + 1][t] * count[t + 1][j]
                count[i][j] = total
        zero = {k: 0 for k in DEFAULT_PAIR_ENERGIES}
        res = partition_fold(seq, pair_energies=zero)
        self.assertAlmostEqual(
            res["partition_function"], float(count[0][n]), places=9
        )
        self.assertAlmostEqual(
            res["ln_partition_function"], math.log(count[0][n]), places=10
        )
        # every marginal is a rational count / total in (0, 1]
        self.assertTrue(all(0.0 < p <= 1.0
                            for p in res["unpaired_probabilities"]))

    def test_string_x_constraint_against_enumeration(self):
        # Dot-bracket-style string constraint ('x' = forced unpaired) must
        # match the enumeration reference too.
        seq = "GCGCUAGCUAA"  # n=11
        pattern = "x....x....."
        self.assertEqual(len(pattern), len(seq))
        res = partition_fold(seq, pattern)
        ref = brute_partition(
            seq, forced_unpaired=frozenset({0, 5})
        )
        self.assertAlmostEqual(
            res["ln_partition_function"], ref["ln_z"], delta=LOG_TOL
        )
        self.assertEqual(res["unpaired_probabilities"][0], 1.0)
        self.assertEqual(res["unpaired_probabilities"][5], 1.0)
        for key, p in res["pair_probabilities"].items():
            self.assertAlmostEqual(
                p, ref["pair_probabilities"][key], delta=PROB_TOL
            )

    def test_free_energy_identity(self):
        rng = random.Random(31)
        for _ in range(20):
            n = rng.randint(1, 14)
            t = rng.choice([0.5, 1.0, 3.7])
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            res = partition_fold(seq, temperature=t)
            self.assertAlmostEqual(
                res["free_energy"],
                -t * res["ln_partition_function"],
                places=10,
            )
            self.assertAlmostEqual(
                res["partition_function"],
                math.exp(res["ln_partition_function"]),
                places=8,
            )

    def test_low_temperature_selects_minimum_energy_structures(self):
        # Default energies are the negation of fold()'s scores, so as T -> 0
        # the ensemble concentrates on fold()'s maximum-score structures.
        # At tiny T every reported marginal must be 0 or 1 (no scoring ties
        # here), and min_energy must equal -score.
        seq = "GGCGAUUAGCCGAU"  # n=13, unique-ish optimum
        opt = fold(seq)
        res = partition_fold(seq, temperature=1e-6)
        self.assertEqual(res["min_energy"], -opt["score"])
        self.assertTrue(
            all(p == 1.0 for p in res["pair_probabilities"].values())
        )
        # the pairs selected at T -> 0 are exactly the lex-min optimum's
        # pairs (that structure achieves the minimum)
        stack: list = []
        chosen = set()
        for i, ch in enumerate(opt["structure"]):
            if ch == "(":
                stack.append(i)
            elif ch == ")":
                chosen.add((stack.pop(), i))
        self.assertEqual(set(res["pair_probabilities"]), chosen)

    def test_tied_minima_share_probability_at_low_temperature(self):
        # GAAACC has two equally minimal structures; at T -> 0 each pair
        # marginal tends to 1/2 (checked against the enumeration reference,
        # whose log-sum-exp centering handles the tiny temperature exactly).
        seq = "GAAACC"
        res = partition_fold(seq, temperature=1e-9)
        ref = brute_partition(seq, temperature=1e-9)
        self.assertAlmostEqual(
            res["pair_probabilities"][(0, 4)],
            ref["pair_probabilities"][(0, 4)],
            places=9,
        )
        self.assertAlmostEqual(
            res["pair_probabilities"][(0, 4)], 0.5, places=6
        )

    def test_high_temperature_approaches_uniform(self):
        # With bounded energies, T -> infinity makes all weights -> 1.
        zero = {k: 0 for k in DEFAULT_PAIR_ENERGIES}
        seq = "GCGCUAGCAA"
        hot = partition_fold(seq, temperature=1e9)
        uniform = partition_fold(seq, pair_energies=zero)
        for key, p in uniform["pair_probabilities"].items():
            self.assertAlmostEqual(hot["pair_probabilities"][key], p,
                                   places=6)


class BruteForceComparisonTests(unittest.TestCase):
    """DP partition function vs direct structure enumeration."""

    def _check_against_brute(
        self, seq, temperature=1.0, pair_energies=None, unpaired_energy=0,
        forced_pairs=frozenset(), forced_up=frozenset(),
    ):
        res = partition_fold(
            seq,
            constraints={
                "pairs": sorted(forced_pairs),
                "unpaired": sorted(forced_up),
            },
            temperature=temperature,
            pair_energies=pair_energies,
            unpaired_energy=unpaired_energy,
        )
        ref = brute_partition(
            seq,
            temperature=temperature,
            pair_energies=pair_energies,
            unpaired_energy=unpaired_energy,
            forced_pairs=forced_pairs,
            forced_unpaired=forced_up,
        )
        self.assertAlmostEqual(
            res["ln_partition_function"], ref["ln_z"],
            delta=LOG_TOL, msg=f"lnZ mismatch for {seq!r}"
        )
        self.assertEqual(res["min_energy"], ref["min_energy"])
        self.assertEqual(
            set(res["pair_probabilities"]), set(ref["pair_probabilities"]),
            msg=f"pair key mismatch for {seq!r}",
        )
        for key, p in res["pair_probabilities"].items():
            self.assertAlmostEqual(
                p, ref["pair_probabilities"][key], delta=PROB_TOL,
                msg=f"pair {key} for {seq!r}",
            )
        for k in range(len(seq)):
            self.assertAlmostEqual(
                res["unpaired_probabilities"][k], ref["unpaired"][k],
                delta=PROB_TOL, msg=f"unpaired {k} for {seq!r}",
            )

    def test_random_sequences_default_energies(self):
        rng = random.Random(20260924)
        for n in range(0, 15):
            for _ in range(10 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                self._check_against_brute(
                    seq, temperature=rng.choice([0.3, 1.0, 2.5, 10.0])
                )

    def test_random_sequences_custom_energies(self):
        rng = random.Random(7777)
        for _ in range(60):
            n = rng.randint(1, 13)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            # Arbitrary integer energies, including positive ones.
            custom = {
                k: rng.randint(-4, 4) for k in DEFAULT_PAIR_ENERGIES
            }
            self._check_against_brute(
                seq,
                temperature=rng.choice([0.7, 1.5, 5.0]),
                pair_energies=custom,
                unpaired_energy=rng.randint(-2, 2),
            )

    def test_constrained_vs_enumeration(self):
        legal_pairs = {
            ("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"),
            ("G", "U"), ("U", "G"),
        }
        rng = random.Random(424)
        for n in (8, 10, 12, 14):
            for _ in range(12):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                legal = [
                    (a, b)
                    for a in range(n)
                    for b in range(a + 4, n)
                    if (seq[a], seq[b]) in legal_pairs
                ]
                rng.shuffle(legal)
                forced: list = []
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
                self._check_against_brute(
                    seq,
                    temperature=rng.choice([0.5, 1.0, 2.0]),
                    forced_pairs=frozenset(forced),
                    forced_up=frozenset(up),
                )

    def test_forced_pair_has_unit_marginal(self):
        seq = "GAAAAAAAAAAAC"  # n=13, GC pair (0,12)
        res = partition_fold(seq, {"pairs": [(0, 12)]})
        self.assertEqual(res["pair_probabilities"][(0, 12)], 1.0)
        self.assertEqual(res["unpaired_probabilities"][0], 0.0)
        self.assertEqual(res["unpaired_probabilities"][12], 0.0)
        # every marginal still normalises
        ok, bad = marginals_normalised(res, 13)
        self.assertTrue(ok, bad)

    def test_string_constraint_form_matches(self):
        seq = "GCGAAAAU"  # n=8
        res = partition_fold(seq, "(......)")
        ref = brute_partition(seq, forced_pairs=frozenset({(0, 7)}))
        self.assertEqual(res["pair_probabilities"][(0, 7)], 1.0)
        self.assertAlmostEqual(
            res["ln_partition_function"], ref["ln_z"], delta=LOG_TOL
        )


class BoundaryTests(unittest.TestCase):
    def test_max_length_runs_and_normalises(self):
        seq = ("GCGAAACGC" * 3)[:20]
        self.assertEqual(len(seq), MAX_PARTITION_SEQUENCE_LENGTH)
        res = partition_fold(seq)
        self.assertEqual(len(res["unpaired_probabilities"]), 20)
        ok, bad = marginals_normalised(res, 20)
        self.assertTrue(ok, bad)
        self.assertTrue(all(0.0 <= p <= 1.0
                            for p in res["pair_probabilities"].values()))

    def test_length_21_rejected(self):
        with self.assertRaises(InvalidSequenceError) as cm:
            partition_fold("A" * 21)
        self.assertIn("21", str(cm.exception))
        self.assertIn(str(MAX_PARTITION_SEQUENCE_LENGTH), str(cm.exception))

    def test_partial_energy_override(self):
        # Override only GC; remaining keys keep their defaults.
        res = partition_fold("GAAAC", pair_energies={"GC": -7})
        z = 1.0 + math.exp(7.0)
        self.assertAlmostEqual(res["partition_function"], z, places=8)
        self.assertEqual(res["pair_energies"]["GC"], -7)
        self.assertEqual(res["pair_energies"]["AU"], -2)
        self.assertEqual(res["unpaired_energy"], 0)

    def test_extreme_integer_energies_stay_probabilistically_valid(self):
        # Energies around 10**400 cannot be represented as finite floats at
        # all, yet exact-int anchoring must still give correct probabilities.
        res = partition_fold(
            "GAAACC", pair_energies={"GC": -10**400}
        )
        # both GC-pair structures overwhelmingly dominate the unpaired one
        self.assertAlmostEqual(
            res["unpaired_probabilities"][0], 0.0, places=15
        )
        self.assertAlmostEqual(
            res["pair_probabilities"][(0, 4)], 0.5, places=9
        )
        self.assertAlmostEqual(
            res["pair_probabilities"][(0, 5)], 0.5, places=9
        )
        # Z itself saturates to infinity honestly; ln Z is +inf
        self.assertEqual(res["partition_function"], math.inf)
        self.assertEqual(res["ln_partition_function"], math.inf)

        # Strongly *unfavourable* pairs: the empty structure is the whole
        # ensemble, all finite.
        res2 = partition_fold(
            "GAAACC", pair_energies={"GC": 10**400}
        )
        self.assertEqual(res2["partition_function"], 1.0)
        self.assertEqual(res2["pair_probabilities"], {})
        self.assertEqual(
            res2["unpaired_probabilities"], (1.0,) * 6
        )

    def test_huge_positive_energies_above_any_numeric_sentinel(self):
        # Regression: an infeasibility marker must not be a finite integer,
        # or feasible intervals with huge positive energies get mistaken for
        # "no structure".  At 10**200 the empty structure (E=0) dominates;
        # a forced pair (E=10**200) still has probability exactly 1 while
        # Z itself underflows to 0.0.
        res = partition_fold(
            "GAAACC", pair_energies={"GC": 10**200}
        )
        self.assertEqual(res["min_energy"], 0)
        self.assertEqual(res["partition_function"], 1.0)

        forced = partition_fold(
            "GAAAAAAAAAAAC",
            {"pairs": [(0, 12)]},
            pair_energies={"GC": 10**200},
        )
        self.assertEqual(forced["min_energy"], 10**200)
        self.assertEqual(forced["partition_function"], 0.0)
        # ln Z = -10**200 stays finite at this magnitude (bit length ~665 <
        # 1024) while Z itself underflows; free_energy = -T ln Z = +10**200.
        self.assertTrue(math.isfinite(forced["ln_partition_function"]))
        self.assertLess(forced["ln_partition_function"], -10**199)
        self.assertGreater(forced["free_energy"], 10**199)
        self.assertEqual(forced["pair_probabilities"][(0, 12)], 1.0)


class InputValidationTests(unittest.TestCase):
    def test_bad_temperature(self):
        for bad in (0, -1, -0.5):
            with self.assertRaises(InvalidParameterError):
                partition_fold("GAAAC", temperature=bad)
        with self.assertRaises(InvalidParameterError):
            partition_fold("GAAAC", temperature=math.nan)
        with self.assertRaises(InvalidParameterError):
            partition_fold("GAAAC", temperature=math.inf)

    def test_temperature_wrong_type(self):
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", temperature=True)  # bool is not a number
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", temperature="1.0")
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", temperature=None)

    def test_bad_pair_energy_key(self):
        with self.assertRaises(InvalidParameterError) as cm:
            partition_fold("GAAAC", pair_energies={"AT": -1})
        self.assertIn("AT", str(cm.exception))

    def test_bad_pair_energy_value(self):
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", pair_energies={"GC": -1.5})
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", pair_energies={"GC": True})
        with self.assertRaises(InvalidParameterError):
            partition_fold("GAAAC", pair_energies={"GC": math.nan})
        with self.assertRaises(InvalidParameterError):
            partition_fold("GAAAC", pair_energies={"GC": math.inf})

    def test_bad_unpaired_energy(self):
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", unpaired_energy=0.5)
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", unpaired_energy=False)
        with self.assertRaises(InvalidParameterError):
            partition_fold("GAAAC", unpaired_energy=math.nan)
        with self.assertRaises(InvalidParameterError):
            partition_fold("GAAAC", unpaired_energy=math.inf)

    def test_pair_energies_wrong_type(self):
        with self.assertRaises(TypeError):
            partition_fold("GAAAC", pair_energies=[("GC", -1)])

    def test_sequence_errors_are_inherited(self):
        with self.assertRaises(InvalidSequenceError) as cm:
            partition_fold("ACGT")
        self.assertIn("position 3", str(cm.exception))
        with self.assertRaises(TypeError):
            partition_fold(b"ACGU")

    def test_constraint_errors_are_inherited(self):
        # crossing forced pairs rejected up front, before any DP work
        with self.assertRaises(ConstraintError):
            partition_fold("GCGCUAGCAA", {"pairs": [(0, 8), (4, 9)]})
        with self.assertRaises(ConstraintError):
            partition_fold("AAAAAAAA", {"pairs": [(0, 4)]})

    def test_no_partial_state_on_failure(self):
        energies = {"GC": -3}
        with self.assertRaises(ConstraintError):
            partition_fold("GCGAAAC", {"pairs": [(0, 9)]},
                           pair_energies=energies)
        self.assertEqual(energies, {"GC": -3})  # input untouched


if __name__ == "__main__":
    unittest.main()
