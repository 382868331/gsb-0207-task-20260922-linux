"""Tests for the Boltzmann partition function and pair marginals.

Expected values come from three independent sources, never from the
implementation under test:

1. hand-computed analytic examples (tiny sequences with one or two
   structures);
2. identities that any correct answer must satisfy (marginal normalization,
   entropy derivative, homogeneity, temperature limits);
3. ``tests.brute_reference.brute_partition`` -- a separate enumeration that
   lists every structure and sums plain ``exp(-E/T)`` weights without
   importing the module under test.
"""

from __future__ import annotations

import math
import random
import unittest

from rna_fold import (
    DEFAULT_PAIR_ENERGIES,
    MAX_PARTITION_LENGTH,
    ConstraintError,
    InvalidSequenceError,
    UnsatisfiableConstraintError,
    partition_function,
)
from tests.brute_reference import brute_partition


def _matrix_row_sum_equals_one(res, tol=1e-9):
    """unpaired + all pair marginals must total 1 at every position."""
    n = len(res["unpaired_probabilities"])
    m = res["pair_prob_matrix"]
    for i in range(n):
        total = res["unpaired_probabilities"][i]
        total += sum(m[i])
        total += sum(m[k][i] for k in range(n))
        if abs(total - 1.0) > tol:
            return False, i, total
    return True, None, None


def _triples_to_map(res):
    return {(i, j): p for i, j, p in res["pair_probabilities"]}


class HandComputedTests(unittest.TestCase):
    def test_single_pair_two_structures(self):
        # n=5 "GAAAC": exactly two admissible structures:
        #   empty         E=0  weight 1
        #   pair (0,4)    E=-3 weight e^3
        seq = "GAAAC"
        res = partition_function(seq)
        z = 1.0 + math.exp(3.0)
        self.assertAlmostEqual(res["log_partition"], math.log(z), places=12)
        self.assertAlmostEqual(res["free_energy"], -math.log(z), places=12)
        m = res["pair_prob_matrix"]
        self.assertAlmostEqual(m[0][4], math.exp(3.0) / z, places=12)
        self.assertAlmostEqual(res["unpaired_probabilities"][0], 1.0 / z, places=12)
        # inner positions are always unpaired
        for k in (1, 2, 3):
            self.assertEqual(res["unpaired_probabilities"][k], 1.0)
        # only one pair can carry probability
        self.assertEqual([t[:2] for t in res["pair_probabilities"]], [(0, 4)])

    def test_no_admissible_pair_is_all_unpaired(self):
        # n=4 cannot host any pair (j - i >= 4 impossible), and a sequence
        # with no complementary letters has the same property at n=5.
        res = partition_function("AAAA")
        self.assertEqual(res["log_partition"], 0.0)
        self.assertEqual(res["pair_probabilities"], ())
        self.assertEqual(res["unpaired_probabilities"], [1.0] * 4)

        res5 = partition_function("AAAAA")
        self.assertEqual(res5["log_partition"], 0.0)
        self.assertEqual(res5["unpaired_probabilities"], [1.0] * 5)

    def test_unpaired_energy_shifts_log_partition(self):
        # With zero possible pairs every structure is the all-unpaired one:
        # E = n * e_u, so ln Z = -n * e_u / T.
        res = partition_function("AAAAA", unpaired_energy=2, temperature=2.0)
        self.assertAlmostEqual(res["log_partition"], -5.0, places=12)
        self.assertEqual(res["unpaired_energy"], 2)

    def test_pair_energy_override_hand_computed(self):
        # Give the only pair (0,4) a positive energy +1: empty wins weight
        # e^0 = 1 vs e^{-1}; pair probability = e^{-1}/(1+e^{-1}).
        res = partition_function(
            "GAAAC",
            pair_energies={"GC": 1, "CG": 1},
        )
        p_pair = math.exp(-1.0) / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(res["pair_prob_matrix"][0][4], p_pair, places=12)
        self.assertAlmostEqual(
            res["log_partition"], math.log1p(math.exp(-1.0)), places=12
        )

    def test_empty_sequence(self):
        res = partition_function("")
        self.assertEqual(res["log_partition"], 0.0)
        self.assertEqual(res["free_energy"], 0.0)
        self.assertEqual(res["pair_prob_matrix"], [])
        self.assertEqual(res["pair_probabilities"], ())
        self.assertEqual(res["unpaired_probabilities"], [])

    def test_two_competing_pairs_sigmoid(self):
        # n=7 "GAAAACC": position 0 (G) can pair with t=5 or t=6 (both C),
        # and the empty structure is allowed too.  The two pair structures
        # have equal energy, so their marginal probabilities are equal.
        seq = "GAAAACC"
        res = partition_function(seq)
        ref = brute_partition(seq)
        self.assertAlmostEqual(
            res["log_partition"], ref["log_partition"], places=12
        )
        p05 = res["pair_prob_matrix"][0][5]
        p06 = res["pair_prob_matrix"][0][6]
        self.assertAlmostEqual(p05, p06, places=12)
        # z = 1 + 2 e^3  -> each pair has probability e^3 / z
        z = 1.0 + 2.0 * math.exp(3.0)
        self.assertAlmostEqual(p05, math.exp(3.0) / z, places=12)

    def test_temperature_breaks_or_softens(self):
        # "GAAACGAAAC" (n=10) has a unique ground state: the two disjoint
        # GC pairs (0,4) and (5,9), energy -6 (any spanning pair blocks one
        # of them and reaches only -3).  Cold -> both marginals tend to 1;
        # hot -> probabilities match the uniform structure enumeration.
        seq = "GAAACGAAAC"
        cold = partition_function(seq, temperature=1e-6)
        self.assertGreater(cold["pair_prob_matrix"][0][4], 1.0 - 1e-9)
        self.assertGreater(cold["pair_prob_matrix"][5][9], 1.0 - 1e-9)
        hot = partition_function(seq, temperature=1e9)
        self.assertTrue(
            all(0.0 <= p <= 1.0 for _, _, p in hot["pair_probabilities"])
        )
        ref = brute_partition(seq, temperature=1e9)
        self.assertAlmostEqual(
            hot["log_partition"], ref["log_partition"], places=6
        )

    def test_zero_pair_probabilities_are_exact_zeros(self):
        res = partition_function("GAAAC")
        m = res["pair_prob_matrix"]
        # (0,1) illegal geometry and (1,4) non-complementary: exact zero.
        self.assertEqual(m[0][1], 0.0)
        self.assertEqual(m[1][4], 0.0)
        self.assertEqual(m[4][0], 0.0)  # upper triangle only


class InvariantsTests(unittest.TestCase):
    def test_marginals_normalize_everywhere(self):
        rng = random.Random(20260924)
        for n in range(0, 17):
            for _ in range(6 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                res = partition_function(
                    seq,
                    temperature=rng.choice([0.3, 1.0, 2.5, 17.0]),
                    unpaired_energy=rng.choice([-1, 0, 1]),
                )
                ok, i, total = _matrix_row_sum_equals_one(res)
                self.assertTrue(ok, f"{seq}: row {i} sums to {total}")

    def test_probabilities_in_unit_interval(self):
        rng = random.Random(77)
        for _ in range(40):
            n = rng.randint(1, 18)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            res = partition_function(seq, temperature=rng.uniform(0.05, 5.0))
            for i, j, p in res["pair_probabilities"]:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0 + 1e-12)
                self.assertLess(j - i, n + 1)

    def test_triples_and_matrix_agree_and_sorted(self):
        seq = "GGCGAUUAGCCG"
        res = partition_function(seq)
        m = res["pair_prob_matrix"]
        triples = res["pair_probabilities"]
        self.assertEqual(triples, tuple(sorted(triples)))
        for i, j, p in triples:
            self.assertEqual(m[i][j], p)
        n = len(seq)
        flat = [m[i][j] for i in range(n) for j in range(n)]
        # every non-zero upper-triangular entry appears in the triples
        nonzero = {(i, j) for i in range(n) for j in range(n) if m[i][j] > 0.0}
        self.assertEqual(nonzero, {(i, j) for i, j, _ in triples})

    def test_log_partition_derivative_is_expected_energy(self):
        # d ln Z / d(1/T) = -<E>; check with a central finite difference and
        # independently evaluate <E> from the brute-force pair marginals.
        seq = "GGCGAUUAGCU"
        b0 = 1.0 / 1.7
        h = 1e-5
        lp_plus = partition_function(seq, temperature=1.0 / (b0 + h))["log_partition"]
        lp_minus = partition_function(seq, temperature=1.0 / (b0 - h))["log_partition"]
        deriv = (lp_plus - lp_minus) / (2 * h)

        ref = brute_partition(seq, temperature=1.0 / b0)
        mean_e = sum(
            DEFAULT_PAIR_ENERGIES[seq[i] + seq[j]] * p
            for (i, j), p in ref["pair_prob"].items()
        )  # unpaired energy is 0 by default
        self.assertAlmostEqual(deriv, -mean_e, places=5)

    def test_energy_scale_temperature_equivalence(self):
        # Doubling every energy and doubling T leaves every probability
        # unchanged (weights exp(-E/T) identical).
        seq = "GGCGAUUAGCCGAAU"
        scaled = {k: 2 * v for k, v in DEFAULT_PAIR_ENERGIES.items()}
        a = partition_function(seq, temperature=1.0)
        b = partition_function(
            seq, temperature=2.0,
            pair_energies=scaled, unpaired_energy=0,
        )
        for (i, j, pa), (_, _, pb) in zip(
            a["pair_probabilities"], b["pair_probabilities"]
        ):
            self.assertAlmostEqual(pa, pb, places=10)
        for ua, ub in zip(
            a["unpaired_probabilities"], b["unpaired_probabilities"]
        ):
            self.assertAlmostEqual(ua, ub, places=10)
        self.assertAlmostEqual(
            b["log_partition"], a["log_partition"], places=10
        )

    def test_extreme_energy_ratios_stay_numerically_stable(self):
        # Huge favorable GC vs everything else: marginals must still
        # normalize despite log-weight spans of hundreds.
        seq = "GAAACGAAACGAAACGAAAC"[:20]
        energies = dict(DEFAULT_PAIR_ENERGIES)
        energies["GC"] = -400
        energies["CG"] = -400
        res = partition_function(seq, temperature=1.0, pair_energies=energies)
        self.assertTrue(math.isfinite(res["log_partition"]))
        ok, _, total = _matrix_row_sum_equals_one(res, tol=1e-7)
        self.assertTrue(ok, f"row sums to {total}")
        self.assertTrue(
            all(math.isfinite(p) for _, _, p in res["pair_probabilities"])
        )


class BruteForceComparisonTests(unittest.TestCase):
    """Log-space DP vs direct exhaustive summation for n <= 14."""

    def _check_against_brute(
        self, seq, forced=frozenset(), up=frozenset(),
        temperature=1.0, pair_energies=None, unpaired_energy=0, tol=1e-9
    ):
        res = partition_function(
            seq,
            {"pairs": sorted(forced), "unpaired": sorted(up)} if (forced or up) else None,
            temperature=temperature,
            pair_energies=pair_energies,
            unpaired_energy=unpaired_energy,
        )
        ref = brute_partition(
            seq, forced, up, temperature, pair_energies, unpaired_energy
        )
        self.assertTrue(ref["feasible"])
        self.assertAlmostEqual(
            res["log_partition"], ref["log_partition"], places=10
        )
        got = _triples_to_map(res)
        self.assertEqual(set(got), set(ref["pair_prob"]), seq)
        for key, p_ref in ref["pair_prob"].items():
            self.assertAlmostEqual(got[key], p_ref, delta=tol, msg=(seq, key))
        for k, p_ref in enumerate(ref["unpaired_prob"]):
            self.assertAlmostEqual(
                res["unpaired_probabilities"][k], p_ref, delta=tol
            )

    def test_random_sequences_unconstrained(self):
        rng = random.Random(424299)
        for n in range(0, 13):
            for _ in range(20 if n else 1):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                self._check_against_brute(
                    seq,
                    temperature=rng.choice([0.25, 1.0, 3.0, 50.0]),
                    unpaired_energy=rng.choice([-2, 0, 1]),
                )

    def test_length_14(self):
        rng = random.Random(3133701)
        for _ in range(6):
            seq = "".join(rng.choice("ACGU") for _ in range(14))
            self._check_against_brute(seq, temperature=0.8)

    def test_with_forced_pairs_and_unpaired(self):
        rng = random.Random(5150)
        legal_scores = {
            ("G", "C"): 3, ("C", "G"): 3,
            ("A", "U"): 2, ("U", "A"): 2,
            ("G", "U"): 1, ("U", "G"): 1,
        }
        for n in (8, 10, 12, 14):
            for _ in range(20):
                seq = "".join(rng.choice("ACGU") for _ in range(n))
                legal = [
                    (a, b)
                    for a in range(n)
                    for b in range(a + 4, n)
                    if (seq[a], seq[b]) in legal_scores
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
                ups = frozenset(
                    rng.sample(free, min(1, len(free)))
                ) if free else frozenset()
                self._check_against_brute(
                    seq, frozenset(forced), ups,
                    temperature=rng.choice([0.5, 1.0, 4.0]),
                )

    def test_string_constraint_form_matches_dict(self):
        seq = "GCGAAAAUCGAC"  # n=12; (0,11) is a legal GC pair
        self.assertEqual(len(seq), 12)
        pattern = "(..........)"
        self.assertEqual(len(pattern), len(seq))
        res_s = partition_function(seq, pattern)
        res_d = partition_function(seq, {"pairs": [(0, 11)]})
        self.assertAlmostEqual(
            res_s["log_partition"], res_d["log_partition"], places=12
        )
        self.assertEqual(
            res_s["pair_probabilities"], res_d["pair_probabilities"]
        )

    def test_custom_pair_energies_against_brute(self):
        custom = {"GC": -5, "CG": 2, "AU": 0, "UA": 0, "GU": 4, "UG": -1}
        rng = random.Random(909)
        for _ in range(15):
            n = rng.randint(5, 13)
            seq = "".join(rng.choice("ACGU") for _ in range(n))
            self._check_against_brute(
                seq, temperature=1.5, pair_energies=custom
            )


class ForcedConstraintBehaviorTests(unittest.TestCase):
    def test_forced_pair_has_marginal_one(self):
        seq = "GAAAAAAAAAAAAU"  # n=14, force (0,13)
        res = partition_function(seq, {"pairs": [(0, 13)]})
        self.assertAlmostEqual(res["pair_prob_matrix"][0][13], 1.0, places=12)
        self.assertAlmostEqual(res["unpaired_probabilities"][0], 0.0)
        self.assertAlmostEqual(res["unpaired_probabilities"][13], 0.0)

    def test_forced_unpaired_has_marginal_one(self):
        seq = "GGGGAAAACCCC"
        res = partition_function(seq, "x" + "." * 11)
        self.assertEqual(res["unpaired_probabilities"][0], 1.0)
        row0 = sum(res["pair_prob_matrix"][0]) + sum(
            row[0] for row in res["pair_prob_matrix"]
        )
        self.assertEqual(row0, 0.0)

    def test_unsatisfiable_error_type_is_stable(self):
        # With current validation every well-formed forced set is realizable
        # (geometry/complementarity/crossings are rejected up front), so the
        # DP's proven-empty guard is defense in depth; pin its class hierarchy
        # as part of the public contract, and verify that a forced-pair
        # endpoint that is also forced unpaired is rejected as ConstraintError.
        self.assertTrue(
            issubclass(UnsatisfiableConstraintError, ConstraintError)
        )
        seq = "GAAACAAAA"
        with self.assertRaises(ConstraintError):
            partition_function(seq, {"pairs": [(0, 4)], "unpaired": [0]})


class InputValidationTests(unittest.TestCase):
    def test_length_cap_is_twenty(self):
        self.assertEqual(MAX_PARTITION_LENGTH, 20)
        partition_function("A" * 20)  # boundary accepted
        with self.assertRaises(InvalidSequenceError) as cm:
            partition_function("A" * 21)
        self.assertIn("21", str(cm.exception))
        self.assertIn("20", str(cm.exception))

    def test_fold_still_allows_long_sequences(self):
        # The existing entry point keeps its 120 cap.
        from rna_fold import fold
        res = fold("A" * 120)
        self.assertEqual(res["structure"], "." * 120)

    def test_illegal_character(self):
        with self.assertRaises(InvalidSequenceError) as cm:
            partition_function("ACGT")
        self.assertIn("position 3", str(cm.exception))

    def test_bad_temperature_values(self):
        for bad in (0, -1, 0.0, -2.5):
            with self.assertRaises(ValueError):
                partition_function("GAAAC", temperature=bad)
        with self.assertRaises(ValueError) as cm:
            partition_function("GAAAC", temperature=math.nan)
        self.assertIn("NaN", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            partition_function("GAAAC", temperature=math.inf)
        self.assertIn("Infinity", str(cm.exception))
        with self.assertRaises(TypeError):
            partition_function("GAAAC", temperature=True)
        with self.assertRaises(TypeError):
            partition_function("GAAAC", temperature="hot")
        with self.assertRaises(ValueError):
            partition_function("GAAAC", temperature=10**400)

    def test_bad_pair_energies(self):
        with self.assertRaises(ValueError) as cm:
            partition_function("GAAAC", pair_energies={"XX": -1})
        self.assertIn("unknown key", str(cm.exception))
        with self.assertRaises(TypeError):
            partition_function("GAAAC", pair_energies={"GC": -1.5})
        with self.assertRaises(TypeError):
            partition_function("GAAAC", pair_energies={"GC": True})
        with self.assertRaises(TypeError):
            partition_function("GAAAC", pair_energies=[("GC", -1)])
        with self.assertRaises(ValueError):
            partition_function(
                "GAAAC", pair_energies={"GC": 10**400}
            )

    def test_bad_unpaired_energy(self):
        with self.assertRaises(TypeError):
            partition_function("GAAAC", unpaired_energy=0.5)
        with self.assertRaises(TypeError):
            partition_function("GAAAC", unpaired_energy=False)

    def test_constraint_errors_reused_from_fold(self):
        with self.assertRaises(ConstraintError):
            partition_function("GAAAC", {"pairs": [(0, 3)]})  # too close
        # n=15: (0,8)=GC and (3,13)=GC cross (0 < 3 < 8 < 13)
        chars = ["A"] * 15
        chars[0], chars[8], chars[3], chars[13] = "G", "C", "G", "C"
        with self.assertRaises(ConstraintError) as cm:
            partition_function("".join(chars), {"pairs": [(0, 8), (3, 13)]})
        self.assertIn("cross", str(cm.exception))
        with self.assertRaises(TypeError):
            partition_function("GAAAC", [])

    def test_partial_override_keeps_other_defaults(self):
        res = partition_function("GAAAC", pair_energies={"GC": -7})
        self.assertEqual(res["pair_energies"]["GC"], -7)
        # only the named oriented key changes; other keys keep their defaults
        self.assertEqual(res["pair_energies"]["CG"], DEFAULT_PAIR_ENERGIES["CG"])
        self.assertEqual(res["pair_energies"]["AU"], DEFAULT_PAIR_ENERGIES["AU"])

    def test_no_partial_state_on_failure(self):
        energies = {"GC": -9}
        with self.assertRaises(ConstraintError):
            partition_function("GAAAC", {"pairs": [(0, 1)]}, pair_energies=energies)
        # caller's mapping untouched
        self.assertEqual(energies, {"GC": -9})

    def test_effective_energy_table_is_independent_copy(self):
        res = partition_function("GAAAC", pair_energies={"GC": -11})
        res["pair_energies"]["GC"] = 999
        again = partition_function("GAAAC")
        self.assertEqual(again["pair_energies"]["GC"], DEFAULT_PAIR_ENERGIES["GC"])


class BoundaryLengthTests(unittest.TestCase):
    def test_length_twenty_runs_and_normalizes(self):
        seq = "GGCGAUUAGCCGAUUAGGCC"
        self.assertEqual(len(seq), 20)
        res = partition_function(seq)
        self.assertTrue(math.isfinite(res["log_partition"]))
        ok, _, total = _matrix_row_sum_equals_one(res, tol=1e-8)
        self.assertTrue(ok, f"row sums to {total}")

    def test_length_twenty_with_constraint(self):
        # Deterministic sequence of length 20 with a spanning GC pair.
        seq = "G" + "A" * 18 + "C"
        self.assertEqual(len(seq), 20)
        res = partition_function(seq, {"pairs": [(0, 19)]})
        self.assertAlmostEqual(res["pair_prob_matrix"][0][19], 1.0)
        ok, _, total = _matrix_row_sum_equals_one(res, tol=1e-8)
        self.assertTrue(ok, f"row sums to {total}")


if __name__ == "__main__":
    unittest.main()
