# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import copy
import unittest

from research_gate import compare


class ResearchGateTest(unittest.TestCase):
    def setUp(self):
        metrics = dict(total=120, t=9, twoq=40, depth=80, tdepth=8)
        self.baseline = {"cases": [dict(
            case_id="adder", source_sha256="source", hold=metrics,
            preserve={**metrics, "total": 100, "t": 10},
            output_sha256="old")]}
        self.candidate = {"cases": [dict(
            case_id="adder", source_sha256="source", metrics=metrics.copy(),
            output_sha256="new", hold_certificate_verified=True)]}

    def test_three_gates_per_saved_t_boundary(self):
        for saved_t in (1, 2):
            for extra in (3 * saved_t, 3 * saved_t + 1):
                metrics = self.candidate["cases"][0]["metrics"]
                metrics.update(total=100 + extra, t=10 - saved_t)
                result = compare(self.baseline, self.candidate)
                self.assertTrue(result["admissible"])
                self.assertEqual(result["versus_preserve"], {"loss": 1})
                label = "t_tradeoff" if extra == 3 * saved_t else "loss"
                self.assertEqual(result["versus_target"], {label: 1})

    def test_gate_win_cannot_give_back_hold_t_savings(self):
        self.candidate["cases"][0]["metrics"].update(total=99, t=10)
        result = compare(self.baseline, self.candidate)
        self.assertFalse(result["admissible"])
        self.assertTrue(any("T ceiling exceeded" in x for x in result["issues"]))

    def test_evidence_and_coverage_are_mandatory(self):
        for mutation in ("missing", "duplicate", "binding", "proof", "failure"):
            candidate = copy.deepcopy(self.candidate)
            if mutation == "missing":
                candidate["cases"] = []
            elif mutation == "duplicate":
                candidate["cases"] *= 2
            elif mutation == "binding":
                candidate["cases"][0]["source_sha256"] = "other"
            elif mutation == "proof":
                candidate["cases"][0]["hold_certificate_verified"] = False
            else:
                candidate["cases"][0]["metrics"] = None
            with self.subTest(mutation=mutation):
                self.assertFalse(compare(self.baseline, candidate)["admissible"])


if __name__ == "__main__":
    unittest.main()
