#!/usr/bin/env python3

import unittest

from run_decoder_sequence import resolve_batch_plan


class Decoder2SequenceBatchingTest(unittest.TestCase):
    def test_default_batch_size_runs_all_selected_shots_in_one_program(self):
        self.assertEqual(resolve_batch_plan(requested=37, batch_size_arg=0),
                         (37, 1))

    def test_explicit_batch_size_splits_selected_shots(self):
        self.assertEqual(resolve_batch_plan(requested=37, batch_size_arg=10),
                         (10, 4))


if __name__ == "__main__":
    unittest.main()
