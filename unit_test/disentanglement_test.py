import unittest

import torch

from neural_dnf.post_training import (
    split_entangled_conjunction,
    split_entangled_disjunction,
    split_positively_used_conjunction,
    split_negatively_used_conjunction,
)


class TestDisentanglement(unittest.TestCase):
    def test_disentangle_conjunction_with_zero_weights(self):
        # Positively used
        self.assertIsNone(split_entangled_conjunction(torch.zeros(10)))

        # Negatively used
        self.assertIsNone(split_entangled_conjunction(torch.zeros(10), sign=-1))

    def test_conjunction_with_only_one_non_zero_weight(self):
        test_conjunction_weights = [
            torch.tensor([0, 0, 6, 0, 0, 0, 0], dtype=torch.float32),
            torch.tensor([0, -2], dtype=torch.float32),
        ]
        expected_number_of_splits = [1, 1]

        for i, tw in enumerate(test_conjunction_weights):
            for sign in [1, -1]:
                split = split_entangled_conjunction(tw, sign=sign)
                self.assertIsNotNone(split)
                assert split is not None  # for pylance type hinting
                self.assertEqual(len(split), expected_number_of_splits[i])

    def test_disentangle_positively_used_conjunction(self):
        test_conjunction_weights = [
            torch.tensor([0, 0, -6, -6, 2, -2, -2, 0, 0], dtype=torch.float32),
            torch.tensor([0, 0, -6, -6, 3, -2, -2, 0, 0], dtype=torch.float32),
            torch.tensor([0, 0, -6, -6, 0, -3, -3, 0, 0], dtype=torch.float32),
        ]
        expected_number_of_splits = [3, 2, 1]

        for i, tw in enumerate(test_conjunction_weights):
            split = split_entangled_conjunction(tw, sign=1)
            self.assertIsNotNone(split)
            assert split is not None  # for pylance type hinting
            self.assertEqual(len(split), expected_number_of_splits[i])

            for s in split:
                input_entry = torch.sign(tw)
                for j in torch.where(s == 0)[0]:
                    input_entry[j] *= -1
                self.assertTrue(
                    torch.sum(tw * input_entry)
                    + torch.abs(tw).max()
                    - torch.abs(tw).sum()
                    > 0
                )
                self.assertTrue(
                    torch.sum(s * input_entry)
                    + torch.abs(s).max()
                    - torch.abs(s).sum()
                    > 0
                )

    def test_disentangle_positively_used_conjunction_with_limit(self):
        tw = torch.tensor(
            [-6, -6, 2, -2, -2, 1, 1, -1, 0.5, 0.5, 0.3, -0.2, 0.1, 0.1, 0.1],
            dtype=torch.float32,
        )

        split = split_positively_used_conjunction(tw, j_minus_explore_limit=3)
        self.assertGreater(len(split), 0)

        for s in split:
            self.assertEqual(torch.where(s == 0)[0].shape[0], 3)
            input_entry = torch.sign(tw)
            for j in torch.where(s == 0)[0]:
                input_entry[j] *= -1
            self.assertTrue(
                torch.sum(tw * input_entry)
                + torch.abs(tw).max()
                - torch.abs(tw).sum()
                > 0
            )
            self.assertTrue(
                torch.sum(s * input_entry)
                + torch.abs(s).max()
                - torch.abs(s).sum()
                > 0
            )

    def test_disentangle_negatively_used_conjunction(self):
        test_conjunction_weights = [
            torch.tensor([-1.3, -2.2, 6], dtype=torch.float32),
            torch.tensor([-2, -1, -6, 2], dtype=torch.float32),
            torch.tensor([-2, 0, 2, -2], dtype=torch.float32),
            torch.tensor([0, 0, -0.4, 0], dtype=torch.float32),
        ]

        expected_number_of_splits_for_opt = [2, 2, 3, 1]

        for i, tw in enumerate(test_conjunction_weights):
            split = split_entangled_conjunction(tw, sign=-1)
            self.assertIsNotNone(split)
            assert split is not None  # for pylance type hinting
            self.assertEqual(len(split), expected_number_of_splits_for_opt[i])

            for s in split:
                input_entry = torch.sign(tw)
                for j in torch.where(s != 0)[0]:
                    input_entry[j] *= -1

                # The og conjunction should be false
                self.assertTrue(
                    torch.sum(tw * input_entry)
                    + torch.abs(tw).max()
                    - torch.abs(tw).sum()
                    < 0
                )
                # The new split should be true
                self.assertTrue(
                    torch.sum(s * input_entry)
                    + torch.abs(s).max()
                    - torch.abs(s).sum()
                    > 0
                )

    def test_disentangle_negatively_used_conjunction_with_limit(self):
        tw = torch.tensor([-2, -1, -6, 2, -0.5, 0.1, 0.3], dtype=torch.float32)

        split = split_negatively_used_conjunction(tw, j_minus_explore_limit=2)
        for s in split:
            self.assertLessEqual(torch.where(s != 0)[0].shape[0], 2)
            input_entry = torch.sign(tw)
            for j in torch.where(s != 0)[0]:
                input_entry[j] *= -1

            # The og conjunction should be false
            self.assertTrue(
                torch.sum(tw * input_entry)
                + torch.abs(tw).max()
                - torch.abs(tw).sum()
                < 0
            )
            # The new split should be true
            self.assertTrue(
                torch.sum(s * input_entry)
                + torch.abs(s).max()
                - torch.abs(s).sum()
                > 0
            )

    def test_disentangle_disjunction_with_zero_weights(self):
        self.assertIsNone(split_entangled_disjunction(torch.zeros(10)))

    def test_disjunction_with_only_one_non_zero_weight(self):
        test_disjunction_weights = [
            torch.tensor([0, 0, 6, 0, 0, 0, 0], dtype=torch.float32),
            torch.tensor([0, -2], dtype=torch.float32),
        ]

        for i, tw in enumerate(test_disjunction_weights):
            split = split_entangled_disjunction(tw)
            self.assertIsNotNone(split)
            assert split is not None  # for pylance type hinting
            self.assertEqual(len(split), 1)

            split_item = split[0]
            # The item should be a tensor and a flag
            # The tensor's sign should be the same as the input's sign, and the flag True
            torch.testing.assert_close(tw.sign(), split_item[0].sign())
            self.assertTrue(split_item[1])

    def test_disentangle_disjunction(self):
        test_disjunction_weights = [
            torch.tensor([0, 0, 6, -3, 3, -6], dtype=torch.float32),
            torch.tensor([0.722, -0.12, 0.24], dtype=torch.float32),
            torch.tensor([1, -1.3, -1.4], dtype=torch.float32),
        ]
        expected_number_split_style = [(3, False), (1, False), (1, True)]

        for i, tw in enumerate(test_disjunction_weights):
            split = split_entangled_disjunction(tw)
            self.assertIsNotNone(split)
            assert split is not None  # for pylance type hinting
            self.assertEqual(len(split), expected_number_split_style[i][0])
            for s in split:
                self.assertEqual(s[1], expected_number_split_style[i][1])
