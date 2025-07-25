import unittest

import torch

from neural_dnf.neural_dnf import NeuralDNF
from neural_dnf.post_training import (
    get_thresholding_upper_bound,
    apply_threshold,
    condense_neural_dnf_model,
)


class TestPostTrainingProcesses(unittest.TestCase):
    def test_get_thresholding_upper_bound(self):
        ndnf = NeuralDNF(2, 3, 2, 1.0)
        ndnf.conjunctions.weights.data = torch.tensor(
            [[-4, 3], [0.1, -2.9], [4.983, -3.2]]
        )
        ndnf.disjunctions.weights.data = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, -5.027, 0.6]]
        )

        self.assertEqual(get_thresholding_upper_bound(ndnf), 5.04)

    def test_apply_threshold(self):
        ndnf = NeuralDNF(2, 3, 2, 1.0)
        ndnf.conjunctions.weights.data = torch.tensor(
            [[0.1, 0.2], [-2, -0.5], [0.04, -3]]
        )
        ndnf.disjunctions.weights.data = torch.tensor(
            [[0.15, -0.002, 3], [-0.09, 0.7, 0]]
        )

        og_conj_weight = ndnf.conjunctions.weights.data.clone()
        og_disj_weight = ndnf.disjunctions.weights.data.clone()

        apply_threshold(ndnf, og_conj_weight, og_disj_weight, 0.1)

        torch.testing.assert_close(
            torch.tensor([[0, 6.0], [-6.0, -6.0], [0, -6.0]]),
            ndnf.conjunctions.weights.data,
        )
        torch.testing.assert_close(
            torch.tensor([[6.0, 0, 6.0], [0, 6.0, 0]]),
            ndnf.disjunctions.weights.data,
        )

    def test_condense_neural_dnf_model(self):
        ndnf = NeuralDNF(7, 10, 3, 1.0)

        ndnf.conjunctions.weights.data = torch.randn(10, 7, dtype=torch.float32)
        # Make sure there is a non-zero weight
        ndnf.disjunctions.weights.data[0, 0] = 0.1

        ndnf.disjunctions.weights.data = torch.randn(3, 10, dtype=torch.float32)
        # Make sure there is a non-zero weight
        ndnf.disjunctions.weights.data[0, 0] = 0.1

        # Trying to condense a model with weights not in the set {-6, 0, 6}
        # should raise an AssertionError
        with self.assertRaises(
            AssertionError,
            msg="Conjunction weights are not in the set {-6, 0, 6}",
        ):
            condense_neural_dnf_model(ndnf)

        ndnf.conjunctions.weights.data = torch.zeros(10, 7, dtype=torch.float32)
        with self.assertRaises(
            AssertionError,
            msg="Disjunction weights are not in the set {-6, 0, 6}",
        ):
            condense_neural_dnf_model(ndnf)

        # Now we set the weights to be in the set {-6, 0, 6}
        ndnf.conjunctions.weights.data = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],  # Conj 0
                [0, 0, 0, 0, 0, 0, 0],  # Conj 1
                [6, 6, 0, 0, 0, 0, 0],  # Conj 2
                [0, 0, 0, 0, 0, 0, 0],  # Conj 3
                [-6, 0, 0, 0, 0, 0, 0],  # Conj 4
                [0, 0, 0, -6, 6, 0, 0],  # Conj 5
                [0, 0, 0, 0, 0, 0, 0],  # Conj 6
                [0, 0, 0, 6, 0, 0, 0],  # Conj 7
                [6, 0, 0, 0, -6, -6, 0],  # Conj 8
                [0, 0, 6, 0, 0, 0, -6],  # Conj 9
            ],
            dtype=torch.float32,
        )
        # Actually used conjunctions: 2, 4, 5, 9
        ndnf.disjunctions.weights.data = torch.tensor(
            [
                [0, 0, 6, 0, -6, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -6, 0, 0, 0, 6],
            ],
            dtype=torch.float32,
        )

        condensed_model = condense_neural_dnf_model(ndnf)
        # The condensed model should have 4 conjunctions, but still have 3
        # disjunctions
        torch.testing.assert_close(
            torch.tensor(
                [
                    [6, 6, 0, 0, 0, 0, 0],  # 2 -> 0
                    [-6, 0, 0, 0, 0, 0, 0],  # 4 -> 1
                    [0, 0, 0, -6, 6, 0, 0],  # 5 -> 2
                    [0, 0, 6, 0, 0, 0, -6],  # 9 -> 3
                ],
                dtype=torch.float32,
            ),
            condensed_model.conjunctions.weights.data,
        )

        torch.testing.assert_close(
            torch.tensor(
                [
                    [6, -6, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, -6, 6],
                ],
                dtype=torch.float32,
            ),
            condensed_model.disjunctions.weights.data,
        )
