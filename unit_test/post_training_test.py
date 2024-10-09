import unittest

import torch

from neural_dnf.neural_dnf import NeuralDNF
from neural_dnf.post_training import (
    get_thresholding_upper_bound,
    apply_threshold,
)


class TestPostTrainingProcesses(unittest.TestCase):
    def test_get_thresholding_upper_bound(self):
        ndnf = NeuralDNF(2, 3, 2, 1.0)
        ndnf.conjunctions.weights.data = torch.Tensor(
            [[-4, 3], [0.1, -2.9], [4.983, -3.2]]
        )
        ndnf.disjunctions.weights.data = torch.Tensor(
            [[0.1, 0.2, 0.3], [0.4, -5.027, 0.6]]
        )

        self.assertEqual(get_thresholding_upper_bound(ndnf), 5.04)

    def test_apply_threshold(self):
        ndnf = NeuralDNF(2, 3, 2, 1.0)
        ndnf.conjunctions.weights.data = torch.Tensor(
            [[0.1, 0.2], [-2, -0.5], [0.04, -3]]
        )
        ndnf.disjunctions.weights.data = torch.Tensor(
            [[0.15, -0.002, 3], [-0.09, 0.7, 0]]
        )

        og_conj_weight = ndnf.conjunctions.weights.data.clone()
        og_disj_weight = ndnf.disjunctions.weights.data.clone()

        apply_threshold(ndnf, og_conj_weight, og_disj_weight, 0.1)

        torch.testing.assert_close(
            torch.Tensor([[0, 6.0], [-6.0, -6.0], [0, -6.0]]),
            ndnf.conjunctions.weights.data,
        )
        torch.testing.assert_close(
            torch.Tensor([[6.0, 0, 6.0], [0, 6.0, 0]]),
            ndnf.disjunctions.weights.data,
        )
