import unittest

import torch

from neural_dnf.neural_dnf import NeuralDNF, NeuralDNFEO


class TestNeuralDNF(unittest.TestCase):
    # This test case also covers all the functionality of `BaseNeuralDNF` class
    # since `NeuralDNF` is a subclass of `BaseNeuralDNF` without any changes.

    def test_get_delta_value(self):
        ndnf = NeuralDNF(5, 3, 2, 0.1)
        self.assertEqual(ndnf.get_delta_val(), [0.1, 0.1])

        ndnf_eo = NeuralDNFEO(5, 3, 2, 0.1)
        self.assertEqual(ndnf_eo.get_delta_val(), [0.1, 0.1, 0.1])

    def test_set_delta_value_with_one_float_change_both_delta_values(self):
        ndnf = NeuralDNF(5, 3, 2, 0.1)
        self.assertEqual(ndnf.get_delta_val(), [0.1, 0.1])

        ndnf.set_delta_val(0.2)
        self.assertEqual(ndnf.get_delta_val(), [0.2, 0.2])

        ndnf_eo = NeuralDNFEO(5, 3, 2, 0.1)
        self.assertEqual(ndnf_eo.get_delta_val(), [0.1, 0.1, 0.1])

        ndnf_eo.set_delta_val(0.4)
        self.assertEqual(ndnf_eo.get_delta_val(), [0.4, 0.4, 0.4])

    def test_set_delta_value_with_list(self):
        ndnf = NeuralDNF(5, 3, 2, 0.1)
        self.assertEqual(ndnf.get_delta_val(), [0.1, 0.1])

        ndnf.set_delta_val([0.2, 0.3])
        self.assertEqual(ndnf.get_delta_val(), [0.2, 0.3])

        ndnf_eo = NeuralDNFEO(5, 3, 2, 0.1)
        self.assertEqual(ndnf_eo.get_delta_val(), [0.1, 0.1, 0.1])

        ndnf_eo.set_delta_val([0.4, 0.5, 0.6])
        self.assertEqual(ndnf_eo.get_delta_val(), [0.4, 0.5, 0.6])

    def test_update_weight_wrt_mask(self):
        ndnf = NeuralDNF(4, 3, 2, 1.0)
        ndnf.conjunctions.weights.data.fill_(6.0)
        ndnf.disjunctions.weights.data.fill_(-6.0)

        ndnf.conj_weight_mask[0, 3] = 0
        ndnf.conj_weight_mask[1, 0] = 0
        ndnf.conj_weight_mask[1, 1] = 0
        ndnf.conj_weight_mask[2, 1] = 0

        ndnf.disj_weight_mask[0, 2] = 0

        ndnf.update_weight_wrt_mask()

        torch.testing.assert_close(
            ndnf.conjunctions.weights[0, 3], torch.tensor(0.0)
        )
        torch.testing.assert_close(
            ndnf.conjunctions.weights[1, 0], torch.tensor(0.0)
        )
        torch.testing.assert_close(
            ndnf.conjunctions.weights[1, 1], torch.tensor(0.0)
        )
        torch.testing.assert_close(
            ndnf.conjunctions.weights[2, 1], torch.tensor(0.0)
        )
        torch.testing.assert_close(
            ndnf.disjunctions.weights[0, 2], torch.tensor(0.0)
        )

    def test_check_delta(self):
        ndnf = NeuralDNF(4, 3, 2, 0.1)
        self.assertTrue(ndnf.check_delta())

        ndnf.set_delta_val([0.1, 0.2])
        self.assertFalse(ndnf.check_delta())

        ndnf_eo = NeuralDNFEO(4, 3, 2, 0.1)
        self.assertTrue(ndnf_eo.check_delta())

        ndnf_eo.set_delta_val([0.1, 0.2, 0.2])
        self.assertFalse(ndnf_eo.check_delta())
