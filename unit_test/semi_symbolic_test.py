import unittest

import torch

from neural_dnf.semi_symbolic import (
    SemiSymbolic,
    SemiSymbolicMutexTanh,
    SemiSymbolicLayerType,
)

RTOL = 1e-4
ATOL = 1e-4


class TestSemiSymbolicLayer(unittest.TestCase):
    def test_conjunctive_semi_symbolic_layer(self):
        layer = SemiSymbolic(3, 5, SemiSymbolicLayerType.CONJUNCTION, 1.0)
        # Create a layer that mimics 5 conjunctions over 3 predicates a, b, c:
        # conj_0 = a ^ b
        # conj_1 = a ^ c
        # conj_2 = -a ^ c
        # conj_3 = b ^ -c
        # conj_4 = c
        layer.weights.data = torch.Tensor(
            [
                [6, 6, 0],
                [6, 0, 6],
                [-6, 0, 6],
                [0, 6, -6],
                [0, 0, 6],
            ]
        )

        # The bias calculation for each conjunction is as follows:
        # conj_0_bias = -6
        # conj_1_bias = -6
        # conj_2_bias = -6
        # conj_3_bias = -6
        # conj_4_bias = 0
        torch.testing.assert_close(
            layer.bias_calculation(), torch.Tensor([-6, -6, -6, -6, 0])
        )

        with torch.no_grad():
            # Input_0: -a, b, c
            # Expected: false, false, true, false, true
            input_0 = torch.Tensor([[-1, 1, 1]])
            out_0 = torch.tanh(layer(input_0))
            expected_0 = torch.Tensor([[-1, -1, 1, -1, 1]])
            torch.testing.assert_close(out_0, expected_0, rtol=RTOL, atol=ATOL)

            # Input_1: a, -b, c
            # Expected: false, true, false, false, true
            input_1 = torch.Tensor([[1, -1, 1]])
            out_1 = torch.tanh(layer(input_1))
            expected_1 = torch.Tensor([[-1, 1, -1, -1, 1]])
            torch.testing.assert_close(out_1, expected_1, rtol=RTOL, atol=ATOL)

            # Input_2: -a, -b, -c
            # Expected: false, false, false, false, false
            input_2 = torch.Tensor([[-1, -1, -1]])
            out_2 = torch.tanh(layer(input_2))
            expected_2 = torch.Tensor([[-1, -1, -1, -1, -1]])
            torch.testing.assert_close(out_2, expected_2, rtol=RTOL, atol=ATOL)

            # Input_3: a, b, c
            # Expected: true, true, false, false, true
            input_3 = torch.Tensor([[1, 1, 1]])
            out_3 = torch.tanh(layer(input_3))
            expected_3 = torch.Tensor([[1, 1, -1, -1, 1]])
            torch.testing.assert_close(out_3, expected_3, rtol=RTOL, atol=ATOL)

    def test_disjunctive_semi_symbolic_layer(self):
        layer = SemiSymbolic(5, 4, SemiSymbolicLayerType.DISJUNCTION, 1.0)
        # Create a layer that mimics 4 disjunctions over 5 predicates
        # a, b, c, d, e:
        # disj_0 = a v b
        # disj_1 = c v d
        # disj_2 = -a
        # disj_3 = -b V -d V e
        layer.weights.data = torch.Tensor(
            [
                [6, 6, 0, 0, 0],
                [0, 0, 6, 6, 0],
                [-6, 0, 0, 0, 0],
                [0, -6, 0, -6, 6],
            ]
        )

        # The bias calculation for each disjunction is as follows:
        # disj_0_bias = 6
        # disj_1_bias = 6
        # disj_2_bias = 0
        # disj_3_bias = 12
        torch.testing.assert_close(
            layer.bias_calculation(), torch.Tensor([6, 6, 0, 12])
        )

        with torch.no_grad():
            # Input_0: a, -b, -c, -d, -e
            # Expected: true, false, false, true
            input_0 = torch.Tensor([[1, -1, -1, -1, -1]])
            out_0 = torch.tanh(layer(input_0))
            expected_0 = torch.Tensor([[1, -1, -1, 1]])
            torch.testing.assert_close(out_0, expected_0, rtol=RTOL, atol=ATOL)

            # Input_1: -a, b, c, d, -e
            # Expected: true, true, true, false
            input_1 = torch.Tensor([[-1, 1, 1, 1, -1]])
            out_1 = torch.tanh(layer(input_1))
            expected_1 = torch.Tensor([[1, 1, 1, -1]])
            torch.testing.assert_close(out_1, expected_1, rtol=RTOL, atol=ATOL)

            # Input_2: -a, -b, -c, -d, -e
            # Expected: false, false, true, true
            input_2 = torch.Tensor([[-1, -1, -1, -1, -1]])
            out_2 = torch.tanh(layer(input_2))
            expected_2 = torch.Tensor([[-1, -1, 1, 1]])
            torch.testing.assert_close(out_2, expected_2, rtol=RTOL, atol=ATOL)

            # Input_3: -a, b, -c, d, e
            # Expected: true, true, true, true
            input_3 = torch.Tensor([[-1, 1, -1, 1, 1]])
            out_3 = torch.tanh(layer(input_3))
            expected_3 = torch.Tensor([[1, 1, 1, 1]])
            torch.testing.assert_close(out_3, expected_3, rtol=RTOL, atol=ATOL)

    def test_disjunctive_semi_symbolic_mutex_tanh_layer(self):
        layer = SemiSymbolicMutexTanh(
            5, 4, SemiSymbolicLayerType.DISJUNCTION, 1.0
        )
        # Create a layer that mimics 4 disjunctions over 5 predicates
        # a, b, c, d, e. If by translating ±6 as present in the rule, the layer
        # represent the bivalent logic as:
        # disj_0 = a v b
        # disj_1 = c v d
        # disj_2 = -a
        # disj_3 = e
        layer.weights.data = torch.Tensor(
            [
                [6, 6, 0, 0, 0],
                [0, 0, 6, 6, 0],
                [-6, 0, 0, 0, 0],
                [0, 0, 0, 0, 6],
            ]
        )

        # The bias calculation for each disjunction is as follows:
        # disj_0_bias = 6
        # disj_1_bias = 6
        # disj_2_bias = 0
        # disj_3_bias = 0
        torch.testing.assert_close(
            layer.bias_calculation(), torch.Tensor([6, 6, 0, 0])
        )

        with torch.no_grad():
            # Input_0: a, -b, -c, -d, -e
            input_0 = torch.Tensor([[1, -1, -1, -1, -1]])
            raw_logits = layer.get_raw_output(input_0)
            mutex_tanh_out = layer(input_0)
            probs = (mutex_tanh_out + 1) / 2
            torch.testing.assert_close(torch.sum(probs), torch.tensor(1.0))
            # Bivalent interpretation: true, false, false, false
            torch.testing.assert_close(
                torch.tanh(raw_logits),
                torch.Tensor([[1, -1, -1, -1]]),
                rtol=RTOL,
                atol=ATOL,
            )

            # Input_1: -a, b, c, d, -e
            input_1 = torch.Tensor([[-1, 1, 1, 1, -1]])
            raw_logits = layer.get_raw_output(input_1)
            mutex_tanh_out = layer(input_1)
            probs = (mutex_tanh_out + 1) / 2
            torch.testing.assert_close(torch.sum(probs), torch.tensor(1.0))
            # Bivalent interpretation: true, true, true, false
            torch.testing.assert_close(
                torch.tanh(raw_logits),
                torch.Tensor([[1, 1, 1, -1]]),
                rtol=RTOL,
                atol=ATOL,
            )

            # Input_2: -a, -b, -c, -d, -e
            input_2 = torch.Tensor([[-1, -1, -1, -1, -1]])
            raw_logits = layer.get_raw_output(input_2)
            mutex_tanh_out = layer(input_2)
            probs = (mutex_tanh_out + 1) / 2
            torch.testing.assert_close(torch.sum(probs), torch.tensor(1.0))
            # Bivalent interpretation: false, false, true, false
            torch.testing.assert_close(
                torch.tanh(raw_logits),
                torch.Tensor([[-1, -1, 1, -1]]),
                rtol=RTOL,
                atol=ATOL,
            )

            # Input_3: -a, b, -c, d, e
            input_3 = torch.Tensor([[-1, 1, -1, 1, 1]])
            raw_logits = layer.get_raw_output(input_3)
            mutex_tanh_out = layer(input_3)
            probs = (mutex_tanh_out + 1) / 2
            torch.testing.assert_close(torch.sum(probs), torch.tensor(1.0))
            # Bivalent interpretation: true, true, true, true
            torch.testing.assert_close(
                torch.tanh(raw_logits),
                torch.Tensor([[1, 1, 1, 1]]),
                rtol=RTOL,
                atol=ATOL,
            )


if __name__ == "__main__":
    unittest.main()
