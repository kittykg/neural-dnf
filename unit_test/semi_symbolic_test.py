import unittest

import torch

from neural_dnf.semi_symbolic import (
    SemiSymbolic,
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


if __name__ == "__main__":
    unittest.main()
