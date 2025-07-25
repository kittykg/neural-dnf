from pathlib import Path
import sys
from typing import overload

import torch
from torch import Tensor

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from neural_dnf.semi_symbolic import *


class BaseNeuralDNF(torch.nn.Module):
    """
    The base class of neural DNF modules. It contains the common functions.
    This should not be directly used / imported, but for type hinting purposes.
    The weight masks are used for ensuring that the weights of the semi-symbolic
    layers are not updated during finetuning. If finetuning is not required, the
    weight masks can be ignored.
    """

    conjunction_semi_symbolic_layer_type: type[BaseSemiSymbolic] = SemiSymbolic
    disjunction_semi_symbolic_layer_type: type[BaseSemiSymbolic] = SemiSymbolic

    conjunctions: BaseSemiSymbolic
    disjunctions: BaseSemiSymbolic

    conj_weight_mask: Tensor
    disj_weight_mask: Tensor

    layer_names: list[str] = ["conjunctions", "disjunctions"]

    def __init__(
        self,
        n_in: int,
        n_conjunctions: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal",
    ) -> None:
        super(BaseNeuralDNF, self).__init__()

        self.conjunctions = self.conjunction_semi_symbolic_layer_type(
            in_features=n_in,  # P
            out_features=n_conjunctions,  # Q
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight: Q x P

        self.disjunctions = self.disjunction_semi_symbolic_layer_type(
            in_features=n_conjunctions,  # Q
            out_features=n_out,  # R
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight: R x Q

        self.conj_weight_mask = torch.ones(self.conjunctions.weights.data.shape)
        self.disj_weight_mask = torch.ones(self.disjunctions.weights.data.shape)

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        conj = self.conjunctions(input)
        # conj: N x Q
        conj = torch.tanh(conj)
        # conj: N x Q
        disj = self.disjunctions(conj)
        # disj: N x R

        return disj

    def get_conjunction(self, x: Tensor) -> Tensor:
        """
        Return the activation of the conjunctive layer
        """
        return torch.tanh(self.conjunctions(x))

    def get_delta_val(self) -> list[float]:
        """
        Return the delta values of the semi-symbolic layers, in the order of
        [conjunctions.delta, disjunctions.delta].
        """
        return [self.conjunctions.delta, self.disjunctions.delta]

    @overload
    def set_delta_val(self, delta: list[float]) -> None: ...

    @overload
    def set_delta_val(self, delta: float) -> None: ...

    def set_delta_val(self, delta) -> None:
        """
        Set the delta values of the semi-symbolic layers. If a single delta
        value is provided, it will be set for both layers. Also accepts a list
        of delta values for each layer.
        """
        if isinstance(delta, list):
            assert len(delta) == 2, "Delta should be a list of length 2."
            conj_delta, disj_delta = delta
            self.conjunctions.delta = conj_delta
            self.disjunctions.delta = disj_delta
        else:
            self.conjunctions.delta = delta
            self.disjunctions.delta = delta

    def update_weight_wrt_mask(self) -> None:
        """
        This should be called during finetuning to ensure that the pruned
        weights are not updated.
        """
        self.conjunctions.weights.data *= self.conj_weight_mask
        self.disjunctions.weights.data *= self.disj_weight_mask

    def check_delta(self) -> bool:
        """
        Check if the delta values of the semi-symbolic layers are the same.
        """
        return self.conjunctions.delta == self.disjunctions.delta


class NeuralDNF(BaseNeuralDNF):
    """
    A neural DNF module is a neural network that represents a DNF formula.
    It contains 2 semi-symbolic layer, a conjunctive one followed by a
    disjunctive one.
    The logical constraint in the semi-symbolic layer is controlled by the delta
    parameter. The delta parameter is a float between 0 and 1. The closer it is
    to 1, the more logically constrained the layer is. The delta parameter is
    expected to be the same for both layers.
    The input of the module is expected to be in the range of [-1, 1], where -1
    represents false and 1 represents true.
    When the weights of the semi-symbolic layers are set to 6 or -6, the tanh-ed
    output would saturate to 1 or -1.
    """

    def __init__(
        self,
        n_in: int,
        n_conjunctions: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal",
    ) -> None:
        super().__init__(n_in, n_conjunctions, n_out, delta, weight_init_type)


class NeuralDNFEO(BaseNeuralDNF):
    """
    A neural DNF-EO module extends the neural DNF module by adding an extra
    conjunctive constraint layer. The conjunctive constraint mimics the
    constraint of exactly one (EO) class can be true at a time. The constraint
    is usually enforced in logic as `:- class(X), class(Y), X < Y.` We implement
    it as `class(I) :- not class(J) for all J != I`.
    The constraint layer is frozen and untrainable. It is also removed during
    inference.
    The delta parameter is expected to be the same for all layers.
    """

    eo_constraint: SemiSymbolic

    layer_names: list[str] = ["conjunctions", "disjunctions", "eo_constraint"]

    def __init__(
        self,
        n_in: int,
        n_conjunctions: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal",
    ) -> None:
        super().__init__(n_in, n_conjunctions, n_out, delta, weight_init_type)

        self.eo_constraint = SemiSymbolic(
            in_features=n_out,  # R
            out_features=n_out,  # R
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
        )  # weight: R x R

        self.eo_constraint.weights.data.fill_(-6)
        self.eo_constraint.weights.data.fill_diagonal_(0)
        self.eo_constraint.requires_grad_(False)

    def forward(self, input: Tensor) -> Tensor:
        disj = super().forward(input)
        # disj: N x R
        disj = torch.tanh(disj)
        # disj: N x R
        out = self.eo_constraint(disj)
        # out: N x R

        return out

    def get_plain_output(self, x: Tensor) -> Tensor:
        return super().forward(x)

    def get_delta_val(self) -> list[float]:
        conj_delta = self.conjunctions.delta
        disj_delta = self.disjunctions.delta
        eo_delta = self.eo_constraint.delta
        return [conj_delta, disj_delta, eo_delta]

    def set_delta_val(self, delta) -> None:
        if isinstance(delta, list):
            assert len(delta) == 3, "Delta should be a list of length 3."
            conj_delta, disj_delta, eo_delta = delta
            self.conjunctions.delta = conj_delta
            self.disjunctions.delta = disj_delta
            self.eo_constraint.delta = eo_delta
        else:
            self.conjunctions.delta = delta
            self.disjunctions.delta = delta
            self.eo_constraint.delta = delta

    def check_delta(self) -> bool:
        return (
            self.conjunctions.delta
            == self.disjunctions.delta
            == self.eo_constraint.delta
        )

    def to_ndnf(self) -> NeuralDNF:
        """
        Convert the NeuralDNFEO module to a NeuralDNF module by removing the
        EO constraint layer.
        """
        assert self.check_delta(), "Delta values are not the same."

        ndnf = NeuralDNF(
            n_in=self.conjunctions.in_features,
            n_conjunctions=self.conjunctions.out_features,
            n_out=self.disjunctions.out_features,
            delta=self.get_delta_val()[0],
        )
        ndnf.conjunctions.weights.data = self.conjunctions.weights.data.clone()
        ndnf.disjunctions.weights.data = self.disjunctions.weights.data.clone()
        return ndnf


class BaseNeuralDNFMutexTanh(BaseNeuralDNF):
    """
    A base neural DNF mutex tanh module. It is constructed the same way as a
    neural DNF module, but it's designed for tasks that requires mutual
    exclusivity on disjunction (in symbolic and probabilistic terms*).

    The base class will only enforce the mutual exclusivity on the disjunctive
    layer. This class should not be directly used / imported, but used for type
    hinting purposes.

    * Mutual exclusivity in symbolic terms means that the if a set of ASP rules
    where each class is represented by a rule head respectively, only one class
    should present in the answer set. Mutual exclusivity in probabilistic terms
    means that the sum of the probabilities of all classes is 1.
    """

    disjunction_semi_symbolic_layer_type: type[BaseSemiSymbolic] = (
        SemiSymbolicMutexTanh
    )
    disjunctions: SemiSymbolicMutexTanh

    def __init__(
        self,
        n_in: int,
        n_conjunctions: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal",
        c: float | None = None,
    ) -> None:
        super().__init__(n_in, n_conjunctions, n_out, delta, weight_init_type)
        self.c = c

    def get_raw_output(self, x: Tensor) -> Tensor:
        """
        Return the raw output of the neural DNF module without the mutex tanh
        applied to the disjunctive layer.
        """
        raise NotImplementedError

    def get_all_forms(self, x: Tensor) -> dict[str, dict[str, Tensor]]:
        """
        For each layer, return the raw output, tanh output and mutex-tanh output
        if applicable.
        """
        raise NotImplementedError

    def to_ndnf(self) -> NeuralDNF:
        """
        Convert the mutex tanh module to a normal NeuralDNF module by converting
        the mutex tanh layers to normal tanh layers.
        """
        assert self.check_delta(), "Delta values are not the same."
        ndnf = NeuralDNF(
            n_in=self.conjunctions.in_features,
            n_conjunctions=self.conjunctions.out_features,
            n_out=self.disjunctions.out_features,
            delta=self.get_delta_val()[0],
        )
        ndnf.conjunctions.weights.data = self.conjunctions.weights.data.clone()
        ndnf.disjunctions.weights.data = self.disjunctions.weights.data.clone()
        return ndnf


class NeuralDNFMutexTanh(BaseNeuralDNFMutexTanh):
    """
    A neural DNF mutex tanh module, designed for tasks that requires mutual
    exclusivity on disjunction (in symbolic and probabilistic terms*).
    This module enforces the mutual exclusivity only on the disjunctive layer.

    * Mutual exclusivity in symbolic terms means that the if a set of ASP rules
    where each class is represented by a rule head respectively, only one class
    should present in the answer set. Mutual exclusivity in probabilistic terms
    means that the sum of the probabilities of all classes is 1.
    """

    def get_raw_output(self, x: Tensor) -> Tensor:
        conj = self.conjunctions(x)
        # conj: N x Q
        conj = torch.tanh(conj)
        # conj: N x Q
        disj = self.disjunctions.get_raw_output(conj)
        # disj: N x R
        return disj

    def get_all_forms(self, x: Tensor) -> dict[str, dict[str, Tensor]]:
        conj_raw = self.conjunctions(x)
        # conj_raw: N x Q
        tanh_conj = torch.tanh(conj_raw)

        disj_raw = self.disjunctions.get_raw_output(tanh_conj)
        # disj_raw: N x R
        disj_tanh = torch.tanh(disj_raw)
        disj_mt = self.disjunctions(tanh_conj)

        return {
            "conjunction": {
                "raw": conj_raw,
                "tanh": tanh_conj,
            },
            "disjunction": {
                "raw": disj_raw,
                "tanh": disj_tanh,
                "mutex_tanh": disj_mt,
            },
        }


class NeuralDNFFullMutexTanh(BaseNeuralDNFMutexTanh):
    """
    A neural DNF full mutex tanh module, designed for tasks that requires mutual
    exclusivity on disjunction (in symbolic and probabilistic terms*).
    This module enforces the mutual exclusivity on the both the conjunctive and
    disjunctive layers.

    * Mutual exclusivity in symbolic terms means that the if a set of ASP rules
    where each class is represented by a rule head respectively, only one class
    should present in the answer set. Mutual exclusivity in probabilistic terms
    means that the sum of the probabilities of all classes is 1.
    """

    conjunction_semi_symbolic_layer_type: type[BaseSemiSymbolic] = (
        SemiSymbolicMutexTanh
    )
    conjunctions: SemiSymbolicMutexTanh

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        conj = self.conjunctions(input)
        # conj: N x Q
        # conj already in (-1, 1) range because of mutex tanh
        disj = self.disjunctions(conj)
        # disj: N x R
        # disj also in (-1, 1) range because of mutex tanh

        return disj

    def get_raw_output(self, x: Tensor) -> Tensor:
        conj = self.conjunctions(x)
        # conj: N x Q
        disj = self.disjunctions.get_raw_output(conj)
        # disj: N x R
        return disj

    def get_all_forms(self, x: Tensor) -> dict[str, dict[str, Tensor]]:
        # All tensors below are: N x Q
        raw = self.conjunctions.get_raw_output(x)
        mt_conj = self.conjunctions(x)
        tanh_conj = torch.tanh(raw)

        # All tensors below are: N x R
        raw = self.disjunctions.get_raw_output(mt_conj)
        mt_disj = self.disjunctions(mt_conj)
        tanh_disj = torch.tanh(raw)

        return {
            "conjunction": {
                "raw": raw,
                "tanh": tanh_conj,
                "mutex_tanh": mt_conj,
            },
            "disjunction": {
                "raw": raw,
                "tanh": tanh_disj,
                "mutex_tanh": mt_disj,
            },
        }
