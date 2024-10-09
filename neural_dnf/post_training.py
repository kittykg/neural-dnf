from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path
import sys
from typing import Any, Callable

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

from neural_dnf.neural_dnf import BaseNeuralDNF
from neural_dnf.semi_symbolic import SemiSymbolicLayerType


def prune_layer_weight(
    neural_dnf: BaseNeuralDNF,
    layer_type: SemiSymbolicLayerType,
    evaluation_function: Callable[..., dict[str, Any]],
    evaluation_function_args: dict[str, Any],
    comparison_function: Callable[[dict[str, Any], dict[str, Any]], bool],
    update_comparison_dict: bool = False,
    logging_fn: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """
    Prune a semi-symbolic layer of the neural DNF model.
    The comparison function would be used to decide whether a weight should be
    pruned. It should use the result dict from the evaluation function.
    If `update_comparison_dict` is True, the comparison dict would be updated
    after each pruning. Default is False.
    If `logging_fn` is provided, it would be called after each weight is
    pruned, with a dictionary of local variables as function argument.
    """
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = neural_dnf.conjunctions.weights.data.clone()
    else:
        curr_weight = neural_dnf.disjunctions.weights.data.clone()

    comparison_dict = evaluation_function(**evaluation_function_args)

    prune_count = 0

    flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))

    for i in range(flatten_weight_len):
        curr_weight_flatten = torch.reshape(curr_weight, (-1,))

        if curr_weight_flatten[i] == 0:
            continue

        mask = torch.ones(flatten_weight_len, device=curr_weight.device)
        mask[i] = 0
        mask = mask.reshape(curr_weight.shape)

        masked_weight = curr_weight * mask

        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            neural_dnf.conjunctions.weights.data = masked_weight
        else:
            neural_dnf.disjunctions.weights.data = masked_weight

        new_result_dict = evaluation_function(**evaluation_function_args)

        if comparison_function(comparison_dict, new_result_dict):
            prune_count += 1
            curr_weight *= mask

            if update_comparison_dict:
                comparison_dict = new_result_dict

            if logging_fn:
                logging_fn(
                    {
                        "index": i,
                        "prune_count": prune_count,
                        "comparison_dict": comparison_dict,
                        "new_result_dict": new_result_dict,
                        "layer_type": layer_type,
                    }
                )

    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        neural_dnf.conjunctions.weights.data = curr_weight
    else:
        neural_dnf.disjunctions.weights.data = curr_weight
    return prune_count


def remove_unused_conjunctions(neural_dnf: BaseNeuralDNF) -> int:
    """
    Remove any conjunctions connected to a pruned disjunction.
    This function should only be called after pruning the disjunctions.
    """
    disj_w = neural_dnf.disjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(disj_w.T):
        if torch.all(w == 0):
            # The conjunction is not used at all
            if torch.any(neural_dnf.conjunctions.weights.data[i, :] != 0):
                # Remove the conjunction
                neural_dnf.conjunctions.weights.data[i, :] = 0
                unused_count += 1

    return unused_count


def remove_disjunctions_when_empty_conjunctions(
    neural_dnf: BaseNeuralDNF,
) -> int:
    """
    If a conjunction has all 0 weights (no input atom is used), then this
    conjunction shouldn't be used in a rule.
    This function should only be called after pruning the conjunctions.
    """
    conj_w = neural_dnf.conjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # This conjunction should not be used
            for j in range(neural_dnf.disjunctions.weights.shape[0]):
                if neural_dnf.disjunctions.weights.data[j, i] != 0:
                    neural_dnf.disjunctions.weights.data[j, i] = 0
                    unused_count += 1

    return unused_count


def prune_neural_dnf(
    neural_dnf: BaseNeuralDNF,
    evaluation_function: Callable[..., dict[str, Any]],
    evaluation_function_args: dict[str, Any],
    comparison_function: Callable[[dict[str, Any], dict[str, Any]], bool],
    options: dict[str, Any] = {},
):
    """
    Prune the neural DNF model. This process consists of 5 steps:
    1. Prune disjunction (mandatory)
    2. Prune unused conjunctions (disabled for now)
        - If a conjunction is not used in any disjunctions, pruned the
        entire disjunct body
        (this step might be problematic for NeuralDNFMT since pruning
        conjunctions can affect the mutex-tanh out and cause weird behaviour
        in the model)
    3. Prune conjunctions (mandatory)
    4. Prune disjunctions that uses empty conjunctions (optional)
        - If a conjunction has no conjunct, no disjunctions should use it
    5. Prune disjunction again (optional)
    You can pass in an options dict to control the pruning process. The options
    are:
        - prune_with_update_comparison_dict: bool
            during the each pruning stage, the comparison dict will be updated
            after each pruning. Default is False
        - prune_logging_fn: Callable[[dict[str, Any]], None] | None
            a function that will be called after each weight is pruned. The
            function should take in a dict of local variables as argument.
            Default is None
        - eval_after_each_step: bool
            this will evaluate the model after each step
        - eval_result_keys: list[str]
            the key to use to get the result from the evaluation function. This
            field is required if `eval_after_each_step` is True
        - skip_prune_disj_with_empty_conj: bool
            this will skip step 4
        - skip_last_prune_disj: bool
            this will skip step 5
    This function return a dict containing the following keys:
        - disj_prune_count_1: int (mandatory)
            the number of disjunctions pruned in step 1
        - eval_result_1: Any (optional)
            the evaluation result after step 1
        - unused_conjunctions_2: int (mandatory)
            the number of unused conjunctions pruned in step 2
        - conj_prune_count_3: int (mandatory)
            the number of conjunctions pruned in step 3
        - eval_result_3: Any (optional)
            the evaluation result after step 3
        - prune_disj_with_empty_conj_count_4: int (optional)
            the number of disjunctions with empty conjunctions pruned in step 4
        - disj_prune_count_5: int (optional)
            the number of disjunctions pruned in step 5
        - eval_result_5: Any (optional)
            the evaluation result after step 5
    """
    prune_result_dict = dict()

    def intermediate_eval(stage_key: str) -> None:
        if options.get("eval_after_each_step", False) and options.get(
            "eval_result_keys", None
        ):
            assert isinstance(options["eval_result_keys"], list), ""
            prune_result_dict[stage_key] = {
                k: evaluation_function(**evaluation_function_args)[k]
                for k in options["eval_result_keys"]
            }

    prune_with_update_comparison_dict = options.get(
        "prune_with_update_comparison_dict", False
    )
    prune_logging_fn = options.get("prune_logging_fn", None)

    # 1. Prune disjunction
    prune_result_dict["disj_prune_count_1"] = prune_layer_weight(
        neural_dnf=neural_dnf,
        layer_type=SemiSymbolicLayerType.DISJUNCTION,
        evaluation_function=evaluation_function,
        evaluation_function_args=evaluation_function_args,
        comparison_function=comparison_function,
        update_comparison_dict=prune_with_update_comparison_dict,
        logging_fn=prune_logging_fn,
    )
    intermediate_eval("eval_result_1")

    # 2. Prune unused conjunctions
    prune_result_dict["unused_conjunctions_2"] = remove_unused_conjunctions(
        neural_dnf
    )
    intermediate_eval("eval_result_2")

    # 3. Prune conjunctions
    prune_result_dict["conj_prune_count_3"] = prune_layer_weight(
        neural_dnf=neural_dnf,
        layer_type=SemiSymbolicLayerType.CONJUNCTION,
        evaluation_function=evaluation_function,
        evaluation_function_args=evaluation_function_args,
        comparison_function=comparison_function,
        update_comparison_dict=prune_with_update_comparison_dict,
        logging_fn=prune_logging_fn,
    )
    intermediate_eval("eval_result_3")

    # 4. Prune disjunctions that uses empty conjunctions
    if not options.get("skip_prune_disj_with_empty_conj", False):
        prune_result_dict["prune_disj_with_empty_conj_count_4"] = (
            remove_disjunctions_when_empty_conjunctions(neural_dnf)
        )
        intermediate_eval("eval_result_4")

    # 5. Prune disjunction again
    if not options.get("skip_last_prune_disj", False):
        prune_result_dict["disj_prune_count_5"] = prune_layer_weight(
            neural_dnf,
            SemiSymbolicLayerType.DISJUNCTION,
            evaluation_function,
            evaluation_function_args,
            comparison_function,
        )
        intermediate_eval("eval_result_5")

    return prune_result_dict


def get_thresholding_upper_bound(
    neural_dnf: BaseNeuralDNF, buffer: float = 0.01
) -> float:
    """
    Get the upper bound of the absolute values of the weights for thresholding
    procedure during post-training processing. We add a small `buffer` (default
    0.01) to the true upper bound and rounded to 2 decimal place for the
    thresholding process.
    """
    conj_min = torch.min(neural_dnf.conjunctions.weights.data)
    conj_max = torch.max(neural_dnf.conjunctions.weights.data)
    disj_min = torch.min(neural_dnf.disjunctions.weights.data)
    disj_max = torch.max(neural_dnf.disjunctions.weights.data)

    return round(
        (
            torch.Tensor([conj_min, conj_max, disj_min, disj_max]).abs().max()
            + buffer
        ).item(),
        2,
    )


def apply_threshold(
    neural_dnf: BaseNeuralDNF,
    og_conj_weight: Tensor,
    og_disj_weight: Tensor,
    t_val: Tensor | float,
    const: float = 6.0,
) -> None:
    """
    Apply a threshold value on the weights. This is part of the thresholding
    procedure during post-training processing. Before calling this function
    please have a saved copy of the original weights. The saved copies are used
    to calculate the thresholded weights.
    """
    neural_dnf.conjunctions.weights.data = (
        (torch.abs(og_conj_weight) > t_val) * torch.sign(og_conj_weight) * const
    )
    neural_dnf.disjunctions.weights.data = (
        (torch.abs(og_disj_weight) > t_val) * torch.sign(og_disj_weight) * const
    )


def thresholding(
    neural_dnf: BaseNeuralDNF,
    evaluation_function: Callable[..., dict[str, Any]],
    evaluation_function_args: dict[str, Any],
    selection_function: Callable[[list[dict[str, Any]], Tensor], list[Tensor]],
    options: dict[str, Any] = {},
) -> float | list[Any]:
    """
    Apply threshold values on the neural DNF model and evaluate the model for
    each threshold value with `evaluation_function`. From all the threshold
    values, select the best threshold value(s). The selection method can be
    controlled `selection_function`.
    """
    og_conj_weight = neural_dnf.conjunctions.weights.data.clone()
    og_disj_weight = neural_dnf.disjunctions.weights.data.clone()

    threshold_upper_bound = get_thresholding_upper_bound(neural_dnf)
    t_vals = torch.arange(0, threshold_upper_bound, 0.01)

    result_dicts = []
    for v in t_vals:
        apply_threshold(neural_dnf, og_conj_weight, og_disj_weight, v)
        result_dicts.append(evaluation_function(**evaluation_function_args))

    if options.get("return_raw_result", False):
        return result_dicts

    possible_t_vals = selection_function(result_dicts, t_vals)

    t_val = (
        max(possible_t_vals)  # type: ignore
        if options.get("take_max", False)
        else min(possible_t_vals)  # type: ignore
    )

    apply_threshold(neural_dnf, og_conj_weight, og_disj_weight, t_val)

    return t_val


def extract_asp_rules(
    sd: dict, format_options: dict[str, str] = {}, return_as_dict: bool = False
) -> list[str] | dict[str, Any]:
    """
    Extract ASP rules from the saved dict of a neural DNF model.
    Use `format_options` to control the syntax of the rules. `format_options`
    contains the following keys:
        - `input_name`: default 'a'
        - `input_syntax`: 'PRED' or 'FO, default 'PRED'
        - `conjunction_name`: default 'conj'
        - `conjunction_syntax`: 'PRED' or 'FO, default 'PRED'
        - `disjunction_name`: default 'disj'
        - `disjunction_syntax`: 'PRED' or 'FO, default 'PRED'
    If `*_syntax` is 'PRED', then the name would be like `a_1`.
    If `*_syntax` is 'FO', then the name would be like `a(1)`.
    """

    @dataclass
    class Atom:
        id: int
        positive: bool
        type: str  # possible values: "input", "conjunction", "disjunction_head"

    # 1. Extract the skeleton
    conjunction_map: dict[int, list[Atom]] = dict()
    disjunction_map: dict[int, list[Atom]] = dict()
    not_covered_classes: list[int] = []
    relevant_input: set[int] = set()

    #       Get all conjunctions
    conj_w = sd["conjunctions.weights"]
    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # No conjunction is applied here
            continue

        conjuncts: list[Atom] = []
        for j, v in enumerate(w):
            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append(Atom(j, False, "input"))
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append(Atom(j, True, "input"))

        conjunction_map[i] = conjuncts

    #       Get the DNF for each class
    disj_w = sd["disjunctions.weights"]
    for i, w in enumerate(disj_w):
        if torch.all(w == 0):
            # No DNF for class i
            not_covered_classes.append(i)
            continue

        disjuncts: list[Atom] = []
        for j, v in enumerate(w):
            if v < 0 and j in conjunction_map:
                # Negative weight, negate the existing conjunction
                disjuncts.append(Atom(j, False, "conjunction"))
                for a in conjunction_map[j]:
                    relevant_input.add(a.id)
            elif v > 0 and j in conjunction_map:
                # Postivie weight, add normal conjunction
                disjuncts.append(Atom(j, True, "conjunction"))
                for a in conjunction_map[j]:
                    relevant_input.add(a.id)

        disjunction_map[i] = disjuncts

    # 2. Convert the skeleton to rules
    input_name = format_options.get("input_name", "a")
    input_syntax = format_options.get("input_syntax", "PRED")
    conjunction_name = format_options.get("conjunction_name", "conj")
    conjunction_syntax = format_options.get("conjunction_syntax", "PRED")
    disjunction_name = format_options.get("disjunction_name", "disj")
    disjunction_syntax = format_options.get("disjunction_syntax", "PRED")

    def get_atom_name(
        atom: Atom, atom_name: str, syntax: str, is_head: bool = False
    ):
        an = "" if (atom.positive or is_head) else "not "
        if syntax == "PRED":
            an += f"{atom_name}_{atom.id}"
        else:
            an += f"{atom_name}({atom.id})"
        return an

    output_rules = []
    for disj, conjs in disjunction_map.items():
        for conj in conjs:
            head = get_atom_name(
                Atom(disj, True, "disjunction_head"),
                disjunction_name,
                disjunction_syntax,
                True,
            )
            body = get_atom_name(conj, conjunction_name, conjunction_syntax)
            output_rules.append(f"{head} :- {body}.")

    for conj, inputs in conjunction_map.items():
        head = get_atom_name(
            Atom(conj, True, "conjunction_head"),
            conjunction_name,
            conjunction_syntax,
            True,
        )
        body = ", ".join(
            [get_atom_name(i, input_name, input_syntax) for i in inputs]
        )
        output_rules.append(f"{head} :- {body}.")

    if return_as_dict:
        return {
            "rules": output_rules,
            "not_covered_classes": not_covered_classes,
            "relevant_input": relevant_input,
            "used_conjunctions": list(conjunction_map.keys()),
            "used_disjunctions": list(disjunction_map.keys()),
            "total_number_classes": len(disj_w),
        }
    return output_rules
