from pathlib import Path
import sys
from typing import overload, Callable

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from neural_dnf.neural_dnf import *
from neural_dnf.semi_symbolic import *

SEMI_SYMBOLIC_BASE_NAME = "SemiSymbolic"
NEURAL_DNF_BASE_NAME = "NeuralDNF"


class DeltaDelayedDecayScheduler:
    """
    To construct a decay scheduler, the following parameters are needed:
    - initial_delta: the initial delta value
    - delta_decay_delay: the number of steps before the decay starts
    - delta_decay_steps: the number of steps before the decay rate is applied
    - delta_decay_rate: the decay rate
    - target_module_type: the class name of the target module. Can be either a
        full module that inherits `BaseNeuralDNF` module, or a layer that
        inherits `SemiSymbolic`. Use module.__class__.__name__ to get the class
        name of the module.
    """

    initial_delta: float
    delta_decay_delay: int
    delta_decay_steps: int
    delta_decay_rate: float
    target_module_type: str  #  the class name of the target neural DNF model

    internal_step_counters: list[int]
    internal_step_counters_len: int

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
        target_module_type: str,
    ):
        self.initial_delta = initial_delta
        self.delta_decay_delay = delta_decay_delay
        self.delta_decay_steps = delta_decay_steps
        self.delta_decay_rate = delta_decay_rate

        assert (
            SEMI_SYMBOLIC_BASE_NAME in target_module_type
            or NEURAL_DNF_BASE_NAME in target_module_type
        ), (
            "Invalid target module type. The target module type should be "
            "either a full module that inherits `BaseNeuralDNF` module, or a "
            "layer that inherits `SemiSymbolic`."
        )

        self.target_module_type = target_module_type

        self._setup_internal_counter()

    def step(
        self, module: BaseNeuralDNF | BaseSemiSymbolic
    ) -> dict[str, list[float]]:
        """
        Calculate the new delta value and set it in the model.
        Returns a dictionary containing the new delta values and the old delta
        values. The keys are "new_delta_vals" and "old_delta_vals".
        """

        assert self._check_type_matching_at_step(
            module
        ), "Type mismatch between the module to change and this scheduler's "
        "target_module_type."

        if isinstance(module, BaseSemiSymbolic):
            old_delta = module.delta
            if old_delta == 1.0:
                # If the delta is already 1 then don't need to calculate for new
                # delta
                new_delta = old_delta
            else:
                new_delta = self._calculate_new_delta(
                    self.internal_step_counters[0]
                )
                new_delta = 1 if new_delta > 1 else new_delta
            module.delta = new_delta
            self.internal_step_counters[0] += 1
            return {
                "new_delta_vals": [new_delta],
                "old_delta_vals": [old_delta],
            }

        old_delta_vals = module.get_delta_val()
        assert len(old_delta_vals) == self.internal_step_counters_len, (
            "The length of the internal step counter "
            f"({self.internal_step_counters_len}) and the length of the "
            f"delta values in the model ({len(old_delta_vals)}) should be "
            "the same."
        )

        new_delta_vals = []
        for i in range(self.internal_step_counters_len):
            old_delta = old_delta_vals[i]
            if old_delta == 1.0:
                # If the delta is already 1 then don't need to calculate for new
                # delta
                # return old_delta_vals, old_delta_vals
                new_delta = old_delta
            else:
                new_delta = self._calculate_new_delta(
                    self.internal_step_counters[i]
                )
                new_delta = 1 if new_delta > 1 else new_delta
            new_delta_vals.append(new_delta)
            self.internal_step_counters[i] += 1

        module.set_delta_val(new_delta_vals)
        return {
            "new_delta_vals": new_delta_vals,
            "old_delta_vals": old_delta_vals,
        }

    def _calculate_new_delta(self, step: int) -> float:
        raise NotImplementedError

    def _setup_internal_counter(self) -> None:
        if "SemiSymbolic" in self.target_module_type:
            # SemiSymbolic has only 1 layer
            self.internal_step_counters = [0]
        elif self.target_module_type == NeuralDNFEO.__name__:
            # 3 layers in NeuralDNFEO, so 3 counters for each layer
            self.internal_step_counters = [0, 0, 0]
        else:
            # The rest of the BaseNeuralDNF models have 2 layers
            self.internal_step_counters = [0, 0]

        self.internal_step_counters_len = len(self.internal_step_counters)

    def _check_type_matching_at_step(
        self, module: BaseNeuralDNF | BaseSemiSymbolic
    ) -> bool:
        return module.__class__.__name__ == self.target_module_type

    def reset(self) -> None:
        self.internal_step_counters = [0] * self.internal_step_counters_len

    @overload
    def set_counters(self, c: list[float]) -> None: ...

    @overload
    def set_counters(self, c: float) -> None: ...

    def set_counters(self, c) -> None:
        """
        Set the internal step counters to the input counter(s) (either a list
        of counters or a single counter).
        This is useful when training is resumed from a checkpoint.
        """
        if isinstance(c, list):
            assert len(c) == self.internal_step_counters_len, (
                "The length of the input counters "
                f"({len(c)}) and the length of the internal "
                f"counters ({self.internal_step_counters_len}) should be "
                "the same."
            )
            self.internal_step_counters = c
        else:
            self.internal_step_counters = [c] * self.internal_step_counters_len


class DeltaDelayedExponentialDecayScheduler(DeltaDelayedDecayScheduler):
    """
    This scheduler calculates the new delta value by multiplying the initial
    delta value with the decay rate raised to the power of the number of steps
    divided by `delta_decay_steps` and add 1. The new delta value is then
    clamped to 1 if it's greater than 1.
    The `delta_decay_rate` should be a value that's greater than 1 so that the
    delta value will goes up to 1.

    To construct a decay scheduler, the following parameters are needed:
    - initial_delta: the initial delta value
    - delta_decay_delay: the number of steps before the decay starts
    - delta_decay_steps: the number of steps before the decay rate is applied
    - delta_decay_rate: the decay rate
    - target_module_type: the class name of the target module. Can be either a
        full module that inherits `BaseNeuralDNF` module, or a layer that
        inherits `SemiSymbolic`. Use module.__class__.__name__ to get the class
        name of the module.
    """

    def _calculate_new_delta(self, step: int) -> float:
        # `step` should be 0-indexed
        if step < self.delta_decay_delay:
            return self.initial_delta
        step_diff = step - self.delta_decay_delay
        new_delta_val = self.initial_delta * (
            self.delta_decay_rate ** (step_diff // self.delta_decay_steps + 1)
        )
        return new_delta_val


class DeltaDelayedOffsetExponentialDecayScheduler(DeltaDelayedDecayScheduler):
    """
    This scheduler is the same as DeltaDelayedExponentialDecayScheduler, except
    that the delta start to be 0 and the after `delta_decay_delay` steps, the
    delta is then set to be the offset initial delta.

    To construct a decay scheduler, the following parameters are needed:
    - initial_delta: the initial delta value, usually 0
    - offset_initial_delta: the initial delta value after the offset
    - delta_decay_delay: the number of steps before the decay starts
    - delta_decay_steps: the number of steps before the decay rate is applied
    - delta_decay_rate: the decay rate
    - target_module_type: the class name of the target module. Can be either a
        full module that inherits `BaseNeuralDNF` module, or a layer that
        inherits `SemiSymbolic`. Use module.__class__.__name__ to get the class
        name of the module.
    """

    initial_delta: float
    offset_initial_delta: float

    def __init__(
        self,
        initial_delta: float,
        offset_initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
        target_module_type: str,
    ):
        super().__init__(
            initial_delta,
            delta_decay_delay,
            delta_decay_steps,
            delta_decay_rate,
            target_module_type,
        )
        self.offset_initial_delta = offset_initial_delta

    def _calculate_new_delta(self, step: int) -> float:
        # `step` should be 0-indexed
        if step < self.delta_decay_delay:
            return 0.0
        step_diff = step - self.delta_decay_delay
        # We use the offset initial delta to calculate the new delta value
        new_delta_val = self.offset_initial_delta * (
            self.delta_decay_rate ** (step_diff // self.delta_decay_steps + 1)
        )
        return new_delta_val


class DeltaDelayedLinearDecayScheduler(DeltaDelayedDecayScheduler):
    """
    This scheduler's `delta_decay_rate` should be a value that's between 0 and 1
    and after `delta_decay_delay` steps, the delta is then set to be the initial
    delta plus the decay rate for every `delta_decay_steps` steps.

    To construct a decay scheduler, the following parameters are needed:
    - initial_delta: the initial delta value
    - delta_decay_delay: the number of steps before the decay starts
    - delta_decay_steps: the number of steps before the decay rate is applied
    - delta_decay_rate: the decay rate
    - target_module_type: the class name of the target module. Can be either a
        full module that inherits `BaseNeuralDNF` module, or a layer that
        inherits `SemiSymbolic`. Use module.__class__.__name__ to get the class
        name of the module.
    """

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
        target_module_type: str,
    ):
        super().__init__(
            initial_delta,
            delta_decay_delay,
            delta_decay_steps,
            delta_decay_rate,
            target_module_type,
        )
        assert (
            0 < delta_decay_rate < 1
        ), "Decay rate should be between 0 and 1 for the linear scheduler."

    def _calculate_new_delta(self, step: int) -> float:
        # `step` should be 0-indexed
        if step < self.delta_decay_delay:
            return self.initial_delta
        step_diff = step - self.delta_decay_delay
        new_delta_val = (
            self.initial_delta
            + (step_diff // self.delta_decay_steps) * self.delta_decay_rate
        )
        return new_delta_val


class DeltaDelayedMonotonicFunctionScheduler(DeltaDelayedDecayScheduler):
    """
    This scheduler is uses f(t) = g(t) / g(1), with 2 options of g(t):
    1. g(t) = \frac{t^2}{2} - \frac{t^3}{3}
    2. g(t) = \frac{t^5}{5} - \frac{t^4}{2} + \frac{t^3}{3}
    t = 0 if step < delta_decay_delay, and when step >= delta_decay_delay,
    t = max(1, (step - delta_decay_delay) // delta_decay_steps / delta_decay_rate)

    To construct a decay scheduler, the following parameters are needed:
    - initial_delta: manually set to 0.0, not used
    - delta_decay_delay: the number of steps before the decay starts
    - delta_decay_steps: the number of steps before the decay rate is applied
    - delta_decay_rate: the number to scale the t value. The larger the number,
        the slower the delta value will increase. Should be greater than 1.
    - target_module_type: the class name of the target module. Can be either a
        full module that inherits `BaseNeuralDNF` module, or a layer that
        inherits `SemiSymbolic`. Use module.__class__.__name__ to get the class
        name of the module.
    """

    use_first_option: bool
    func_g: Callable[[float], float]

    def __init__(
        self,
        initial_delta: float,  # not used
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
        target_module_type: str,
        use_first_option: bool = True,
    ):
        super().__init__(
            0.0,
            delta_decay_delay,
            delta_decay_steps,
            delta_decay_rate,
            target_module_type,
        )
        assert self.delta_decay_rate > 1, (
            "Decay rate should be greater than 1 for the monotonic function "
            "scheduler."
        )
        self.use_first_option = use_first_option
        self.func_g = lambda t: (
            (t**2) / 2 - (t**3) / 3
            if use_first_option
            else (t**5) / 5 - (t**4) / 2 + (t**3) / 3
        )

    def _calculate_new_delta(self, step: int) -> float:
        # `step` should be 0-indexed
        if step < self.delta_decay_delay:
            return 0.0

        step_diff = (step - self.delta_decay_delay) // self.delta_decay_steps
        t = step_diff / self.delta_decay_rate
        if t > 1:
            t = 1

        # Calculate the new delta value using the monotonic function
        new_delta_val = self.func_g(t) / self.func_g(1)
        return new_delta_val


class DeltaDelayedMonitoringDecayScheduler(DeltaDelayedDecayScheduler):
    """
    This is the base scheduler where delta is only updated when the target
    performance is reached. Caution: under this scheduler the delta might never
    reach 1.
    """

    internal_target_performance: float
    performance_offset: float

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,  # unused
        delta_decay_rate: float,
        target_module_type: str,
        performance_offset: float = 1e-10,
    ):
        super().__init__(
            initial_delta,
            delta_decay_delay,
            0,
            delta_decay_rate,
            target_module_type,
        )

        self.internal_target_performance = 0.0
        self.performance_offset = performance_offset

    def step(
        self,
        module: BaseNeuralDNF | BaseSemiSymbolic,
        current_performance: float,
        order: str = "ascending",
    ) -> dict[str, list[float]]:
        """
        Calculate the new delta value and set it in the model.
        Returns the new delta value and the old delta value.
        `target_performance` is the performance that the model should reach
        before delta is updated. This is set at the `step` that equals to
        `delta_decay_delay` by taking the `current_performance`.
        `order` should be either "ascending" or "descending" to indicate the
        order of the performance.
        """

        assert order in ["ascending", "descending"], "Invalid order."

        assert self._check_type_matching_at_step(
            module
        ), "Type mismatch between the module to change and this scheduler's "
        "target_module_type."
        if isinstance(module, BaseSemiSymbolic):
            old_delta = module.delta
            new_delta, _ = self._step_layer(
                old_delta,
                self.internal_step_counters[0],
                current_performance,
                order,
            )
            module.delta = new_delta
            self.internal_step_counters[0] += 1
            return {
                "new_delta_vals": [new_delta],
                "old_delta_vals": [old_delta],
            }

        old_delta_vals = module.get_delta_val()
        assert len(old_delta_vals) == self.internal_step_counters_len, (
            "The length of the internal step counter "
            f"({self.internal_step_counters_len}) and the length of the "
            f"delta values in the model ({len(old_delta_vals)}) should be "
            "the same."
        )

        new_delta_vals = []
        for i in range(self.internal_step_counters_len):
            old_delta = old_delta_vals[i]
            new_delta, _ = self._step_layer(
                old_delta,
                self.internal_step_counters[i],
                current_performance,
                order,
            )
            new_delta_vals.append(new_delta)
            self.internal_step_counters[i] += 1

        module.set_delta_val(new_delta_vals)
        return {
            "new_delta_vals": new_delta_vals,
            "old_delta_vals": old_delta_vals,
        }

    def _step_layer(
        self,
        old_delta_val: float,
        step: int,
        current_performance: float,
        order: str,
    ) -> tuple[float, float]:
        if step < self.delta_decay_delay:
            return old_delta_val, old_delta_val

        if step == self.delta_decay_delay:
            self.internal_target_performance = current_performance
        if (
            order == "ascending"
            and current_performance - self.internal_target_performance
            < -self.performance_offset
        ):
            # If the current performance is less than the target performance
            # then don't update delta
            return old_delta_val, old_delta_val
        if (
            order == "descending"
            and current_performance - self.internal_target_performance
            > -self.performance_offset
        ):
            # If the current performance is greater than the target performance
            # then don't update delta
            return old_delta_val, old_delta_val
        if old_delta_val == 1.0:
            # If the delta is already 1 then don't need to calculate for new
            # delta
            return old_delta_val, old_delta_val

        new_delta_val = self._calculate_new_delta(step)
        new_delta_val = 1 if new_delta_val > 1 else new_delta_val
        # Since we update the delta, we also update the target performance
        # to be the current performance
        self.internal_target_performance = current_performance
        return new_delta_val, old_delta_val

    def _calculate_new_delta(self, step: int) -> float:
        raise NotImplementedError


class DeltaDelayedMonitoringExponentialDecayScheduler(
    DeltaDelayedMonitoringDecayScheduler
):
    """
    This scheduler implements the base DeltaDelayedMonitoringDecayScheduler,
    with the delta calculation based on the exponential calculation.
    """

    def _calculate_new_delta(self, step: int) -> float:
        return self.initial_delta * (self.delta_decay_rate ** (step + 1))


class DeltaDelayedMonitoringLinearDecayScheduler(
    DeltaDelayedMonitoringDecayScheduler
):
    """
    This scheduler implements the base DeltaDelayedMonitoringDecayScheduler,
    with the delta calculation based on the linear calculation.
    """

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,  # unused
        delta_decay_rate: float,
        target_module_type: str,
        performance_offset: float = 1e-10,
    ):
        super().__init__(
            initial_delta,
            delta_decay_delay,
            delta_decay_steps,
            delta_decay_rate,
            target_module_type,
            performance_offset,
        )
        assert (
            0 < delta_decay_rate < 1
        ), "Decay rate should be between 0 and 1 for the linear scheduler."

    def _calculate_new_delta(self, step: int) -> float:
        return self.initial_delta + step * self.delta_decay_rate
