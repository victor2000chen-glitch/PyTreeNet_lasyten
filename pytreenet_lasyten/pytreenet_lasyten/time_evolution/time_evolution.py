"""
This module provides the abstract TimeEvolution class and local propagators.

LASyTen extension: TimeEvoMode.RK23 and RK45 integrate native tensors without
flattening, block packing, or dense tensor export. The native FASTEST policy
selects RK45. Existing NumPy/SciPy propagation and result storage are retained.
Native propagation solves dA/dt = forward.exp_factor() * H_action(t, A),
preserves tensor spaces, and never normalizes the state. A complex duration
follows the straight contour from zero to that duration.

For native tensors rtol/atol are nonnegative scalar tolerances with global
Frobenius error control: ||error|| <= atol + rtol*max(||A||, ||A_new||).
This differs from SciPy's componentwise weighted RMS criterion. Defaults are
rtol=1e-8 and atol=1e-10. Optional first_step/max_step/min_step are positive
path-length increments, max_steps bounds attempted substeps, and stats may be
a dict receiving counters. Other SciPy options are not silently ignored.
Native EXPM/CHEBYSHEV/SPARSE/DOP853/BDF are outside this initial port.

ChargedTensor inputs require a one-dimensional fixed-charge leg. Their hidden
charge-state amplitude is absorbed into the invariant part and a unit charge
state is used internally. This prevents zero coefficients from producing a
zero hidden state that breaks subsequent native additions. Only scalar charge
amplitudes/norms are read on the CPU; tensor coefficients stay in the backend.

This file does not port BUG basis enrichment/truncation or the separate
effective_time_evolution.py wrapper. Use RK23/RK45 explicitly while those
call sites still use action_evolvable() without a tensor argument.
"""
from __future__ import annotations
from typing import List, Union, Any, Dict, Iterable, Callable
from enum import Enum
from copy import deepcopy
from math import modf, isfinite, nextafter, inf
from numbers import Number, Integral
from warnings import warn
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numpy.typing import DTypeLike

from ..util.ttn_exceptions import positivity_check, non_negativity_check
from ..util import fast_exp_action
from ..util import lasyten_backend as lb
from ..util.tensor_dispatch import is_lasyten_tensor
from .results import Results

# Native BUG speedups: internal helper binding.
from ..util import bug_native_speedups as _bug_native_speedups

@dataclass
class TimeEvoConfig:
    """
    Abstract configuration for time evolution algorithms.
    
    This is to ensure that all time evolution algorithms can be configured.
    """
    time_dep: bool = False

class TimeEvolution:
    """
    An abstract class that can be used for various time-evolution algorithms.

    Contains all methods required to cover common discrete time evolutions and
    avoid code duplication. The algorithms runs a number of time steps of given
    size until a specified final time is reached.

    While results can be called directly as a big martix using the `results`
    attribute, there are convenience functions to easily extract the desired
    results.

    Attributes:
        initial_state (Any): The initial state of the system to be time evolved.
        time_step_size (float): The size of each discreet time step.
        final_time (float): The time at which to conclude the time-evolution.
        operators (List[Any]): A list of operators to be evaluated.
    """
    config_class = TimeEvoConfig

    def __init__(self,
                 initial_state: Any,
                 time_step_size: float,
                 final_time: float,
                 operators: Union[List[Any], Dict[str, Any], Any],
                 solver_options: Union[Dict[str, Any], None] = None,
                 config: Union[TimeEvoConfig, None] = None):
        """
        Initialises a TimeEvolution object.

        Args:
            initial_state (Any): The initial state.
            time_step_size (float): The size of one time step.
            final_time (float): The final time.
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values. Can be a single
                operator or a list of operators. If a dictionary is given, the
                results can be called by using the keys of the dictionary.
            solver_options (Union[Dict[str, Any], None], optional): Most time
                evolutions algorithms use some kind of solver to resolve a
                partial differential equation. This dictionary can be used to
                pass additional options to the solver. Refer to the
                documentation of `ptn.time_evolution.TimeEvoMode` for further
                information. Defaults to None.
            config (Union[TimeEvoConfig, None], optional): The configuration
                for the time evolution algorithm. If None, a default
                configuration is used.
        """
        self._initial_state = initial_state
        self.state = deepcopy(initial_state)
        positivity_check(time_step_size, "size of one time step")
        self._time_step_size = time_step_size
        positivity_check(final_time, "final time")
        self._final_time = final_time
        self._num_time_steps = self._compute_num_time_steps()
        self.operators = self.init_operators(operators)
        if config is None:
            self.config = self.config_class()
        else:
            self.config = config
        if solver_options is None:
            self.solver_options = {}
        else:
            self.solver_options = solver_options
        self.results = Results()

    def _compute_num_time_steps(self) -> int:
        """
        Compute the number of time steps from attributes.

        If the decimal part of the time steps is close to 0, the calculated
        number of time steps is directly returned. Otherwise, it is assumed
        to be better to run one more time step.
        """
        decimal, integer = modf(self._final_time / self._time_step_size)
        threshold = 0.1
        if decimal < threshold:
            return int(integer)
        return int(integer + 1)

    def init_operators(self,
                       operators: Union[List[Any], Dict[str, Any], Any]
                       ) -> Dict[str, Any]:
        """
        Initialises the operators for which to compute the expectation values.

        Args:
            operators (Union[List[Any], Dict[str, Any], Any]): The operators
                for which to compute the expectation values. Can be a single
                operator or a list of operators. If a dictionary is given, the
                results can be called by using the keys of the dictionary
                otherwise the keys are the indices of the operators in the list.

        Returns:
            Dict[str, Any]: A dictionary mapping the operator names to the
                operators. If a list of operators is given, the keys are the
        """
        if isinstance(operators, List):
            return {str(i): op for i, op in enumerate(operators)}
        if isinstance(operators, Dict):
            return operators
        return {"0": operators}

    @property
    def initial_state(self) -> Any:
        """
        Returns the initial state.
        """
        return self._initial_state

    @property
    def time_step_size(self) -> float:
        """
        Returns the size of one time step.
        """
        return self._time_step_size

    @property
    def final_time(self) -> float:
        """
        Returns the final time.
        """
        return self._final_time

    @property
    def num_time_steps(self) -> int:
        """
        Returns the current number of time steps.
        """
        return self._num_time_steps

    def set_num_time_steps(self, num_time_steps: int):
        """
        Set the number of time steps to be run.
        
        Sometimes it is more convenient to define the size of the time steps
        and the number of steps to be run, rather than using a final time.
        This method modifies the internal attributes accordingly.
        """
        non_negativity_check(num_time_steps, "number of time steps")
        self._num_time_steps = num_time_steps
        self._final_time = num_time_steps * self._time_step_size

    def set_num_time_steps_constant_final_time(self, num_time_steps: int):
        """
        Sets the number of time-steps and keeps the final time constant.
        
        The internal attributes are modified accordingly.
        """
        non_negativity_check(num_time_steps, "number of time steps")
        self._num_time_steps = num_time_steps
        self._time_step_size = self._final_time / num_time_steps

    def run_one_time_step(self, **kwargs):
        """
        Abstract method to run one time step.
        """
        raise NotImplementedError()

    def evaluate_operator(self, operator: Any) -> complex:
        """
        Abstract method to evaluate the expectation value of a single operator.

        Args:
            operator (Any): The operator for which to compute the expectation
                value.

        Returns:
            complex: The expectation value of the operator.
        """
        raise NotImplementedError()

    def update_hamiltonian(self):
        """
        Abstract method to update the Hamiltonian for the next time step.
        
        This is only required for time-dependent Hamiltonians.
        """
        raise NotImplementedError("This method should be implemented in a subclass!")

    def evaluate_operators(self, index: int):
        """
        Evaluate the expectation value for all operators for the current state.

        Args:
            index (int): The index at which to save the results.
        """
        for key, operator in self.operators.items():
            exp_val = self.evaluate_operator(operator)
            self.results.set_element(key, index, exp_val)

    def result_init_dictionary(self) -> Dict[str, DTypeLike]:
        """
        Returns a dictionary with the initialisation of the results.

        The keys are the operator names and the values are the data types
        of the results.
        
        Returns:
            Dict[str, DTypeLike]: The initialisation dictionary for the results.
        """
        return {key: complex for key in self.operators.keys()}

    def init_results(self, evaluation_time: Union[int,"inf"] = 1):
        """
        Initialises an appropriately sized zero valued numpy array for storage.

        Each row contains the results obtained for one operator, while the
        last row contains the times. Note, the the entry with index zero
        corresponds to time 0.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
        """
        if evaluation_time == "inf":
            num_results = 1
        else:
            num_results = self.num_time_steps // evaluation_time
        res_dtypes = self.result_init_dictionary()
        self.results.initialize(res_dtypes, num_results)

    def save_time(self, time_step: int, index: int):
        """
        Saves the current time at a given index.

        Args:
            time_step (int): The current time step.
            index (int): The index at which to save the time.

        """
        self.results.set_time(index, time_step*self.time_step_size)

    def result_index(self,
                     evalutation_time: Union[int,"inf"],
                     time_step: int) -> int:
        """
        Returns the time index at which to save the results.

        Args:
            evalutation_time (Union[int,"inf"]): The difference in time steps
                after which to evaluate the operator expectation values.
            time_step (int): The current time step.
        
        Returns:
            int: The index at which to save the results.

        """
        if evalutation_time == "inf":
            return 0
        return time_step // evalutation_time

    def should_evaluate(self,
                        evaluation_time: Union[int,"inf"],
                        time_step: int
                        ) -> bool:
        """
        Returns if the operators should be evaluated at the current time step.

        Args:
            evaluation_time (Union[int,"inf"]): The difference in time steps after which
                to evaluate the operator expectation values.
            time_step (int): The current time step.
        
        Returns:
            bool: If the operators should be evaluated at the current time step.
        
        """
        some_time_step = evaluation_time != "inf" and time_step % evaluation_time == 0
        end_time = evaluation_time == "inf" and time_step == self.num_time_steps
        return some_time_step or end_time

    def evaluate_and_save_results(self,
                                  evaluation_time: Union[int,"inf"],
                                  time_step: int):
        """
        Evaluates and saves the operator results at a given time step.

        Args:
            evaluation_time (Union[int,"inf"]): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
            time_step (int): The current time step.

        """
        if self.should_evaluate(evaluation_time, time_step):
            index = self.result_index(evaluation_time, time_step)
            self.evaluate_operators(index)
            self.save_time(time_step, index)

    def create_run_tqdm(self, pgbar: bool = True) -> Iterable:
        """
        Creates the decorated iterator for the progress bar.

        Args:
            pgbar (bool, optional): If True, a progress bar is shown. 
                Defaults to True.
        
        Returns:
            Iterable: The decorated iterator.

        """
        return tqdm(range(self.num_time_steps + 1), disable=not pgbar)

    def run(self,
            evaluation_time: Union[int,"inf"] = 1,
            pgbar: bool = True):
        """
        Runs this time evolution algorithm for the given parameters.

        The desired operator expectation values are evaluated and saved.

        Args:
            evaluation_time (int, optional): The difference in time steps after which
                to evaluate the operator expectation values, e.g. for a value of 10
                the operators are evaluated at time steps 0,10,20,... If it is set to
                "inf", the operators are only evaluated at the end of the time.
                Defaults to 1.
            pgbar (bool, optional): Toggles the progress bar. Defaults to True.
        """
        self.init_results(evaluation_time)
        for i in self.create_run_tqdm(pgbar):
            if i != 0:  # We also measure the initial expectation_values
                self.run_one_time_step()
            self.evaluate_and_save_results(evaluation_time, i)
            if self.config.time_dep and i != 0:
                self.update_hamiltonian()

    def reset_hamiltonian(self):
        """
        Resets the Hamiltonian to its initial state.

        This is only required for time-dependent Hamiltonians.
        """
        raise NotImplementedError("This method should be implemented in a subclass!")

    def reset_to_initial_state(self):
        """
        Resets the current state to the intial state
        """
        self.state = deepcopy(self._initial_state)
        self.results = Results()
        if self.config_class.time_dep:
            self.reset_hamiltonian()

# Embedded Bogacki-Shampine 3(2) and Dormand-Prince 5(4) tableaux.
# Mathematical coefficients, also documented/used by SciPy's RK23/RK45:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
_NATIVE_RK = {
    "RK23": (
        (0., 1/2, 3/4),
        ((), (1/2,), (0., 3/4)),
        (2/9, 1/3, 4/9),
        (5/72, -1/12, -1/9, 1/8),
        3,
    ),
    "RK45": (
        (0., 1/5, 3/10, 4/5, 8/9, 1.),
        ((), (1/5,), (3/40, 9/40), (44/45, -56/15, 32/9),
         (19372/6561, -25360/2187, 64448/6561, -212/729),
         (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656)),
        (35/384, 0., 500/1113, 125/192, -2187/6784, 11/84),
        (-71/57600, 0., 71/16695, -71/1920, 17253/339200, -22/525, 1/40),
        5,
    ),
}


def _native_scale(tensor, coefficient):
    return _bug_native_speedups.native_scale(tensor, coefficient)


def _native_unit_charge(tensor, reference=None):
    """Absorb one scalar charge amplitude into invariant data, without exporting it."""
    ct = lb._lasyten()
    native = tensor.native
    if not isinstance(native, ct.ChargedTensor):
        return tensor
    if native.charge_leg.dim != 1:
        raise NotImplementedError("Native propagation currently requires a one-dimensional fixed-charge leg.")
    if reference is None:
        unit = native.backend.block_backend.ones_block(
            [1], dtype=native.dtype, device=native.device)
    else:
        unit = reference.native.charged_state
    factor = lb._scalar(native.backend.block_backend.inner(native.charged_state, unit, False))
    if factor == 1:
        return tensor
    return lb.LasyTenArray(ct.ChargedTensor(native.invariant_part * factor, unit))


def _native_check(value, reference):
    """Validate a derivative/action without permitting NumPy conversion."""
    if not is_lasyten_tensor(value):
        raise TypeError("A native Hamiltonian action must return a LASyTen tensor, not a NumPy array.")
    value = lb.as_lasyten(value)
    if value.shape != reference.shape or value.legs != reference.legs:
        raise ValueError("The native Hamiltonian action changed the tensor shape or public leg spaces.")
    if (value.native.device != reference.native.device
            or type(value.native.backend) is not type(reference.native.backend)):
        raise ValueError("The native Hamiltonian action changed the backend or device.")
    ct = lb._lasyten()
    charged = isinstance(reference.native, ct.ChargedTensor)
    if isinstance(value.native, ct.ChargedTensor) != charged:
        raise ValueError("The native Hamiltonian action changed the charged/invariant representation.")
    if charged and value.native.charge_leg != reference.native.charge_leg:
        raise ValueError("The native Hamiltonian action changed the fixed-charge space.")
    value = value.astype(np.complex128, copy=False)
    return _native_unit_charge(value, reference) if charged else value


def _native_sum(base, coefficients, vectors, step=1., include_base=True):
    return _bug_native_speedups.native_sum(base, coefficients, vectors, step, include_base)


def _native_real_option(name, value, *, positive=False, infinite=False):
    if not isinstance(value, Number) or np.iscomplexobj(value):
        raise TypeError(f"Native {name} must be a real scalar.")
    value = float(value)
    if (not isfinite(value) and not (infinite and value == inf)):
        raise ValueError(f"Native {name} must be finite" + (" or positive infinity." if infinite else "."))
    if value < 0 or (positive and value == 0):
        raise ValueError(f"Native {name} must be {'positive' if positive else 'nonnegative'}.")
    return value


def _native_time_evolve_action(psi, action, duration, forward, method, **options):
    """Adaptive native RK, optionally removing a scalar phase rotation.

    energy_shift='auto' uses the real initial Rayleigh quotient; a finite
    real scalar selects an explicit shift. Shifting is restricted to real
    durations so the phase has unit modulus and norm-based error tolerances
    keep their meaning. The final phase is restored without normalization.
    """
    method = "RK45" if method == "fastest" else method
    if method not in _NATIVE_RK:
        raise NotImplementedError(
            f"Native LASyTen propagation does not implement {method}. Use RK23, RK45, or FASTEST.")
    if not callable(action):
        raise TypeError("time_evo_action must be callable as action(t, tensor).")
    if not isinstance(duration, Number):
        raise TypeError("Native time_difference must be a finite real or complex scalar.")
    duration = complex(duration)
    if not (isfinite(duration.real) and isfinite(duration.imag)):
        raise ValueError("Native time_difference must be finite.")
    length = abs(duration)
    if not isfinite(length):
        raise ValueError("The native contour length is not finite.")
    if isinstance(forward, (bool, np.bool_)):
        forward = EvoDirection.from_bool(bool(forward))
    if not isinstance(forward, EvoDirection):
        raise TypeError("forward must use this package's EvoDirection enum or a boolean.")
    shift_option = options.pop("energy_shift", None)
    automatic_shift = isinstance(shift_option, str) and shift_option == "auto"
    if automatic_shift or shift_option is None:
        energy_shift = 0.
    elif (isinstance(shift_option, bool) or not isinstance(shift_option, Number)
          or np.iscomplexobj(shift_option) or not isfinite(float(shift_option))):
        raise ValueError("energy_shift must be None, 'auto', or a finite real scalar.")
    else:
        energy_shift = float(shift_option)
    if duration.imag and (automatic_shift or energy_shift != 0):
        raise ValueError("Native energy shifting currently requires a real time duration.")
    rtol = _native_real_option("rtol", options.pop("rtol", 1e-8))
    atol = _native_real_option("atol", options.pop("atol", 1e-10))
    if rtol == 0 and atol == 0:
        raise ValueError("At least one native error tolerance must be positive.")
    max_step = _native_real_option("max_step", options.pop("max_step", inf), positive=True, infinite=True)
    min_step = _native_real_option("min_step", options.pop("min_step", 0.))
    if min_step > max_step:
        raise ValueError("Native min_step exceeds max_step.")
    first_step = options.pop("first_step", None)
    first_step = _bug_native_speedups.propose_first_step(length, first_step)
    if first_step is not None:
        first_step = _native_real_option("first_step", first_step, positive=True)
        if length and first_step > length:
            raise ValueError("Native first_step exceeds the contour length.")
        if length > first_step and first_step < min_step:
            raise ValueError("Native first_step is smaller than min_step.")
    max_steps = options.pop("max_steps", 100000)
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 1:
        raise ValueError("Native max_steps must be a positive integer.")
    stats = options.pop("stats", None)
    if stats is not None and not isinstance(stats, dict):
        raise TypeError("Native stats must be a dict or None.")
    if options:
        raise TypeError("Unsupported native propagation options: " + ", ".join(sorted(options)))
    counters = dict(method=method, nfev=0, n_steps=0, n_accepted=0, n_rejected=0,
                    path_length=0., success=False, energy_shift=energy_shift)
    if stats is not None:
        stats.update(counters)

    reference = lb.as_lasyten(psi).copy().astype(np.complex128, copy=False)
    if reference.ndim < 1 or reference.size == 0:
        raise ValueError("The native initial tensor must have nonempty public legs.")
    reference = _native_unit_charge(reference)
    y = reference
    norm_y = lb.norm(y)
    if not isfinite(norm_y):
        raise ValueError("The native initial tensor has a nonfinite norm.")
    if length == 0:
        counters["success"] = True
        if stats is not None:
            stats.update(counters)
        return y
    direction = duration / length if duration.imag else float(duration.real / length)
    exponent = forward.exp_factor() * direction

    initial_action = None
    if automatic_shift:
        counters["nfev"] += 1
        initial_action = _native_check(action(0., y), reference)
        # Sequential division avoids forming norm_y**2 unnecessarily.
        estimate = lb.vdot(y, initial_action) / norm_y / norm_y if norm_y else 0.
        energy_shift = float(np.real(estimate))
        if not isfinite(energy_shift):
            raise ValueError("Automatic energy shift is nonfinite; rescale the problem or supply a finite shift.")
        counters["energy_shift"] = energy_shift
        if stats is not None:
            stats.update(counters)

    def rhs(path_time, value, precomputed=None):
        if precomputed is None:
            counters["nfev"] += 1
            if stats is not None:
                stats["nfev"] = counters["nfev"]
            action_value = _native_check(action(direction * path_time, value), reference)
        else:
            action_value = precomputed
        if energy_shift:
            action_value = _native_sum(action_value, (-energy_shift,), (value,))
        return _native_scale(action_value, exponent)

    c, a, b, e, error_power = _NATIVE_RK[method]
    time = 0.
    step = min(max_step, length, first_step if first_step is not None else max(length / 10., min_step))
    if step == 0:
        step = min(max_step, length)  # subnormal duration/10 may underflow
    f_current = rhs(time, y, precomputed=initial_action)
    if not isfinite(lb.norm(f_current)):
        raise ValueError("The native Hamiltonian action returned a nonfinite initial derivative.")
    rejected = False
    for attempt in range(1, int(max_steps) + 1):
        counters["n_steps"] = attempt
        remaining = length - time
        step = min(step, max_step, remaining)
        machine_min = 10. * (nextafter(time, inf) - time)
        required_min = max(machine_min, min_step)
        if step <= 0 or time + step == time or (step < required_min and step != remaining):
            raise RuntimeError("Native RK cannot advance at the required step size; loosen tolerances or rescale the problem.")
        new_time = length if step == remaining else time + step
        stages = [f_current]
        for stage in range(1, len(c)):
            trial = _native_sum(y, a[stage], stages, step)
            stages.append(rhs(time + c[stage] * step, trial))
        candidate = _native_sum(y, b, stages, step)
        f_candidate = rhs(new_time, candidate)  # FSAL derivative reused after acceptance
        stages.append(f_candidate)
        error = _native_sum(y, e, stages, step, include_base=False)
        norm_candidate, norm_error = lb.norm(candidate), lb.norm(error)
        scale = atol + rtol * max(norm_y, norm_candidate)
        if not (isfinite(norm_candidate) and isfinite(norm_error)):
            ratio = inf
        else:
            ratio = norm_error / scale if scale else (0. if norm_error == 0 else inf)
        if ratio <= 1.:
            time, y, f_current, norm_y = new_time, candidate, f_candidate, norm_candidate
            counters["n_accepted"] += 1
            counters["path_length"] = time
            factor = 10. if ratio == 0 else min(10., .9 * ratio ** (-1. / error_power))
            step *= min(1., factor) if rejected else factor
            rejected = False
        else:
            counters["n_rejected"] += 1
            step *= max(.2, .9 * ratio ** (-1. / error_power))
            rejected = True
        if stats is not None:
            stats.update(counters)
        if time == length:
            counters["success"] = True
            if stats is not None:
                stats.update(counters)
            return _native_scale(y, np.exp(exponent * energy_shift * length)) if energy_shift else y
    raise RuntimeError(f"Native RK exceeded max_steps={max_steps} before reaching the requested time.")


def _native_hamiltonian_action(psi, hamiltonian):
    """Adapt a native operator tensor or a tensor-valued matrix-free matvec."""
    if is_lasyten_tensor(hamiltonian):
        h = lb.as_lasyten(hamiltonian)
        n = psi.ndim
        if h.ndim != 2 * n:
            raise ValueError(
                "For an n-leg native state, supply an operator with output legs followed by n input legs, "
                "or use time_evolve_action with the tensor-shaped matrix-free Hamiltonian. "
                "A fused matrix cannot implicitly flatten a native site tensor.")
        if h.shape[:n] != psi.shape or h.shape[n:] != psi.shape:
            raise ValueError("The native Hamiltonian's input/output dimensions do not match the state.")
        if tuple(h.legs[:n]) != psi.legs or any(
                not lb.spaces_contractible(h.legs[n+i], psi.legs[i]) for i in range(n)):
            raise ValueError("The native Hamiltonian's input/output spaces do not match the state.")
        return lambda t, y: lb.tensordot(h, y, axes=(tuple(range(n, 2*n)), tuple(range(n))))
    if callable(getattr(hamiltonian, "matvec", None)):
        return lambda t, y: hamiltonian.matvec(y)
    if callable(hamiltonian):
        return lambda t, y: hamiltonian(y)
    raise TypeError("A native state requires a native Hamiltonian tensor or a native tensor-valued matvec.")


class EvoDirection(Enum):
    """
    Enum for the direction of time evolution.
    """
    FORWARD = -1
    BACKWARD = 1

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_bool(forward: bool) -> EvoDirection:
        """
        Converts a boolean to a EvoDirection enum.

        Args:
            forward (bool): If True, FORWARD is returned. Otherwise BACKWARD.

        Returns:
            EvoDirection: The corresponding EvoDirection enum.
        """
        return EvoDirection.FORWARD if forward else EvoDirection.BACKWARD

    def exp_sign(self) -> int:
        """
        Returns the sign of the exponent for the time evolution.

        Returns:
            int: The sign of the exponent.
        """
        return self.value

    def exp_factor(self) -> complex:
        """
        Returns the factor for the exponent in the time evolution.

        Returns:
            complex: The factor for the exponent.
        """
        return 1.0j * self.exp_sign()

class TimeEvoMode(Enum):
    """
    Mode for the time evolution of a matrix.
    """

    FASTEST = "fastest"
    EXPM = "expm"
    CHEBYSHEV = "chebyshev"
    SPARSE = "sparse"
    RK45 = "RK45"
    RK23 = "RK23"
    DOP853 = "DOP853"
    BDF = "BDF"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def fastest_equivalent(psi=None) -> TimeEvoMode:
        """
        Selects the backend policy: native RK45 or the existing NumPy CHEBYSHEV.
        This is a deterministic default, not a measured performance guarantee.
        """
        return TimeEvoMode.RK45 if is_lasyten_tensor(psi) else TimeEvoMode.CHEBYSHEV

    @staticmethod
    def scipy_modes() -> List[TimeEvoMode]:
        """
        Returns a list of all modes that are scipy ODE solvers.
        
        Returns:
            List[TimeEvoMode]: The list of scipy modes.
        """
        return [TimeEvoMode.RK45,
                TimeEvoMode.RK23,
                TimeEvoMode.DOP853,
                TimeEvoMode.BDF]

    def is_scipy(self) -> bool:
        """
        Determines, if this mode is a scipy ODE solver.
        """
        if self == TimeEvoMode.FASTEST:
            return self.fastest_equivalent().is_scipy()
        return self in self.scipy_modes()

    def action_evolvable(self, psi=None) -> bool:
        """
        Determines, if this mode can be used for time evolution via an action.

        Returns:
            bool: True if the mode is suitable for time evolution via an action.
        """
        if is_lasyten_tensor(psi):
            return self in (TimeEvoMode.FASTEST, TimeEvoMode.RK23, TimeEvoMode.RK45)
        return self.is_scipy()

    def time_evolve_action(self,
                           psi: np.ndarray,
                           time_evo_action: Callable,
                           time_difference: float,
                           forward: EvoDirection| bool = EvoDirection.FORWARD,
                           **options
                           ) -> np.ndarray:
        """
        Performs the time evolution of a tensor from an action.

        Args:
            psi (np.ndarray): The initial state as an arbitrary tensor.
            time_evo_action (Callable): The action to be performed on the
                right hand side of the effective Schrödinger equation. Takes
                the time and the state as arguments.
            time_difference (float): The duration of the time-evolution.
            forward (EvoDirection|bool, optional): The direction of the time evolution.
            **options: Additional options for the solver
                underlying the time evolution. See the scipy documentation for
                the available options of the specific solver.

        Returns:
            np.ndarray: The time-evolved state.
        """
        if is_lasyten_tensor(psi):
            return _native_time_evolve_action(psi, time_evo_action, time_difference,
                                              forward, self.value, **options)
        if self.is_scipy():
            if isinstance(forward, bool):
                forward = EvoDirection.from_bool(forward)
            exp_factor = forward.exp_factor()
            orig_shape = psi.shape
            if not np.issubdtype(psi.dtype, np.complexfloating):
                psi = psi.astype(np.complex64)
                warnstr ="You supplied a real dtyped tensor, but the time"
                warnstr += " evolution is complex. The tensor will be cast to "
                warnstr += "complex64. This might impede performance!"
                warn(warnstr, UserWarning)
            def ode_rhs(t, y_vec):
                orig_array = y_vec.reshape(orig_shape)
                action_result = time_evo_action(t, orig_array).flatten()
                return exp_factor * action_result
            t_span = (0, time_difference)
            res = solve_ivp(ode_rhs, t_span, psi.flatten(),
                            method=self.value,
                            t_eval=[time_difference],
                            **options)
            return res.y[:, 0].reshape(orig_shape)
        raise NotImplementedError(
            "The time evolution via action is not implemented for this mode: "
            + str(self.value)
        )

    def time_evolve(self,
                    psi: np.ndarray,
                    hamiltonian: np.ndarray,
                    time_difference: float,
                    forward: EvoDirection| bool = EvoDirection.FORWARD,
                    **options: dict[str, Any]) -> np.ndarray:
        """
        Time evolves a state psi via a Hamiltonian matrix.

        Args:
            psi (np.ndarray): The initial state as a vector.
            hamiltonian (np.ndarray): The Hamiltonian determining the dynamics as
                a matrix.
            time_difference (float): The duration of the time-evolution
            forward (EvoDirection|bool, optional): The direction of the time evolution.
                    Defaults to EvoDirection.FORWARD.
            **options (dict[str, Any]): Additional options for the solver
                underlying the time evolution. See the scipy documentation for
                the available options of the specific solver.
        
        Returns:
            np.ndarray: The time evolved state
        """
        if is_lasyten_tensor(psi):
            native_state = lb.as_lasyten(psi)
            action = _native_hamiltonian_action(native_state, hamiltonian)
            return self.time_evolve_action(native_state, action, time_difference,
                                           forward=forward, **options)
        if is_lasyten_tensor(hamiltonian):
            raise TypeError("A native LASyTen Hamiltonian requires a native state.")
        if isinstance(forward, bool):
            forward = EvoDirection.from_bool(forward)
        if self.is_scipy():
            return self.time_evolve_action(psi,
                                           lambda t, y: hamiltonian @ y.flatten(),
                                           time_difference,
                                           forward=forward,
                                           **options)
        exponent = forward.exp_sign() * 1.0j * hamiltonian * time_difference
        return fast_exp_action(exponent, psi.flatten(),
                               mode=self.value).reshape(psi.shape)

def time_evolve(psi: np.ndarray, hamiltonian: np.ndarray,
                time_difference: float,
                forward: EvoDirection| bool = EvoDirection.FORWARD,
                mode: TimeEvoMode = TimeEvoMode.FASTEST,
                **options) -> np.ndarray:
    """
    Time evolves a state psi via a Hamiltonian.
     
    The evolution can be either forward or backward in time:
        psi(t +/- dt) = exp(-/+ i*h*dt) @ psi(t)
        -iHdt: forward = True
        +iHdt: forward = False

    Args:
        psi (np.ndarray): The initial state as a vector.
        hamiltonian (np.ndarray): The Hamiltonian determining the dynamics as
            a matrix.
        time_difference (float): The duration of the time-evolution
        forward (EvoDirection|bool, optional): The direction of the time evolution.
                Defaults to EvoDirection.FORWARD.
        mode (TimeEvoMode, optional): The mode to use for the time evolution.

    Returns:
        np.ndarray: The time evolved state
    """
    if is_lasyten_tensor(psi) or is_lasyten_tensor(hamiltonian) or options:
        return mode.time_evolve(psi, hamiltonian, time_difference, forward=forward, **options)
    if isinstance(forward, bool):
        forward = EvoDirection.from_bool(forward)
    sign = forward.exp_sign()
    rhs_matrix = sign * 1.0j * hamiltonian
    if mode.is_scipy():
        def ode_rhs(_, y_vec):
            return rhs_matrix @ y_vec
        t_span = (0, time_difference)
        solution = solve_ivp(ode_rhs, t_span, psi.flatten(),
                                method=mode.value,
                                t_eval=[time_difference])
        result_vector = solution.y[:,0]
    else:
        exponent = rhs_matrix * time_difference
        result_vector = fast_exp_action(exponent, psi.flatten(),
                                        mode=mode.value)
    return result_vector.reshape(
                      psi.shape)
