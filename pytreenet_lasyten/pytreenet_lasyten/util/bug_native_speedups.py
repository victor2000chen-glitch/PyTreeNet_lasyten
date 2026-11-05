
"""Native BUG kernels; installed beside lasyten_backend.py.

All dense-equivalent size limits count coefficients, not total process memory.
No tensor coefficients are exported to NumPy by this module.
"""
from collections import Counter
from functools import lru_cache
import importlib
from math import prod

DEFAULTS = dict(full_first_step=True, prepare_actions=True, fast_wrapping=True,
                fast_conjugate=True, max_operator_elements=4096,
                max_intermediate_elements=65536)
_settings = dict(DEFAULTS)
_counts = Counter()
_combine_cache = (None, None)


@lru_cache(maxsize=1)
def _backend():
    return importlib.import_module('.lasyten_backend', __package__)


def configure(**options):
    """Configure this process. Use the installer to persist settings for workers."""
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown BUG speedup options: ' + ', '.join(sorted(unknown)))
    updated = dict(_settings, **options)
    for name, default in DEFAULTS.items():
        value = updated[name]
        if type(default) is bool:
            if type(value) is not bool:
                raise TypeError(name + ' must be boolean.')
        elif type(value) is not int or value < 0:
            raise ValueError(name + ' must be a nonnegative integer.')
    _settings.update(updated)
    global _combine_cache
    _combine_cache = (None, None)
    return dict(_settings)


def settings():
    return dict(_settings)


def reset_stats():
    _counts.clear()


def stats():
    return dict(_counts)


def propose_first_step(length, requested):
    if requested is None and length > 0 and _settings['full_first_step']:
        return length
    return requested


def _wrap_result(reference, native):
    """Wrap an internally derived U1 result; public input validation stays intact.

    The input wrapper has already passed LasyTenArray's symmetry validation.
    Tensor arithmetic preserves that symmetry. Retain its public leg convention
    while avoiding repeated symmetry checks. NoSymmetry uses the original
    constructor because its dual-leg normalization has different semantics.
    """
    lb = _backend()
    known_u1 = getattr(reference, '_bug_speedups_known_u1', False)
    if not _settings['fast_wrapping'] or (not known_u1 and lb._no_symmetry(reference)):
        return lb.LasyTenArray(native)
    ct = lb._lasyten()
    if native.num_codomain_legs != native.num_legs:
        native = ct.permute_legs(native, codomain=list(range(native.num_legs)), domain=[])
    result = object.__new__(lb.LasyTenArray)
    result.native = native
    result._bug_speedups_known_u1 = True
    _counts['trusted_u1_wraps'] += 1
    return result


def conjugate(tensor):
    lb = _backend()
    a = lb.as_lasyten(tensor)
    ct = lb._lasyten()
    axes = tuple(reversed(range(a.ndim)))
    if not _settings['fast_conjugate']:
        return lb.transpose(lb.LasyTenArray(ct.dagger(a.native)), axes)
    # Perform the dagger and its compensating public-axis reversal before
    # wrapping. This removes an intermediate wrapper and all-codomain move.
    native = ct.permute_legs(ct.dagger(a.native), codomain=list(axes), domain=[])
    _counts['fused_conjugates'] += 1
    return _wrap_result(a, native)


def native_scale(tensor, coefficient):
    lb = _backend()
    ct = lb._lasyten()
    if isinstance(tensor.native, ct.ChargedTensor):
        native = ct.ChargedTensor(tensor.native.invariant_part * coefficient,
                                  tensor.native.charged_state)
    else:
        native = tensor.native * coefficient
    return _wrap_result(tensor, native)


def _linear_combination(ct):
    global _combine_cache
    if _combine_cache[0] is ct:
        return _combine_cache[1]
    combine = getattr(ct, 'linear_combination', None)
    if not callable(combine):
        combine = getattr(getattr(ct, 'tensors', None), 'linear_combination', None)
    if not callable(combine):
        try:
            from lasyten.tensors import linear_combination as combine
        except ImportError:
            combine = None
    _combine_cache = (ct, combine if callable(combine) else None)
    return _combine_cache[1]


def native_sum(base, coefficients, vectors, step=1., include_base=True):
    lb = _backend()
    ct = lb._lasyten()
    combine = _linear_combination(ct)
    if combine is None:
        result = base.copy() if include_base else native_scale(base, 0.)
        for coefficient, vector in zip(coefficients, vectors):
            if coefficient != 0:
                result = result + native_scale(vector, step * coefficient)
        return result
    charged = isinstance(base.native, ct.ChargedTensor)
    accumulator = (base.native.invariant_part if charged else base.native) if include_base else None
    changed = False
    for coefficient, vector in zip(coefficients, vectors):
        coefficient = step * coefficient
        if coefficient == 0:
            continue
        native = vector.native.invariant_part if charged else vector.native
        accumulator = native * coefficient if accumulator is None else combine(1., accumulator, coefficient, native)
        changed = True
    if not changed:
        return base.copy() if include_base else native_scale(base, 0.)
    if charged:
        accumulator = ct.ChargedTensor(accumulator, base.native.charged_state)
    _counts['fused_rk_sums'] += 1
    return _wrap_result(base, accumulator)


def _unlabelled_copy(native):
    copied = native.copy(deep=False)
    copied.set_labels([None] * copied.num_legs)
    return copied


def _dot(ct, left, right, left_axes, right_axes):
    _counts['prepared_tdot_calls'] += 1
    return ct.tdot(left, right, list(left_axes), list(right_axes))


class _PreparedAction:
    def __init__(self, reference, ct, operator, environments, block_axes, ham_axes,
                 local_operator=None):
        self.reference = reference
        self.ct = ct
        self.operator = operator
        self.environments = environments
        self.block_axes = block_axes
        self.ham_axes = ham_axes
        self.local_operator = local_operator
        self.rank = reference.ndim
        self.labels = list(reference.native.labels)

    def __call__(self, time, tensor):
        _counts['prepared_matvecs'] += 1
        if self.local_operator is not None:
            result = _dot(self.ct, self.local_operator, tensor.native,
                          range(self.rank, 2*self.rank), range(self.rank))
        else:
            result = tensor.native
            for environment in self.environments:
                result = _dot(self.ct, result, environment, [0], [0])
            result = _dot(self.ct, result, self.operator, self.block_axes, self.ham_axes)
        result.set_labels(self.labels)
        return _wrap_result(tensor, result)


def _precontract_plan(ham_shape, env_shapes, ham_axes, output_axes, input_axes):
    """Predict every intermediate before allocating any precontracted tensor."""
    final_limit = _settings['max_operator_elements']
    intermediate_limit = _settings['max_intermediate_elements']
    if not final_limit or not intermediate_limit:
        return None
    tokens = [('h', i) for i in range(len(ham_shape))]
    shape = list(ham_shape)
    steps = []
    largest = 0
    for index, (environment, ham_axis) in enumerate(zip(env_shapes, ham_axes)):
        axis = tokens.index(('h', ham_axis))
        if shape[axis] != environment[1]:
            raise ValueError('Hamiltonian/environment bond dimensions disagree.')
        shape = shape[:axis] + shape[axis+1:] + [environment[0], environment[2]]
        tokens = tokens[:axis] + tokens[axis+1:] + [('ket', index), ('bra', index)]
        elements = prod(shape)
        if elements > intermediate_limit:
            _counts['precontract_intermediate_guard'] += 1
            return None
        largest = max(largest, elements)
        steps.append(axis)
    if prod(shape) > final_limit:
        _counts['precontract_operator_guard'] += 1
        return None
    output = [('bra', i) for i in range(len(env_shapes))] + [('h', i) for i in output_axes]
    inputs = [('ket', i) for i in range(len(env_shapes))] + [('h', i) for i in input_axes]
    if len(output) + len(inputs) != len(tokens) or set(output + inputs) != set(tokens):
        raise ValueError('Prepared local operator axes do not form a complete partition.')
    return steps, [tokens.index(t) for t in output], [tokens.index(t) for t in inputs], largest


def prepare_action(state_tensor, state_node, ham_tensor, ham_node, cache):
    """Prepare a fixed U1 action for one local solve, with bounded intermediates.

    Environment coefficients are already cached by SandwichCache. This routine
    captures them once and removes intermediate PyTreeNet wrappers from each
    matvec. Small operators are precontracted; large problems stay matrix-free.
    A new action is prepared after every basis/environment change.
    """
    if not _settings['prepare_actions']:
        return None
    lb = _backend()
    if not isinstance(state_tensor, lb.LasyTenArray) or not isinstance(ham_tensor, lb.LasyTenArray):
        return None
    if lb._no_symmetry(state_tensor):
        return None
    ct = lb._lasyten()
    if not isinstance(state_tensor.native, (ct.SymmetricTensor, ct.ChargedTensor)):
        _counts['unsupported_state_fallback'] += 1
        return None
    if not isinstance(ham_tensor.native, ct.SymmetricTensor):
        _counts['unsupported_operator_fallback'] += 1
        return None
    neighbours = tuple(state_node.neighbouring_nodes())
    v, p = len(neighbours), state_node.nopen_legs()
    if state_tensor.ndim != v + p or ham_node.nopen_legs() != 2*p:
        _counts['unsupported_rank_fallback'] += 1
        return None
    environments = []
    for neighbour in neighbours:
        value = cache.get_entry(neighbour, state_node.identifier)
        if (not isinstance(value, lb.LasyTenArray) or value.ndim != 3
                or not isinstance(value.native, ct.SymmetricTensor)):
            _counts['unsupported_environment_fallback'] += 1
            return None
        environments.append(value)
    ham_axes = [ham_node.neighbour_index(neighbour) for neighbour in neighbours]
    ham_input = list(range(ham_node.nneighbours() + p, ham_node.nneighbours() + 2*p))
    ham_output = list(range(ham_node.nneighbours(), ham_node.nneighbours() + p))
    env_shapes = [tuple(value.shape) for value in environments]
    input_shape = tuple(s[0] for s in env_shapes) + tuple(ham_tensor.shape[i] for i in ham_input)
    output_shape = tuple(s[2] for s in env_shapes) + tuple(ham_tensor.shape[i] for i in ham_output)
    if input_shape != state_tensor.shape or output_shape != state_tensor.shape:
        _counts['shape_fallback'] += 1
        return None
    plan = _precontract_plan(ham_tensor.shape, env_shapes, ham_axes, ham_output, ham_input)
    # These shallow copies own their labels; coefficient blocks stay shared.
    # Removing labels here prevents contraction-order-dependent label clashes.
    operator = _unlabelled_copy(ham_tensor.native)
    raw_environments = tuple(_unlabelled_copy(value.native) for value in environments)
    local_operator = None
    if plan is not None:
        axes, outputs, inputs, largest = plan
        local_operator = operator
        for axis, environment in zip(axes, raw_environments):
            local_operator = _dot(ct, local_operator, environment, [axis], [1])
        local_operator = ct.permute_legs(local_operator, codomain=outputs, domain=list(reversed(inputs)))
        _counts['precontracted_actions'] += 1
        _counts['largest_precontract_intermediate_elements'] = max(
            _counts['largest_precontract_intermediate_elements'], largest)
    else:
        _counts['matrix_free_actions'] += 1
    _counts['prepared_actions'] += 1
    block_axes = list(range(p, 2*v + p, 2)) + list(range(p))
    action = _PreparedAction(state_tensor, ct, operator, raw_environments,
                             block_axes, ham_axes + ham_input, local_operator)
    from pytreenet_lasyten.util.bug_performance import improve_action
    return improve_action(action, state_tensor, state_node, ham_tensor,
                          ham_node, cache, _wrap_result, _counts)

# Settings persisted by the installer.
configure(**{'full_first_step': True, 'prepare_actions': True, 'fast_wrapping': True, 'fast_conjugate': True, 'max_operator_elements': 4096, 'max_intermediate_elements': 65536})
