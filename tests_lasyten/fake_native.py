"""Protected U1 tensor double for algebra/layout tests, not performance claims."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


@dataclass(frozen=True)
class Space:
    charges: tuple
    tag: str
    is_dual: bool = False

    @property
    def dim(self):
        return len(self.charges)

    @property
    def dual(self):
        return Space(self.charges, self.tag, not self.is_dual)

    @property
    def signed(self):
        return np.asarray(self.charges) * (-1 if self.is_dual else 1)


@dataclass(frozen=True)
class Symmetry:
    kind: str = 'u1'

    def is_equivalent_to(self, other):
        return self == other


class BlockBackend:
    def ones_block(self, shape, dtype, device):
        return np.ones(shape, dtype=dtype)

    def inner(self, a, b, do_dagger):
        return np.vdot(a, b) if do_dagger else np.dot(a, b)


BACKEND = SimpleNamespace(block_backend=BlockBackend())
U1, NONE = Symmetry(), Symmetry('none')


def charge_sum(legs):
    shape = tuple(leg.dim for leg in legs)
    total = np.zeros(shape)
    for axis, leg in enumerate(legs):
        dimensions = [1]*len(legs)
        dimensions[axis] = leg.dim
        total += leg.signed.reshape(dimensions)
    return total


class Symmetric:
    def __init__(self, data, legs, labels=None, codomain=None, symmetry=U1):
        self.data = np.asarray(data)
        self.legs = tuple(legs)
        assert self.data.shape == tuple(leg.dim for leg in legs)
        self.labels = list(labels) if labels is not None else [None]*len(legs)
        self.num_codomain_legs = len(legs) if codomain is None else codomain
        self.symmetry = symmetry
        self.backend, self.device = BACKEND, 'cpu'
        if symmetry == U1 and self.data.size:
            assert np.max(np.abs(self.data[charge_sum(self.legs) != 0]), initial=0) < 1e-11

    @property
    def shape(self):
        return self.data.shape

    @property
    def num_legs(self):
        return len(self.legs)

    @property
    def dtype(self):
        return self.data.dtype

    def set_labels(self, labels):
        assert len(labels) == self.num_legs
        assert len([x for x in labels if x is not None]) == len(set(x for x in labels if x is not None))
        self.labels = list(labels)

    def copy(self, deep=True):
        return Symmetric(self.data.copy() if deep else self.data, self.legs, self.labels,
                         self.num_codomain_legs, self.symmetry)

    def __mul__(self, coefficient):
        return Symmetric(self.data*coefficient, self.legs, self.labels,
                         self.num_codomain_legs, self.symmetry)


class Charged:
    def __init__(self, invariant_part, charged_state):
        self.invariant_part = invariant_part
        self.charged_state = np.array(charged_state, copy=True)
        self.symmetry = invariant_part.symmetry
        self.backend, self.device = BACKEND, 'cpu'
        self.legs = invariant_part.legs[:-1]
        self.labels = list(invariant_part.labels[:-1])
        self.num_codomain_legs = min(invariant_part.num_codomain_legs, len(self.legs))

    @property
    def charge_leg(self):
        return self.invariant_part.legs[-1].dual

    @property
    def num_legs(self):
        return len(self.legs)

    @property
    def shape(self):
        return tuple(leg.dim for leg in self.legs)

    @property
    def dtype(self):
        return np.result_type(self.invariant_part.dtype, self.charged_state.dtype)

    def set_labels(self, labels):
        assert len(labels) == self.num_legs
        self.invariant_part.set_labels(list(labels) + [None])
        self.labels = list(labels)

    def copy(self, deep=True):
        return Charged(self.invariant_part.copy(deep=deep), self.charged_state)

    def __mul__(self, coefficient):
        # Match LASyTen's scalar-in-hidden-state representation.
        return Charged(self.invariant_part, self.charged_state*coefficient)


class Native:
    SymmetricTensor, ChargedTensor = Symmetric, Charged

    def __init__(self):
        self.calls = dict(tdot=0, permute=0, wrappers=0, linear_combination=0)
        self.tensors = SimpleNamespace(linear_combination=self.linear_combination)

    def tdot(self, a, b, axes_a, axes_b):
        self.calls['tdot'] += 1
        for ia, ib in zip(axes_a, axes_b):
            assert a.legs[ia] == b.legs[ib].dual
        assert not (isinstance(a, Charged) and isinstance(b, Charged))
        charged_a, charged_b = isinstance(a, Charged), isinstance(b, Charged)
        aa = a.invariant_part if charged_a else a
        bb = b.invariant_part if charged_b else b
        values = np.tensordot(aa.data, bb.data, axes=(axes_a, axes_b))
        remain_a = [i for i in range(a.num_legs) if i not in axes_a]
        remain_b = [i for i in range(b.num_legs) if i not in axes_b]
        legs = [a.legs[i] for i in remain_a] + [b.legs[i] for i in remain_b]
        labels = [a.labels[i] for i in remain_a] + [b.labels[i] for i in remain_b]
        if charged_a:
            values = np.moveaxis(values, len(remain_a), -1)
        if charged_a or charged_b:
            original = a if charged_a else b
            hidden = original.invariant_part.legs[-1]
            invariant = Symmetric(values, legs + [hidden], labels + [None], codomain=len(remain_a))
            return Charged(invariant, original.charged_state)
        return Symmetric(values, legs, labels, codomain=len(remain_a), symmetry=a.symmetry)

    def permute_legs(self, a, codomain, domain):
        self.calls['permute'] += 1
        axes = list(codomain) + list(reversed(domain))
        assert sorted(axes) == list(range(a.num_legs))
        charged = isinstance(a, Charged)
        original = a.invariant_part if charged else a
        values = original.data.transpose(axes + [a.num_legs] if charged else axes)
        legs = [a.legs[i] for i in axes]
        labels = [a.labels[i] for i in axes]
        if charged:
            inv = Symmetric(values, legs + [original.legs[-1]], labels + [None], codomain=len(codomain))
            return Charged(inv, a.charged_state)
        return Symmetric(values, legs, labels, codomain=len(codomain), symmetry=a.symmetry)

    def dagger(self, a):
        axes = list(reversed(range(a.num_legs)))
        labels = [None if a.labels[i] is None else a.labels[i].removesuffix('*')
                  if a.labels[i].endswith('*') else a.labels[i]+'*' for i in axes]
        if isinstance(a, Charged):
            inv = a.invariant_part
            values = inv.data.conj().transpose(axes + [a.num_legs])
            part = Symmetric(values, [a.legs[i].dual for i in axes] + [inv.legs[-1].dual],
                             labels + [None], codomain=a.num_legs-a.num_codomain_legs)
            return Charged(part, a.charged_state.conj())
        return Symmetric(a.data.conj().transpose(axes), [a.legs[i].dual for i in axes],
                         labels, codomain=a.num_legs-a.num_codomain_legs, symmetry=a.symmetry)

    def linear_combination(self, a, v, b, w):
        self.calls['linear_combination'] += 1
        assert isinstance(v, Symmetric) and isinstance(w, Symmetric) and v.legs == w.legs
        return Symmetric(a*v.data+b*w.data, v.legs, v.labels, v.num_codomain_legs, v.symmetry)


def backend():
    ct = Native()

    class Array:
        def __init__(self, native):
            ct.calls['wrappers'] += 1
            if native.num_codomain_legs != native.num_legs:
                native = ct.permute_legs(native, list(range(native.num_legs)), [])
            if native.symmetry == NONE and any(l.is_dual for l in native.legs):
                native = Symmetric(native.data, [l.dual if l.is_dual else l for l in native.legs],
                                   native.labels, symmetry=NONE)
            self.native = native

        @property
        def shape(self):
            return self.native.shape

        @property
        def ndim(self):
            return self.native.num_legs

        @property
        def size(self):
            return int(np.prod(self.shape))

        @property
        def legs(self):
            return self.native.legs

        @property
        def dtype(self):
            return self.native.dtype

        def copy(self):
            return Array(self.native.copy(deep=True))

        def astype(self, dtype, copy=True):
            assert np.dtype(dtype) == self.dtype
            return self.copy() if copy else self

        def __add__(self, other):
            a, b = self.native, other.native
            if isinstance(a, Charged):
                assert np.array_equal(a.charged_state, b.charged_state)
                return Array(Charged(ct.linear_combination(1., a.invariant_part, 1., b.invariant_part), a.charged_state))
            return Array(ct.linear_combination(1., a, 1., b))

        def __array__(self, *args, **kwargs):
            raise AssertionError('Implicit dense conversion is forbidden.')

        def to_numpy(self):
            raise AssertionError('Dense export is forbidden.')

    lb = SimpleNamespace(LasyTenArray=Array, _lasyten=lambda: ct,
                         as_lasyten=lambda x: x, _no_symmetry=lambda x: x.native.symmetry == NONE,
                         _scalar=lambda x: np.asarray(x).item())
    lb.transpose = lambda a, axes: Array(ct.permute_legs(a.native, list(axes), []))
    lb.norm = lambda a: float(np.linalg.norm(data(a)))
    lb.vdot = lambda a,b: np.vdot(data(a), data(b))
    return lb, ct


def data(array):
    native = array.native
    return native.invariant_part.data[..., 0]*native.charged_state.item() if isinstance(native, Charged) else native.data


def random_invariant(rng, legs, labels=None):
    shape = tuple(l.dim for l in legs)
    values = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    values *= charge_sum(legs) == 0
    return Symmetric(values, legs, labels)


def fixture(lb, neighbours=2, physical=2, charge=0, ham_order=None, charged=True):
    rng = np.random.default_rng(513 + neighbours + physical)
    ids = [f'v{i}' for i in range(neighbours)]
    spaces = [Space((-1,0,1), key) for key in ids]
    p = Space((0,1) if physical == 2 else (0,), 'p')
    legs = spaces + [p]
    if charged:
        hidden = Space((charge,), 'N', True)
        inv = random_invariant(rng, legs + [hidden], [f's{i}' for i in range(len(legs))]+[None])
        state = lb.LasyTenArray(Charged(inv, np.array([.8+.2j])))
    else:
        state = lb.LasyTenArray(random_invariant(rng, legs, [f's{i}' for i in range(len(legs))]))
    envs = {}
    op_spaces = [Space((-2,-1,0,1,2), f'w{i}') for i in range(neighbours)]
    for i, key in enumerate(ids):
        envs[key] = lb.LasyTenArray(random_invariant(rng, [spaces[i].dual, op_spaces[i], spaces[i]],
                                                   [f'e{i}_{j}' for j in range(3)]))
    ham_order = ids if ham_order is None else ham_order
    hlegs = [op_spaces[ids.index(key)].dual for key in ham_order] + [p, p.dual]
    ham = lb.LasyTenArray(random_invariant(rng, hlegs, [f'h{i}' for i in range(len(hlegs))]))
    node = SimpleNamespace(identifier='center', neighbouring_nodes=lambda: list(ids),
                           nopen_legs=lambda: 1, nneighbours=lambda: neighbours)
    hnode = SimpleNamespace(neighbour_index=lambda key: ham_order.index(key),
                            nopen_legs=lambda: 2, nneighbours=lambda: neighbours)
    cache = SimpleNamespace(get_entry=lambda neighbour, node: envs[neighbour])
    return state, node, ham, hnode, cache, envs


def baseline(lb, state, node, ham, hnode, cache):
    ct = lb._lasyten()
    result = state
    for key in node.neighbouring_nodes():
        result = lb.LasyTenArray(ct.tdot(result.native, cache.get_entry(key,node.identifier).native, [0], [0]))
    v = node.nneighbours()
    return lb.LasyTenArray(ct.tdot(result.native, ham.native,
        list(range(1,2*v+1,2))+[0], [hnode.neighbour_index(k) for k in node.neighbouring_nodes()]+[v+1]))
