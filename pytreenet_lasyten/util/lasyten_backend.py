"""Explicit PyTreeNet -> LASyTen tensor adapter and topology storage helpers.

Target API: tenpy/lasyten cc8d5ab5951ee613cf5a9e3c87beece90a57f29e.
That LasyTen revision requires Python >= 3.14 and a built ``lasyten._core``.
This file itself uses Python 3.12-compatible syntax and NumPy for metadata.

Install this file as ``pytreenet_lasyten/util/lasyten_backend.py`` (or import it directly).
Replace numerical operations in PyTreeNet selectively, for example::

    from pytreenet_lasyten.util import lasyten_backend as cb
    result = cb.tensordot(a, b, axes=([1], [0]))
    q, r = cb.tensor_qr_decomposition(a, (0, 2), (1,))
    u, s, vh = cb.truncated_tensor_svd(a, (0, 2), (1,), svd_params)
    reconstructed = cb.tensordot(cb.scale_axis(u, s, -1), vh, axes=1)

The SVD spectrum is a LasyTenSpectrum, NOT a sorted ndarray. Use scale_axis
instead of np.diag(s), and use LasyTen masks instead of slicing bond indices.
Only the small 1-D spectrum is copied to CPU for PyTreeNet-compatible cutoff
selection. Tensor contractions and decompositions remain in LasyTen.

LasyTenArray supplies shape/ndim, transpose/conj/copy/astype and scalar algebra
for Node and TensorDict. Its native tensor is kept with all *public* legs in
the codomain. Decompositions explicitly repartition those legs into a map;
the right-hand domain order is reversed to preserve PyTreeNet's NumPy order.
Never modify ``native`` in place behind the wrapper.

Initial scope: ordinary (bosonic) NoSymmetry and U(1). Explicit charge spaces
are required for U(1); from_numpy does not infer virtual charges. ChargedTensor
is supported for a fixed nonzero total charge. Fermionic/anyonic conventions,
arbitrary reshape/slicing, zero-padded KEEP/FULL QR, randomized SVD, and an
automatic migration of all PyTreeNet/BUG/SciPy code are outside this version.
Non-Hermitian Krylov evolution remains disabled: the original Cyten adapter
revision failed an independent ArnoldiEvolution accuracy comparison. That
path has not been revalidated for this LASyTen revision.
Unsupported operations fail rather than silently densifying. In particular,
flatten() does not return a SciPy-compatible dense vector. Native adaptive BUG
uses the tensor-valued RK23/RK45 implementation in time_evolution.py; native
one-site DMRG uses the tensor-valued Davidson implementation in dmrg_utils.py.

No monkey-patching of NumPy or PyTreeNet is performed. ``network_to_lasyten``
converts storage, but does not make unported np.* calls backend compatible.
``to_numpy`` and ``network_to_numpy`` are the explicit dense boundaries.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from numbers import Integral, Number
from typing import Any, Mapping, Protocol, Sequence
from types import SimpleNamespace

import numpy as np

LASYTEN_SOURCE_COMMIT = "cc8d5ab5951ee613cf5a9e3c87beece90a57f29e"


# Native BUG speedups: internal helper binding.
from . import bug_native_speedups as _bug_native_speedups

class TensorLike(Protocol):
    """The structural interface needed by Node's topology bookkeeping."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    def transpose(self, axes: Sequence[int]) -> TensorLike: ...


def as_storage_tensor(value, *, backend="auto"):
    """Normalize tensors once at a TTN storage boundary, without densifying.

    ``auto`` preserves NumPy arrays and existing wrapped LASyTen tensors,
    wrapping raw ``lasyten.Tensor`` objects as needed. ``lasyten`` additionally
    converts NumPy arrays with NoSymmetry; U(1) inputs must be constructed with
    explicit spaces first. ``numpy`` rejects LASyTen inputs instead of silently
    transferring data. The default keeps legacy QNNode tuple inputs unchanged.
    """
    if backend not in ("auto", "numpy", "lasyten"):
        raise ValueError("backend must be 'auto', 'numpy', or 'lasyten'.")
    is_native = type(value).__module__.split(".")[0] == "lasyten"
    if isinstance(value, LasyTenArray) or is_native:
        if backend == "numpy":
            raise TypeError("Use to_numpy() explicitly before storing LASyTen data in a NumPy TTN.")
        return as_lasyten(value)
    if backend == "lasyten":
        if not isinstance(value, np.ndarray):
            raise TypeError("LASyTen storage expects a NumPy array, LasyTenArray, or lasyten.Tensor.")
        return from_numpy(value)
    return value


def storage_spaces(value):
    """Outward leg spaces in raw storage order, or None for NumPy data."""
    return value.legs if isinstance(value, LasyTenArray) else None


def spaces_contractible(left, right):
    """Check representations after the caller has checked bond dimensions."""
    if left is None or right is None:
        return left is None and right is None
    trivial = _supported_symmetries()[0]
    if left.symmetry.is_equivalent_to(trivial) and right.symmetry.is_equivalent_to(trivial):
        # Dense axes have no variance; the wrapper intentionally normalizes it.
        return left.dim == right.dim
    return left == right.dual


@lru_cache(maxsize=1)
def _lasyten():
    try:
        import lasyten
    except ImportError as exc:
        raise ImportError(
            "A working LasyTen installation is required. The inspected revision "
            "requires Python >= 3.14 and its compiled lasyten._core extension."
        ) from exc
    return lasyten


def _symmetry(value):
    return value.as_Symmetry() if hasattr(value, "as_Symmetry") else value


@lru_cache(maxsize=1)
def _supported_symmetries():
    ct = _lasyten()
    return _symmetry(ct.NoSymmetry()), _symmetry(ct.U1()), _symmetry(ct.U1() * ct.U1())


def _no_symmetry(a):
    return a.native.symmetry.is_equivalent_to(_supported_symmetries()[0])


def _scalar(value):
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value).item()


def _axes(axes, ndim):
    values = (axes,) if isinstance(axes, Integral) else tuple(axes)
    result = []
    for axis in values:
        if not isinstance(axis, Integral) or not -ndim <= axis < ndim:
            raise ValueError(f"Invalid axis {axis!r} for rank {ndim}.")
        result.append(int(axis) % ndim)
    if len(set(result)) != len(result):
        raise ValueError("Repeated axes are not allowed.")
    return tuple(result)


def _partition(tensor, left, right):
    a = as_lasyten(tensor)
    left, right = _axes(left, a.ndim), _axes(right, a.ndim)
    if not left or not right:
        raise NotImplementedError("This adapter requires two nonempty groups for decompositions.")
    if sorted(left + right) != list(range(a.ndim)):
        raise ValueError("The left and right groups must partition all tensor axes.")
    if _no_symmetry(a):
        # A dense ndarray has no covariant/contravariant distinction. Build a
        # consistent ordinary matrix map, using a native block (no NumPy
        # transfer). This also makes square NoSymmetry maps endomorphisms.
        ct = _lasyten()
        ordered = ct.permute_legs(a.native, codomain=list(left + right), domain=[])
        spaces = [ct.ElementarySpace.from_trivial_sector(d, symmetry=a.native.symmetry)
                  for d in ordered.shape]
        return ct.SymmetricTensor.from_dense_block(
            ordered.to_dense_block(), codomain=spaces[:len(left)],
            domain=list(reversed(spaces[len(left):])), backend=a.native.backend,
        )
    # LasyTen's public order is codomain + reversed(domain).
    return _lasyten().permute_legs(a.native, codomain=list(left), domain=list(reversed(right)))


def u1_space(charges: Sequence[int], *, dual: bool = False):
    """Create a U(1) leg; charges follow the original, unsorted dense basis.

    For a spinful site use charges=(0, 1, 1, 2). ``dual=True`` returns the
    dual representation (opposite outward charge), not a second dense basis.
    """
    values = np.asarray(charges)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
        raise ValueError("charges must be a nonempty one-dimensional integer sequence.")
    ct = _lasyten()
    leg = ct.ElementarySpace.from_basis(_symmetry(ct.U1()), values.astype(np.int64)[:, None])
    return leg.dual if dual else leg


class LasyTenArray:
    """A LasyTen tensor with a small, explicit PyTreeNet array interface."""

    __array_priority__ = 1000

    def __init__(self, native):
        ct = _lasyten()
        if not isinstance(native, ct.Tensor):
            raise TypeError("LasyTenArray expects a lasyten.Tensor.")
        if not any(native.symmetry.is_equivalent_to(s) for s in _supported_symmetries()):
            raise NotImplementedError("Initial adapter supports bosonic NoSymmetry and U(1) only.")
        if native.num_codomain_legs != native.num_legs:
            native = ct.permute_legs(native, codomain=list(range(native.num_legs)), domain=[])
        if native.symmetry.is_equivalent_to(_supported_symmetries()[0]) and any(leg.is_dual for leg in native.legs):
            # Dense NumPy axes carry no variance. Normalize NoSymmetry axes
            # so QR factors, sums, inner products and original input tensors
            # can be combined without artificial duality mismatches.
            spaces = [leg.dual if leg.is_dual else leg for leg in native.legs]
            native = ct.SymmetricTensor.from_dense_block(
                native.to_dense_block(), codomain=spaces, backend=native.backend,
                labels=native.labels,
            )
        self.native = native

    @property
    def shape(self):
        return tuple(int(d) for d in self.native.shape)

    @property
    def ndim(self):
        return int(self.native.num_legs)

    @property
    def size(self):
        return prod(self.shape)

    @property
    def dtype(self):
        return np.dtype(self.native.dtype.to_numpy_dtype())

    @property
    def legs(self):
        return tuple(self.native.legs)

    @property
    def T(self):
        return transpose(self)

    def to_numpy(self):
        return np.asarray(self.native.to_numpy())

    def copy(self):
        return LasyTenArray(self.native.copy(deep=True))

    def __copy__(self):
        return LasyTenArray(self.native.copy(deep=False))

    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result

    def transpose(self, *axes):
        if not axes:
            return transpose(self)
        return transpose(self, axes[0] if len(axes) == 1 else axes)

    def conj(self):
        return conjugate(self)

    conjugate = conj

    def reshape(self, *shape, order="C"):
        return reshape(self, shape[0] if len(shape) == 1 else shape, order=order)

    def flatten(self, order="C"):
        return reshape(self, (-1,), order=order).copy()

    def astype(self, dtype, copy=True):
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self.copy() if copy else self
        return LasyTenArray(self.native.as_dtype(_lasyten().Dtype.from_numpy_dtype(dtype)))

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Implicit LasyTen -> NumPy conversion is disabled; call to_numpy() explicitly.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            return NotImplemented
        if ufunc is np.conjugate:
            return conjugate(inputs[0])
        if ufunc is np.negative:
            return -inputs[0]
        a, b = inputs if len(inputs) == 2 else (None, None)
        if ufunc is np.multiply:
            if isinstance(a, Number) and isinstance(b, LasyTenArray):
                return b * a
            if isinstance(a, LasyTenArray) and isinstance(b, Number):
                return a * b
            return NotImplemented
        if ufunc is np.true_divide and isinstance(a, LasyTenArray) and isinstance(b, Number):
            return a / b
        if ufunc is np.add and isinstance(a, LasyTenArray) and isinstance(b, LasyTenArray):
            return a + b
        if ufunc is np.subtract and isinstance(a, LasyTenArray) and isinstance(b, LasyTenArray):
            return a - b
        return NotImplemented

    def __add__(self, other):
        if not isinstance(other, LasyTenArray):
            return NotImplemented
        return LasyTenArray(self.native + other.native)

    def __sub__(self, other):
        if not isinstance(other, LasyTenArray):
            return NotImplemented
        return LasyTenArray(self.native - other.native)

    def __mul__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        return LasyTenArray(self.native * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        return LasyTenArray(self.native / other)

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        b = as_lasyten(other)
        if self.ndim != 2 or b.ndim not in (1, 2):
            raise NotImplementedError("@ currently supports a matrix times a vector or matrix.")
        return tensordot(self, b, axes=((-1,), (0,)))

    def __repr__(self):
        return f"LasyTenArray(shape={self.shape}, dtype={self.dtype}, symmetry={self.native.symmetry!r})"


@dataclass(frozen=True)
class LasyTenSpectrum:
    """A diagonal spectrum retaining its graded bond space and ordering."""

    native: Any

    @property
    def leg(self):
        return self.native.leg

    def to_numpy(self):
        """Export the diagonal, without assuming descending order."""
        return np.asarray(self.native.diagonal_as_numpy())

    def __len__(self):
        return int(self.leg.dim)


def from_numpy(array, *, spaces=None, charge=None, block_backend="numpy", device=None, tol=1e-12):
    """Convert dense data once, validating the supplied symmetry spaces.

    ``spaces`` are the OUTWARD representations of the public PyTreeNet axes.
    They default to NoSymmetry spaces. For nonzero charge, supply a one-sector
    charge space, e.g. ``charge=u1_space([N])`` for a vector with N particles.
    LasyTen rejects data outside the specified invariant/charged subspace.
    """
    ct = _lasyten()
    data = np.asarray(array)
    if data.ndim == 0:
        raise ValueError("Use ordinary Python/NumPy scalars for rank-zero inputs.")
    if any(d == 0 for d in data.shape):
        raise ValueError("Zero-dimensional legs are not supported in this initial adapter.")
    if spaces is None:
        symmetry = _symmetry(ct.NoSymmetry())
        spaces = [ct.ElementarySpace.from_trivial_sector(d, symmetry=symmetry) for d in data.shape]
    else:
        spaces = list(spaces)
        if len(spaces) != data.ndim or tuple(int(s.dim) for s in spaces) != data.shape:
            raise ValueError("One compatible space is required per dense tensor axis.")
        symmetry = spaces[0].symmetry
    backend = ct.get_backend(symmetry=symmetry, block_backend=block_backend)
    cls = ct.SymmetricTensor if charge is None else ct.ChargedTensor
    kwargs = dict(codomain=spaces, backend=backend, device=device, tol=tol)
    if charge is not None:
        kwargs["charge"] = charge
    return LasyTenArray(cls.from_dense_block(data, **kwargs))


def as_lasyten(value):
    """Preserve existing LasyTen data; plain arrays enter with NoSymmetry."""
    if isinstance(value, LasyTenArray):
        return value
    if isinstance(value, _lasyten().Tensor):
        return LasyTenArray(value)
    return from_numpy(value)


def to_numpy(value):
    return value.to_numpy() if isinstance(value, (LasyTenArray, LasyTenSpectrum)) else np.asarray(value)


def transpose(a, axes=None):
    a = as_lasyten(a)
    axes = tuple(reversed(range(a.ndim))) if axes is None else _axes(axes, a.ndim)
    if len(axes) != a.ndim:
        raise ValueError("transpose requires every axis exactly once.")
    return LasyTenArray(_lasyten().permute_legs(a.native, codomain=list(axes), domain=[]))


def moveaxis(a, source, destination):
    a = as_lasyten(a)
    source, destination = _axes(source, a.ndim), _axes(destination, a.ndim)
    if len(source) != len(destination):
        raise ValueError("source and destination must have equal lengths.")
    order = [i for i in range(a.ndim) if i not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return transpose(a, order)


def conjugate(a):
    return _bug_native_speedups.conjugate(a)


def tensordot(a, b, axes=2):
    a, b = as_lasyten(a), as_lasyten(b)
    if isinstance(axes, Integral):
        if axes < 0 or axes > min(a.ndim, b.ndim):
            raise ValueError("Invalid contraction rank.")
        ax_a, ax_b = tuple(range(a.ndim - axes, a.ndim)), tuple(range(axes))
    else:
        if len(axes) != 2:
            raise ValueError("axes must be an integer or a pair of axis sequences.")
        ax_a, ax_b = _axes(axes[0], a.ndim), _axes(axes[1], b.ndim)
    if len(ax_a) != len(ax_b):
        raise ValueError("Contraction axis counts differ.")
    ct = _lasyten()
    if (len(ax_a) == a.ndim == b.ndim
            and isinstance(a.native, ct.ChargedTensor)
            and isinstance(b.native, ct.ChargedTensor)):
        # This LASyTen revision's general charged x charged tdot calls an
        # unimplemented state_tensor_product. Its native inner product can
        # close both charge legs directly. Conjugate the first input so this
        # remains a bilinear tensordot, for any complete axis pairing.
        left = conjugate(transpose(a, ax_a))
        right = transpose(b, ax_b)
        return _scalar(ct.inner(left.native, right.native))
    native_b = b.native
    if _no_symmetry(a) and _no_symmetry(b):
        spaces = list(b.legs)
        for left, right in zip(ax_a, ax_b):
            if a.shape[left] != b.shape[right]:
                raise ValueError("Contracted dimensions differ.")
            spaces[right] = a.legs[left].dual
        native_b = _lasyten().SymmetricTensor.from_dense_block(
            b.native.to_dense_block(), codomain=spaces, backend=b.native.backend,
        )
    result = _lasyten().tdot(a.native, native_b, list(ax_a), list(ax_b))
    if isinstance(result, _lasyten().Tensor):
        return LasyTenArray(result) if result.num_legs else _scalar(_lasyten().item(result))
    return _scalar(result)


def vdot(a, b):
    """NumPy convention: conjugate a, not b. This is NOT TTNS.scalar_product's argument order."""
    return _scalar(_lasyten().inner(as_lasyten(a).native, as_lasyten(b).native))


def norm(a):
    return float(_scalar(_lasyten().norm(a.native if isinstance(a, LasyTenSpectrum) else as_lasyten(a).native)))


def zeros_like(a):
    return LasyTenArray(_lasyten().zero_like(as_lasyten(a).native))


def fuse_legs(a, *groups, pipe_dualities=False, pipes=None):
    """Fuse complete public axes; keep LasyTen's LegPipe metadata."""
    a = as_lasyten(a)
    groups = [_axes(g, a.ndim) for g in groups]
    if any(not g for g in groups):
        raise ValueError("Cannot fuse an empty group.")
    flat = [i for group in groups for i in group]
    if len(set(flat)) != len(flat):
        raise ValueError("Fusion groups overlap.")
    return LasyTenArray(_lasyten().combine_legs(
        a.native, *[list(g) for g in groups],
        pipe_dualities=pipe_dualities, pipes=pipes))


def split_legs(a, axes=None):
    a = as_lasyten(a)
    kwargs = {} if axes is None else {"legs": list(_axes(axes, a.ndim))}
    return LasyTenArray(_lasyten().split_legs(a.native, **kwargs))


def reshape(a, shape, *, order="C"):
    """Only fuse consecutive whole axes, or unfuse stored LegPipes; never densify.

    Arbitrary refactorization of a leg would require new charge spaces and
    therefore cannot be inferred from a tuple of integers.
    """
    a = as_lasyten(a)
    if order != "C":
        raise NotImplementedError("Only C-order whole-leg fusion is supported.")
    shape = (int(shape),) if isinstance(shape, Integral) else tuple(int(d) for d in shape)
    if not shape:
        raise NotImplementedError("Use an explicit scalar contraction instead of reshaping to rank zero.")
    if shape.count(-1) > 1 or any(d < -1 or d == 0 for d in shape):
        raise ValueError("Use positive dimensions with at most one -1.")
    if -1 in shape:
        known = prod(d for d in shape if d != -1)
        if a.size % known:
            raise ValueError("Reshape size mismatch.")
        shape = tuple(a.size // known if d == -1 else d for d in shape)
    if prod(shape) != a.size:
        raise ValueError("Reshape size mismatch.")
    if shape == a.shape:
        return a
    if a.native.has_pipes:
        a = split_legs(a)
        if shape == a.shape:
            return a
    groups, start = [], 0
    for target in shape:
        stop, dim = start, 1
        while stop < a.ndim and (dim < target or stop == start):
            dim *= a.shape[stop]
            stop += 1
        if dim != target:
            raise NotImplementedError("This reshape needs a new leg factorization; use explicit spaces/fusion.")
        groups.append(tuple(range(start, stop)))
        start = stop
    if start != a.ndim:
        raise NotImplementedError("This reshape needs explicit handling of trivial legs.")
    return fuse_legs(a, *groups)


def tensor_matricization(a, output_legs, input_legs, correctly_ordered=False):
    """Return a LasyTen matrix with one fused output and one fused input leg."""
    output_legs, input_legs = tuple(output_legs), tuple(input_legs)
    matrix = _partition(a, output_legs, input_legs)
    nleft = len(tuple(output_legs))
    return LasyTenArray(_lasyten().combine_legs(matrix, list(range(nleft)), list(range(nleft, matrix.num_legs))))


def tensor_qr_decomposition(a, q_legs, r_legs, mode="reduced"):
    a = as_lasyten(a)
    q_legs, r_legs = tuple(q_legs), tuple(r_legs)
    q, r = _lasyten().qr(_partition(a, q_legs, r_legs))
    q, r = LasyTenArray(q), LasyTenArray(r)
    mode = str(getattr(mode, "value", mode)).lower()
    if mode not in ("reduced", "keep", "full", "complete"):
        raise ValueError(f"Unknown QR mode {mode!r}.")
    target = prod(a.shape[i] for i in _axes(r_legs, a.ndim)) if mode == "keep" else prod(a.shape[i] for i in _axes(q_legs, a.ndim))
    if mode != "reduced" and q.shape[-1] != target:
        raise NotImplementedError("LasyTen QR is reduced; this KEEP/FULL request needs a charge-aware bond embedding.")
    return q, r


def tensor_svd(a, u_legs, v_legs, mode="reduced", decomp="svd"):
    mode = str(getattr(mode, "value", mode)).lower()
    decomp = str(getattr(decomp, "value", decomp)).lower()
    if mode != "reduced" or decomp != "svd":
        raise NotImplementedError("Initial adapter supports reduced, exact SVD only.")
    u, s, vh = _lasyten().svd(_partition(a, u_legs, v_legs))
    return LasyTenArray(u), LasyTenSpectrum(s), LasyTenArray(vh)


def truncated_tensor_svd(a, u_legs, v_legs, svd_params):
    """Match upstream PyTreeNet's SVDParameters for NoSymmetry/U(1).

    Selection is global across sectors, in descending singular-value order,
    but the returned native spectrum stays in LasyTen bond-basis order. The
    PyTreeNet ``renorm`` option preserves SUM(s), as its implementation does,
    rather than silently substituting LasyTen's two-norm normalization.
    """
    if getattr(svd_params, "random", False):
        raise NotImplementedError("Randomized decompositions are not implemented here.")
    decomp = getattr(svd_params, "decomp", "svd")
    if str(getattr(decomp, "value", decomp)).lower() != "svd":
        raise NotImplementedError("Only SVD truncation is supported.")
    u, spectrum, vh = tensor_svd(a, u_legs, v_legs)
    values = spectrum.to_numpy()
    order = np.argsort(-values, kind="stable")
    sorted_s = values[order]
    limit = svd_params.max_bond_dim
    if limit <= 0 or (np.isfinite(limit) and int(limit) != limit):
        raise ValueError("max_bond_dim must be a positive integer or infinity.")
    if getattr(svd_params, "sum_trunc", False):
        norm_sq = float(np.sum(sorted_s ** 2))
        count, discarded = 0, 0.0
        for i in range(len(sorted_s) - 1, -1, -1):
            discarded += sorted_s[i] ** 2
            compare = discarded / norm_sq if getattr(svd_params, "sum_renorm", False) and norm_sq else discarded
            if compare > svd_params.total_tol ** 2:
                count = i + 1
                break
    else:
        cutoff = max(svd_params.total_tol, svd_params.rel_tol * sorted_s[0])
        count = int(np.count_nonzero(sorted_s > cutoff))
    count = max(1, min(count, int(limit) if np.isfinite(limit) else len(values)))
    keep = np.zeros(len(values), dtype=bool)
    keep[order[:count]] = True
    ct = _lasyten()
    mask = ct.Mask.from_block_mask(keep, large_leg=spectrum.leg,
                                  backend=spectrum.native.backend, device=spectrum.native.device)
    # svd_apply_mask expects the original map signatures: U's bond is in
    # its domain, Vh's bond is in its codomain. Wrappers store public legs
    # in the codomain, so restore those signatures before applying masks.
    u_map = _partition(u, tuple(range(u.ndim - 1)), (-1,))
    v_map = _partition(vh, (0,), tuple(range(1, vh.ndim)))
    un, sn, vn = ct.tensors.svd_apply_mask(u_map, spectrum.native, v_map, mask)
    if getattr(svd_params, "renorm", False):
        kept_sum = float(np.sum(values[keep]))
        if kept_sum == 0:
            raise ValueError("Cannot renormalize a zero singular spectrum.")
        sn = sn * float(np.sum(values) / kept_sum)
    return LasyTenArray(un), LasyTenSpectrum(sn), LasyTenArray(vn)


def scale_axis(a, spectrum, axis):
    if not isinstance(spectrum, LasyTenSpectrum):
        raise TypeError("Expected a LasyTenSpectrum; a dense diagonal loses the bond charge metadata.")
    a = as_lasyten(a)
    axis = _axes(axis, a.ndim)[0]
    return LasyTenArray(_lasyten().scale_axis(a.native, spectrum.native, axis))


def contr_truncated_svd_splitting(a, u_legs, v_legs, contr_mode="vcontr", svd_params=None):
    """PyTreeNet's split-and-absorb operation, preserving the graded bond."""
    if svd_params is None:
        svd_params = SimpleNamespace(max_bond_dim=100, rel_tol=1e-15,
                                    total_tol=1e-15, renorm=False,
                                    sum_trunc=False, sum_renorm=False)
    u, s, v = truncated_tensor_svd(a, u_legs, v_legs, svd_params)
    mode = str(getattr(contr_mode, "value", contr_mode)).lower()
    if mode == "vcontr":
        return u, scale_axis(v, s, 0)
    if mode == "ucontr":
        return scale_axis(u, s, -1), v
    if mode == "equal":
        root = LasyTenSpectrum(_lasyten().sqrt(s.native))
        return scale_axis(u, root, -1), scale_axis(v, root, 0)
    raise ValueError(f"Unknown SVD contraction mode {mode!r}.")


def eigh(a, output_legs=(0,), input_legs=(1,)):
    """Hermitian eigenproblem; eigenvalues retain their LasyTen bond ordering."""
    values, vectors = _lasyten().eigh(_partition(a, output_legs, input_legs), new_labels=[None, None, None])
    return LasyTenSpectrum(values), LasyTenArray(vectors)


def krylov_expm_action(hamiltonian, state, dt, *, hermitian=True, options=None):
    """Compute exp(-i H dt)|state> using native LasyTen Krylov, without normalization.

    Initial interface: one matrix leg each for input/output, and a one-leg
    vector. Fuse local tensor legs explicitly before calling; preserve the
    LegPipe for splitting afterward. H is never converted to a dense matrix.
    Complex dt is allowed. H must be Hermitian and uses Lanczos. The original Cyten
    ArnoldiEvolution path failed an independent accuracy check, and is deliberately disabled rather than exposed silently.
    """
    h, state = as_lasyten(hamiltonian), as_lasyten(state)
    if h.ndim != 2 or state.ndim != 1:
        raise ValueError("Supply a two-leg Hamiltonian and a one-leg state.")
    if not hermitian:
        raise NotImplementedError(
            "Non-Hermitian propagation is disabled: native LasyTen ArnoldiEvolution "
            "has not been revalidated against dense-reference evolution."
        )
    ct = _lasyten()
    if h.shape[0] != h.shape[1] or state.shape[0] != h.shape[1]:
        raise ValueError("The Hamiltonian must be square and match the state dimension.")
    h_map = _partition(h, (0,), (1,))
    if not ct.almost_equal(h_map, ct.dagger(h_map), rtol=1e-12, atol=1e-12):
        raise ValueError("This propagation path requires a Hermitian Hamiltonian.")
    operator = ct.sparse.TensorLinearOperator(h_map)
    engine = ct.krylov_based.LanczosEvolution(operator, state.native, {} if options is None else dict(options))
    evolved, _ = engine.run(-1j * dt, normalize=False)
    return LasyTenArray(evolved)


def concatenate(arrays, axis=0):
    """Direct-sum an existing axis through LasyTen's tensor_from_grid.

    This is the primitive for BUG's old/new basis concatenation. All other
    spaces must agree. A rank-one concatenation needs a separate trivial leg
    and is deliberately not inferred by this initial adapter.
    """
    arrays = [as_lasyten(a) for a in arrays]
    if not arrays:
        raise ValueError("No arrays to concatenate.")
    first = arrays[0]
    axis = _axes(axis, first.ndim)[0]
    if first.ndim < 2:
        raise NotImplementedError("Concatenation currently requires at least two tensor axes.")
    others = tuple(i for i in range(first.ndim) if i != axis)
    maps = []
    for a in arrays:
        if a.ndim != first.ndim or any(a.shape[i] != first.shape[i] for i in others):
            raise ValueError("Non-concatenated dimensions must match.")
        maps.append(_partition(a, (axis,), others))
    ct = _lasyten()
    native = ct.tensor_from_grid([[m] for m in maps])
    # A grid stacks two axes. Its single column must not leave a one-summand
    # DirectSumSpace on an otherwise unchanged physical/environment leg.
    if isinstance(native.legs[1], ct.DirectSumSpace):
        native = ct.grid_project(native, legs=[1], cells=[0])
    result = LasyTenArray(native)
    inverse = np.argsort((axis,) + others).tolist()
    return transpose(result, inverse)


def network_to_lasyten(network, *, spaces: Mapping | None = None, charges: Mapping | None = None,
                     block_backend="numpy", device=None):
    """Deep-copy a PyTreeNet TTNS/TTNO and convert its node tensors.

    ``spaces[node_id]`` follows that node's *current public tensor order*,
    not its raw TensorDict storage order. ``charges`` optionally assigns a
    LasyTen charge space to individual nodes. The original network is unchanged.
    Existing ordinary Node topology, leg permutations and caches are retained;
    QNNode's tuple-based API is not supported by this storage conversion.
    """
    result = deepcopy(network)
    if hasattr(result.tensors, "backend"):
        result.tensors.backend = "lasyten"
    if spaces is not None and set(spaces) != set(result.nodes):
        raise ValueError("Supply spaces for every node, or omit spaces for NoSymmetry.")
    charges = {} if charges is None else charges
    if set(charges) - set(result.nodes):
        raise ValueError("Charge mapping contains unknown nodes.")
    # Reading result[node_id] realizes pending leg permutations on the copy.
    for node_id in list(result.nodes):
        node, data = result[node_id]
        if hasattr(node, "qn"):
            raise NotImplementedError("Use ordinary Nodes; LasyTen's leg spaces own the charge metadata.")
        converted = as_lasyten(data) if isinstance(data, LasyTenArray) else from_numpy(
            data, spaces=None if spaces is None else spaces[node_id],
            charge=charges.get(node_id), block_backend=block_backend, device=device,
        )
        node.link_tensor(converted)
        result.tensors[node_id] = converted
    if spaces is not None:
        for node_id, node in result.nodes.items():
            parent_tensor = result.tensors[node_id]
            if _no_symmetry(parent_tensor):
                continue
            for child_id in node.children:
                child_tensor = result.tensors[child_id]
                parent_axis = node.neighbour_index(child_id)
                child_axis = result.nodes[child_id].parent_leg
                if parent_tensor.legs[parent_axis] != child_tensor.legs[child_axis].dual:
                    raise ValueError(f"Incompatible LasyTen charge spaces on bond {node_id!r} -- {child_id!r}.")
    return result


def network_to_numpy(network):
    """Explicitly export a copied LasyTen-backed network for reference calculations."""
    result = deepcopy(network)
    if hasattr(result.tensors, "backend"):
        result.tensors.backend = "numpy"
    for node_id in list(result.nodes):
        node, data = result[node_id]
        dense = to_numpy(data).copy()
        node.link_tensor(dense)
        result.tensors[node_id] = dense
    return result


__all__ = [
    "TensorLike", "as_storage_tensor", "storage_spaces", "spaces_contractible",
    "LASYTEN_SOURCE_COMMIT", "LasyTenArray", "LasyTenSpectrum", "as_lasyten", "from_numpy",
    "to_numpy", "u1_space", "transpose", "moveaxis", "conjugate", "tensordot", "vdot",
    "norm", "zeros_like", "fuse_legs", "split_legs", "reshape", "tensor_matricization",
    "tensor_qr_decomposition", "tensor_svd", "truncated_tensor_svd", "scale_axis",
    "contr_truncated_svd_splitting",
    "eigh", "krylov_expm_action", "concatenate", "network_to_lasyten", "network_to_numpy",
]


def _smoke_test():
    """Small real-LasyTen checks, runnable without installing pytest."""
    rng = np.random.default_rng(11)
    dense = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    a = from_numpy(dense)
    np.testing.assert_allclose(a.conj().to_numpy(), dense.conj())
    q, r = tensor_qr_decomposition(a, (2, 0), (1,))
    np.testing.assert_allclose(tensordot(q, r, axes=1).to_numpy(), dense.transpose(2, 0, 1), atol=1e-11)
    u, s, v = tensor_svd(a, (2, 0), (1,))
    np.testing.assert_allclose(tensordot(scale_axis(u, s, -1), v, axes=1).to_numpy(), dense.transpose(2, 0, 1), atol=1e-11)
    p = u1_space([0, 1, 1, 2])
    matrix = np.diag([1., 2., 3., 4.]).astype(complex)
    h = from_numpy(matrix, spaces=(p, p.dual))
    np.testing.assert_allclose(tensordot(h, h, axes=1).to_numpy(), matrix @ matrix)
    print("LasyTen adapter smoke tests passed: conjugation, axis order, QR, SVD, U(1) contraction.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run smoke tests against your LasyTen installation.")
    args = parser.parse_args()
    if args.self_test:
        _smoke_test()
    else:
        parser.print_help()

def u1xu1_space(charges, *, dual=False):
    """Create a leg from (N_up,N_down) charges in dense basis order."""
    from .abelian_charges import charge_space
    return charge_space(charges, rank=2, dual=dual)

