# Adapted from linalg_helper.py in pyscf (Apache License 2.0)
# Source: https://github.com/pyscf/pyscf/blob/master/pyscf/lib/linalg_helper.py

"""Hermitian Davidson with NumPy vectors or native LASyTen tensors.

Install as pytreenet_lasyten/dmrg/dmrg_utils.py. Relative imports also support
an installation named pytreenet. LASyTen guesses may have any tensor rank;
never flatten them. aop and a callable preconditioner must return tensors with
exactly the same public leg spaces and charge sector as their inputs.

Full tensor inner products/norms use lasyten_backend; basis vectors stay in
Python lists and Ritz vectors are assembled by native tensor algebra. The
small projected Hamiltonian, Ritz coefficients, eigenvalues and convergence
flags remain NumPy/SciPy. No symmetry-block packing is required for this path.

Use precond=None or lambda residual, energy, vector: residual initially.
NumPy diagonal preconditioners are supported only for NumPy vectors.
Only in-core basis storage is provided (as in this supplied file); a basis
estimate exceeding max_memory raises MemoryError rather than using the
undefined _Xlist from the original partial PySCF extraction.
"""

import numpy
import scipy.linalg

from ..util import lasyten_backend as lb
from ..util.tensor_dispatch import is_lasyten_tensor, conjugate

SAFE_EIGH_LINDEP =  1e-15
DAVIDSON_LINDEP = 1e-14
DSOLVE_LINDEP = 1e-13
MAX_MEMORY = 4000  # 4GB

# sort by similarity has problem which flips the ordering of eigenvalues when
# the initial guess is closed to excited state.  In this situation, function
# _sort_by_similarity may mark the excited state as the first eigenvalue and
# freeze the first eigenvalue.
SORT_EIG_BY_SIMILARITY = False
# Projecting out converged eigenvectors has problems when conv_tol is loose.
# In this situation, the converged eigenvectors may be updated in the
# following iterations.  Projecting out the converged eigenvectors may lead to
# large errors to the yet converged eigenvectors.
PROJECT_OUT_CONV_EIGS = False

FOLLOW_STATE = False

class LinearDependenceError(RuntimeError):
    """Davidson could not construct a linearly independent basis."""
    pass


def _as_vector(value):
    """Preserve native tensor spaces; NumPy conversion is for NumPy vectors only."""
    return lb.as_lasyten(value) if is_lasyten_tensor(value) else numpy.asarray(value)


def _vector_dtype(value):
    return _as_vector(value).dtype


def _scale_vector(value, coefficient):
    """Scale invariant data while retaining a ChargedTensor's hidden state.

    Native ChargedTensor scalar multiplication instead scales charged_state;
    a zero state then makes a subsequent native linear combination divide by
    zero in the targeted LASyTen revision. Zero invariant data is safe.
    """
    if is_lasyten_tensor(value):
        value = lb.as_lasyten(value)
        ct = lb._lasyten()
        if isinstance(value.native, ct.ChargedTensor):
            return lb.LasyTenArray(ct.ChargedTensor(
                value.native.invariant_part * coefficient,
                value.native.charged_state,
            ))
    return value * coefficient


def _inner(left, right, dot=numpy.dot):
    """Hermitian inner product, with the legacy bilinear custom-dot convention."""
    if is_lasyten_tensor(left) or is_lasyten_tensor(right):
        if not (is_lasyten_tensor(left) and is_lasyten_tensor(right)):
            raise TypeError("Davidson vectors must all use the same tensor backend.")
        if dot is numpy.dot:
            # vdot already conjugates its first argument. Do not conjugate twice.
            return lb.vdot(left, right)
        return dot(conjugate(left), right)
    return dot(numpy.conjugate(left), right)


def _norm(value, dot=numpy.dot):
    if is_lasyten_tensor(value) and dot is numpy.dot:
        result = lb.norm(value)
    else:
        result = float(max(0., _inner(value, value, dot).real)) ** .5
    if not numpy.isfinite(result):
        raise ValueError("Davidson encountered a vector with a nonfinite norm.")
    return result


def _vector_dimension(value):
    if is_lasyten_tensor(value):
        native = lb.as_lasyten(value).native
        if isinstance(native, lb._lasyten().ChargedTensor):
            native = native.invariant_part
        # Includes all symmetry-allowed coefficients, even initially zero blocks.
        return int(native.num_parameters)
    return int(numpy.asarray(value).size)


def _vector_nbytes(value):
    if is_lasyten_tensor(value):
        return _vector_dimension(value) * _vector_dtype(value).itemsize
    return numpy.asarray(value).nbytes


def _check_vector(reference, value, where):
    """Check the fixed local variational space after an action/preconditioner."""
    if is_lasyten_tensor(reference) != is_lasyten_tensor(value):
        raise TypeError(f"{where} changed the tensor backend.")
    reference, value = _as_vector(reference), _as_vector(value)
    if reference.shape != value.shape:
        raise ValueError(f"{where} changed the vector shape: {reference.shape} -> {value.shape}.")
    if is_lasyten_tensor(reference):
        if reference.legs != value.legs:
            raise ValueError(f"{where} changed the LASyTen leg spaces.")
        cls = lb._lasyten().ChargedTensor
        charged = isinstance(reference.native, cls)
        if charged != isinstance(value.native, cls):
            raise ValueError(f"{where} changed the tensor charge representation.")
        if charged and reference.native.charge_leg != value.native.charge_leg:
            raise ValueError(f"{where} changed the total charge sector.")
    return value


def make_diag_precond(diagonal, level_shift=1e-3):
    """NumPy diagonal preconditioner; native tensors require a callable instead."""
    if is_lasyten_tensor(diagonal):
        raise TypeError("Supply a callable LASyTen preconditioner, e.g. lambda r, e, x: r.")
    diagonal = numpy.asarray(diagonal)

    def precond(residual, energy, vector):
        if is_lasyten_tensor(residual):
            raise TypeError("A NumPy diagonal cannot precondition a LASyTen tensor; supply a callable.")
        denominator = diagonal - (energy - level_shift)
        denominator = numpy.where(abs(denominator) < 1e-8, 1e-8, denominator)
        return residual / denominator
    return precond


def _fill_heff_hermitian(heff, xs, ax, xt, axt, dot):
    nrow = len(axt)
    row1 = len(ax)
    row0 = row1 - nrow
    for ip, i in enumerate(range(row0, row1)):
        for jp, j in enumerate(range(row0, i)):
            heff[i,j] = _inner(xt[ip], axt[jp], dot)
            heff[j,i] = heff[i,j].conj()
        heff[i,i] = _inner(xt[ip], axt[ip], dot).real

    for i in range(row0):
        axi = _as_vector(ax[i])
        for jp, j in enumerate(range(row0, row1)):
            heff[j,i] = _inner(xt[jp], axi, dot)
            heff[i,j] = heff[j,i].conj()
        axi = None
    return heff

def _fill_heff(heff, xs, ax, xt, axt, dot):
    nrow = len(axt)
    row1 = len(ax)
    row0 = row1 - nrow
    for ip, i in enumerate(range(row0, row1)):
        for jp, j in enumerate(range(row0, row1)):
            heff[i,j] = _inner(xt[ip], axt[jp], dot)

    for i in range(row0):
        axi = _as_vector(ax[i])
        xi = _as_vector(xs[i])
        for jp, j in enumerate(range(row0, row1)):
            heff[i,j] = _inner(xi, axt[jp], dot)
            heff[j,i] = _inner(xt[jp], axi, dot)
        axi = xi = None
    return heff

def _qr(xs, dot, lindep=1e-14):
    """Orthogonalize a list of Davidson vectors; this is not a tensor-leg QR.

    Native tensors remain a Python list. R contains only scalar coefficients
    and satisfies xs[i] = sum(qs[j] * R[j, i]) up to discarded dependence.
    """
    if any(is_lasyten_tensor(x) for x in xs):
        if not all(is_lasyten_tensor(x) for x in xs):
            raise TypeError("Davidson guesses must all use the same tensor backend.")
        xs = [_check_vector(xs[0], x, "Initial guess").copy() for x in xs]
        qs = []
        dtype = numpy.result_type(*(_vector_dtype(x) for x in xs))
        rmat = numpy.zeros((len(xs), len(xs)), dtype=dtype)
        for column, xi in enumerate(xs):
            if _norm(xi, dot)**2 <= lindep:
                continue
            # Reorthogonalize to keep the projected problem well conditioned.
            for _ in range(2):
                for row, qi in enumerate(qs):
                    coefficient = _inner(qi, xi, dot)
                    xi = xi - _scale_vector(qi, coefficient)
                    rmat[row, column] += coefficient
            norm = _norm(xi, dot)
            if norm**2 > lindep:
                rmat[len(qs), column] = norm
                qs.append(_scale_vector(xi, 1 / norm))
        return qs, rmat[:len(qs)]

    # Original NumPy vector path.
    nvec = len(xs)
    xs = qs = numpy.array(xs, copy=True)
    rmat = numpy.eye(nvec, order='F', dtype=xs.dtype)
    nv = 0
    for i in range(nvec):
        xi = xs[i]
        for j in range(nv):
            coefficient = _inner(qs[j], xi, dot)
            xi -= qs[j] * coefficient
            rmat[:, nv] -= rmat[:, j] * coefficient
        innerprod = _inner(xi, xi, dot).real
        norm = numpy.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi / norm
            rmat[:nv+1, nv] /= norm
            nv += 1
    return qs[:nv], numpy.linalg.inv(rmat[:nv, :nv])

def _outprod_to_subspace(v, xs):
    """Combine full vectors using the small projected eigenvector matrix v."""
    ndim = v.ndim
    if ndim == 1:
        v = v[:, None]
    space, nroots = v.shape
    if any(is_lasyten_tensor(x) for x in xs[:space]):
        if not all(is_lasyten_tensor(x) for x in xs[:space]):
            raise TypeError("Davidson basis vectors must use one tensor backend.")
        result = []
        for root in range(nroots):
            vector = _scale_vector(xs[space-1], v[space-1, root])
            for i in reversed(range(space-1)):
                vector = vector + _scale_vector(xs[i], v[i, root])
            result.append(vector)
        return result[0] if ndim == 1 else result
    x0 = numpy.einsum('c,x->cx', v[space-1], numpy.asarray(xs[space-1]))
    for i in reversed(range(space-1)):
        xsi = numpy.asarray(xs[i])
        for k in range(nroots):
            x0[k] += v[i, k] * xsi
    return x0[0] if ndim == 1 else x0

_gen_x0 = _outprod_to_subspace

def _normalize_xt_(xt, xs, threshold, dot):
    """Project corrections off the basis, retaining native tensor objects."""
    xt = [_as_vector(x).copy() for x in xt if x is not None]
    for _ in range(2):
        for basis in xs:
            for i, vector in enumerate(xt):
                xt[i] = vector - _scale_vector(basis, _inner(basis, vector, dot))
    norm_min = 1.
    out = []
    for vector in xt:
        norm = _norm(vector, dot)
        if norm**2 > threshold:
            out.append(_scale_vector(vector, 1 / norm))
            norm_min = min(norm_min, norm)
    return out, norm_min


def _sort_elast(elast, conv_last, vlast, v, log):
    '''
    Eigenstates may be flipped during the Davidson iterations.  Reorder the
    eigenvalues of last iteration to make them comparable to the eigenvalues
    of the current iterations.
    '''
    head, nroots = vlast.shape
    ovlp = abs(numpy.dot(v[:head].conj().T, vlast))
    mapping = numpy.argmax(ovlp, axis=1)
    found = numpy.any(ovlp > .5, axis=1)

    # if log.verbose >= logger.DEBUG:
    #     ordering_diff = (mapping != numpy.arange(len(mapping)))
    #     if any(ordering_diff & found):
    #         log.debug('Old state -> New state')
    #         for i in numpy.where(ordering_diff)[0]:
    #             log.debug('  %3d     ->   %3d ', mapping[i], i)

    conv = conv_last[mapping]
    e = elast[mapping]
    conv[~found] = False
    e[~found] = 0.
    return e, conv

def davidson(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
             lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
             dot=numpy.dot, callback=None,
             nroots=1, lessio=False, pick=None, verbose=0,
             follow_state=FOLLOW_STATE):
    r'''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function(x) => array_like_x
            aop(x) to mimic the matrix vector multiplication :math:`\sum_{j}a_{ij}*x_j`.
            NumPy inputs/outputs are 1D vectors; native inputs/outputs are tensors.
        x0 : NumPy vector, LASyTen tensor, or list of guesses
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iteration
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        e : float or list of floats
            Eigenvalue.  By default it's one float number.  If :attr:`nroots` > 1, it
            is a list of floats for the lowest :attr:`nroots` eigenvalues.
        c : 1D array or list of 1D arrays
            Eigenvector.  By default it's a 1D array.  If :attr:`nroots` > 1, it
            is a list of arrays for the lowest :attr:`nroots` eigenvectors.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10))
    >>> a = a + a.T
    >>> aop = lambda x: numpy.dot(a,x)
    >>> precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    >>> x0 = a[0]
    >>> e, c = lib.davidson(aop, x0, precond)
    '''
    e, x = davidson1(lambda xs: [aop(x) for x in xs],
                     x0, precond, tol, max_cycle, max_space, lindep,
                     max_memory, dot, callback, nroots, lessio, pick, verbose,
                     follow_state)[1:]
    if nroots == 1:
        return e[0], x[0]
    else:
        return e, x

def davidson1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
              lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
              dot=numpy.dot, callback=None,
              nroots=1, lessio=False, pick=None, verbose=0,
              follow_state=FOLLOW_STATE, tol_residual=None,
              fill_heff=_fill_heff_hermitian):
    r'''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function([x]) => [array_like_x]
            Matrix vector multiplication :math:`y_{ki} = \sum_{j}a_{ij}*x_{jk}`.
        x0 : NumPy vector, LASyTen tensor, or list of guessess or a function to generate x0 array(s)
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iteration
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        conv : bool
            Converged or not
        e : list of floats
            The lowest :attr:`nroots` eigenvalues.
        c : list of 1D arrays
            The lowest :attr:`nroots` eigenvectors.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10))
    >>> a = a + a.T
    >>> aop = lambda xs: [numpy.dot(a,x) for x in xs]
    >>> precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    >>> x0 = a[0]
    >>> e, c = lib.davidson(aop, x0, precond, nroots=2)
    >>> len(e)
    2
    '''
    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual

    if max_cycle < 1 or nroots < 1 or max_space < 1:
        raise ValueError("max_cycle, nroots, and max_space must be positive.")
    if SORT_EIG_BY_SIMILARITY:
        raise NotImplementedError("Similarity sorting was not included in this Davidson implementation.")
    if precond is None:
        precond = lambda residual, energy, vector: residual
    elif not callable(precond):
        precond = make_diag_precond(precond)

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if is_lasyten_tensor(x0) or (isinstance(x0, numpy.ndarray) and x0.ndim == 1):
        x0 = [x0]
    else:
        x0 = list(x0)
    if not x0:
        raise LinearDependenceError("Initial guess is empty or zero")
    reference = _as_vector(x0[0])
    x0 = [_check_vector(reference, value, "Initial guess").copy() for value in x0]
    max_space = max(max_space + (nroots-1) * 4, len(x0), nroots)
    # An upper bound from all allowed block coefficients; no tensor export.
    vector_bytes = max(1, _vector_nbytes(reference))
    _incore = max_memory * 1e6 / vector_bytes > max_space * 2 + nroots * 3
    if not _incore:
        raise MemoryError("Davidson basis estimate exceeds max_memory; out-of-core storage is unavailable.")
    dtype = None
    heff = None
    fresh_start = True
    e = None
    v = None
    conv = numpy.zeros(nroots, dtype=bool)

    for icyc in range(max_cycle):
        if fresh_start:
            xs = []
            ax = []
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt = _qr(x0, dot, lindep)[0]
            if len(xt) != x0len:
                print(f"QR decomposition removed {x0len - len(xt)} vectors.")
                if callable(pick):
                    print(f"Check whether `pick` function {pick.__name__} is providing "
                          "linearly dependent vectors.")
                if len(xt) == 0:
                    if icyc == 0:
                        msg = 'Initial guess is empty or zero'
                    else:
                        msg = ('No more linearly independent basis were found. '
                               'Unless loosen the lindep tolerance (current value '
                               f'{lindep}), the diagonalization solver is not able '
                               'to find eigenvectors.')
                    raise LinearDependenceError(msg)
            x0 = None
            max_dx_last = 1e9
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt = list(aop(xt))
        if len(axt) != len(xt):
            raise ValueError("aop must return one vector per trial vector.")
        axt = [_check_vector(reference, value, "Hamiltonian action") for value in axt]
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space+rnow

        if not axt:
            raise LinearDependenceError('No linearly independent basis found by the diagonalization solver.')
        # Read dtype metadata instead of coercing full tensors to arrays.
        cycle_dtype = numpy.result_type(*(_vector_dtype(x) for x in [*xt, *axt]))
        dtype = cycle_dtype if dtype is None else numpy.result_type(dtype, cycle_dtype)
        if heff is None:  # Lazy initialize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        fill_heff(heff, xs, ax, xt, axt, dot)
        xt = axt = None
        w, v = scipy.linalg.eigh(heff[:space,:space])
        if callable(pick):
            w, v, idx = pick(w, v, nroots, locals())
            if len(w) == 0:
                raise RuntimeError(f'Not enough eigenvalues found by {pick}')

        e = w[:nroots]
        v = v[:,:nroots]
        conv = numpy.zeros(e.size, dtype=bool)
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v, verbose)

        if elast is None:
            de = e
        elif elast.size != e.size:
            print(f"Number of roots different from the previous step ({e.size},{elast.size})")
            de = e
        else:
            de = e - elast

        x0 = None
        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = list(aop(x0))
            if len(ax0) != len(x0):
                raise ValueError("aop must return one vector per Ritz vector.")
            ax0 = [_check_vector(reference, value, "Hamiltonian action") for value in ax0]
        else:
            ax0 = _gen_x0(v, ax)

        dx_norm = numpy.zeros(e.size)
        xt = [None] * nroots
        for k, ek in enumerate(e):
            xt[k] = ax0[k] - _scale_vector(x0[k], ek)
            dx_norm[k] = _norm(xt[k], dot)
            conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose

        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots+2):
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        # remove subspace linear dependency
        for k, ek in enumerate(e):
            if (not conv[k]) and dx_norm[k]**2 > lindep:
                xt[k] = _check_vector(reference, precond(xt[k], ek, x0[k]), "Preconditioner")
                norm = _norm(xt[k], dot)
                xt[k] = _scale_vector(xt[k], 1 / norm) if norm > 0 else None
            elif not conv[k]:
                # Remove linearly dependent vector
                xt[k] = None
            else:
                xt[k] = None

        xt, norm_min = _normalize_xt_(xt, xs, lindep, dot)
        if len(xt) == 0:
            conv = dx_norm < toloose
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    x0 = list(x0)  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = _vector_dimension(x0[0])
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        print(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    return numpy.asarray(conv), e, x0
