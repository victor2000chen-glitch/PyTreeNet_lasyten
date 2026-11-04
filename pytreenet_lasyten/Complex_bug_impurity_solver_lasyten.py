"""Helper functions used to run Complex Time Krylov subspace methods using LASYTEN backend
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import threading
import time

import numpy as np
import psutil
from scipy.integrate import simpson

import pytreenet_lasyten as ptn
from pytreenet_lasyten.core.node import Node
from pytreenet_lasyten.ttns import TreeTensorNetworkState
from pytreenet_lasyten.operators import Hamiltonian, TensorProduct
from pytreenet_lasyten.ttns.ttns_ttno.application import (
    apply_ttno_to_ttns, ApplicationMethod,
)
from pytreenet_lasyten.time_evolution.bug import BUG, BUGConfig
from pytreenet_lasyten.time_evolution.fixed_bug import FixedBUG, FixedBUGConfig
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
from pytreenet_lasyten.dmrg.dmrg_sparse import DMRGAlgorithm_sparse
from pytreenet_lasyten.util.tensor_splitting import SplitMode, SVDParameters
from pytreenet_lasyten.core.truncation.svd_truncation import svd_truncation
from pytreenet_lasyten.util import lasyten_backend as lb
from pytreenet_lasyten.util.tensor_dispatch import is_lasyten_tensor
from pytreenet_lasyten.util.u1_t3ns import random_u1_state_like, check_u1_state
from pytreenet_lasyten.util.u1_ttno import u1_ttno_from_hamiltonian
from pytreenet_lasyten.util.u1_excitations import (
    create_impurity_excitations, apply_local_charge_operator,
)

HK_HELPER_API_VERSION = "2026.11.04"

DMRGAlgorithm = DMRGAlgorithm_sparse

def _native_module():
    return lb._lasyten()

def BUG_time_evolution(Tree_state, TTNO_Hamil, time_step_size, final_time, operators, max_bond_dim=20, rel_tol=1e-16, abs_tol=1e-16, time_evo_mode=TimeEvoMode.RK23):
    """Original helper interface; use the native BUG class and local solver."""
    config = BUGConfig(max_bond_dim=max_bond_dim, rel_tol=rel_tol, total_tol=abs_tol, time_evo_mode=time_evo_mode, record_norm=False, record_max_bdim=True, record_average_bdim=False, record_total_size=False, record_loschmidt_amplitude=False)
    bug = BUG(Tree_state, TTNO_Hamil, time_step_size, final_time, operators, config=config, solver_options={'rtol': 1e-08, 'atol': 1e-10})
    bug.run()
    return (bug, bug.results.results, bug.results.times())

def BUG_time_evolution_fixed(Tree_state, TTNO_Hamil, time_step_size, final_time, operators, max_bond_dim=20, rel_tol=1e-16, abs_tol=1e-16, time_evo_mode=TimeEvoMode.RK23):
    """Native fixed-rank BUG; the existing bond spaces determine the rank.

    max_bond_dim and SVD tolerances remain accepted for call compatibility;
    fixed-rank BUG does not perform adaptive SVD truncation.
    """
    config = FixedBUGConfig(time_evo_mode=time_evo_mode, record_norm=False, record_max_bdim=True, record_average_bdim=False, record_total_size=False, record_loschmidt_amplitude=False)
    bug = FixedBUG(Tree_state, TTNO_Hamil, time_step_size, final_time, operators, config=config, solver_options={'rtol': 1e-08, 'atol': 1e-10})
    bug.run()
    return (bug, bug.results.results, bug.results.times())

def imaginary_time_evolution_TTN(tree_state, ttno_imaginary, ttno_physical, tau, dtau, max_bond_dim):
    if tau <= 0 or dtau <= 0:
        raise ValueError('tau and dtau must be positive')
    n_segments = int(np.ceil(tau / dtau))
    steps_per_segment = 10
    time_step = tau / (n_segments * steps_per_segment)
    state = deepcopy(tree_state)
    state.normalise()
    config = BUGConfig(max_bond_dim=max_bond_dim, rel_tol=1e-05, total_tol=1e-05, time_evo_mode=TimeEvoMode.RK23, record_max_bdim=False)
    bug = BUG(initial_state=state, hamiltonian=ttno_imaginary, time_step_size=time_step, final_time=tau, operators={}, config=config)
    energies = []
    for _ in range(n_segments):
        for _ in range(steps_per_segment):
            bug.run_one_time_step()
        bug.state.normalise()
        energy = float(np.real(bug.state.operator_expectation_value(ttno_physical)))
        energies.append(energy)
    ground_state = bug.state
    E_0 = energies[-1]
    print('Physical ground-state energy:', E_0)
    print('Energy by segment:', energies)
    return (ground_state, E_0)

def Grams_schmidt(vectors, tolerance=1e-10, Keep_all=False):
    """Whiten dense reference vectors; TTNS callers use Gram_schmidt_TTN."""
    vectors = list(vectors)
    if not vectors:
        raise ValueError('vectors must be nonempty.')
    matrix = np.column_stack([np.asarray(v) for v in vectors])
    gram = matrix.conj().T @ matrix
    eigenvalues, basis = np.linalg.eigh(gram)
    keep = eigenvalues > (0.0 if Keep_all else tolerance)
    return (basis[:, keep] / np.sqrt(eigenvalues[keep]), eigenvalues[keep])

def pivoted_cholesky(vectors, tol=1e-8, svd_tol=1e-10, *,
                     gram_matrix=None, return_gram=False):
    """Return (L, permutation, remaining_diagonal, error, X).

    X contains orthonormalizing snapshot coefficients. Only the Hermitian
    triangle of Gram overlaps is evaluated. With return_gram=True, append S
    as a sixth result so projection and Lehmann weights can reuse it.
    """
    vectors = list(vectors)
    if not np.isfinite(tol) or tol < 0 or not np.isfinite(svd_tol) or svd_tol < 0:
        raise ValueError('Cholesky and SVD tolerances must be finite and nonnegative.')
    S = _krylov_gram(vectors) if gram_matrix is None else np.asarray(gram_matrix, dtype=complex)
    n = len(vectors)
    if n == 0 or S.shape != (n, n) or not np.all(np.isfinite(S)):
        raise ValueError('Invalid Krylov Gram matrix.')
    if not np.allclose(S, S.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError('The Krylov Gram matrix must be Hermitian.')
    d = S.diagonal().real.copy()
    if np.any(d <= 0):
        raise ValueError('A Krylov snapshot has zero or negative squared norm.')
    p = np.arange(n)
    factors = np.zeros_like(S)
    error = float(d.sum())
    m = 0
    while error > tol and m < n:
        best = m + int(np.argmax(d[p[m:]]))
        p[m], p[best] = p[best], p[m]
        pivot = p[m]
        if d[pivot] <= 0:
            break
        factors[m, pivot] = np.sqrt(d[pivot])
        for index in p[m + 1:]:
            correction = np.vdot(factors[:m, pivot], factors[:m, index])
            factors[m, index] = (S[pivot, index] - correction) / factors[m, pivot]
            d[index] = max(0.0, d[index] - abs(factors[m, index]) ** 2)
        m += 1
        error = float(d[p[m:]].sum())
    if m == 0:
        raise ValueError('Cholesky tolerance removed the entire Krylov space.')
    L = factors[:m]
    _, singular_values, Vh = np.linalg.svd(L, full_matrices=False)
    keep = singular_values > svd_tol * singular_values[0]
    if not np.any(keep):
        raise ValueError('SVD tolerance removed the entire Krylov space.')
    X = Vh.conj().T[:, keep] / singular_values[keep]
    result = (L, p, d, error, X)
    return result + (S,) if return_gram else result

def krylov_projection(H, X, vectors, tol=0.001, Keep_all=False):
    """Takes a Hamiltonian and projects it into an orthonormalised Krylov subspace"""
    V = np.column_stack([np.asarray(v, dtype=np.complex128) for v in vectors])
    HV = H @ V
    H_eff = V.conj().T @ HV
    H_orthog = X.conj().T @ H_eff @ X
    energies, Q = np.linalg.eigh(H_orthog)
    coefficients = X @ Q
    ritz_vectors = V @ coefficients
    residuals = np.empty(len(energies), dtype=float)
    relative_residuals = np.empty_like(residuals)
    for k, energy in enumerate(energies):
        phi = ritz_vectors[:, k]
        norm_phi = np.linalg.norm(phi)
        if norm_phi == 0:
            residuals[k] = np.inf
            relative_residuals[k] = np.inf
            continue
        phi = phi / norm_phi
        ritz_vectors[:, k] = phi
        residual_vector = H @ phi - energy * phi
        residuals[k] = np.linalg.norm(residual_vector)
        relative_residuals[k] = residuals[k] / max(1.0, abs(energy))
    if Keep_all:
        keep = None
    else:
        E_scale = 1.0
        scaled_residuals = residuals / np.maximum(np.abs(energies), E_scale)
        keep = scaled_residuals <= tol
        energies = energies[keep]
        Q = Q[:, keep]
        ritz_vectors = ritz_vectors[:, keep]
        residuals = residuals[keep]
        relative_residuals = relative_residuals[keep]
    print('Ritz values and residuals:')
    for k, energy in enumerate(energies):
        print(f'{k:2d}: E = {energy: .10f}, residual = {residuals[k]:.3e}, relative = {relative_residuals[k]:.3e}')
    return (np.real_if_close(energies), Q, ritz_vectors, residuals, relative_residuals)

def krylov_projection_TTN(H, X, vectors, tol=1e-3, Keep_all=False,
                          application_svd_params=None, *, gram_matrix=None,
                          full_residuals=True, return_diagnostics=False):
    """Project the physical Hermitian TTNO using native H applications.

    The five-result legacy interface is preserved. Full residuals retain the
    compressed H|v> states and are residuals of that compressed action, not
    certified exact-H errors. Set full_residuals=False with Keep_all=True to
    release each H|v> after its projected column; residual arrays then hold NaN.
    """
    vectors = list(vectors)
    X = np.asarray(X, dtype=np.complex128)
    m = len(vectors)
    if m == 0 or X.ndim != 2 or X.shape[0] != m or X.shape[1] == 0:
        raise ValueError('X must have shape (number_of_snapshots, positive_rank).')
    if not np.all(np.isfinite(X)) or not np.isfinite(tol) or tol < 0:
        raise ValueError('Nonfinite coefficients or invalid residual tolerance.')
    if not full_residuals and not Keep_all:
        raise ValueError('Residual filtering requires full_residuals=True.')
    S = _krylov_gram(vectors) if gram_matrix is None else np.asarray(gram_matrix, dtype=complex)
    if S.shape != (m, m) or not np.all(np.isfinite(S)):
        raise ValueError('Invalid cached Krylov Gram matrix.')
    white_error = np.linalg.norm(X.conj().T @ S @ X - np.eye(X.shape[1]))
    if not np.isfinite(white_error) or white_error > 1e-4:
        raise ValueError('The Krylov basis is not orthonormal; increase its cutoff.')
    if application_svd_params is None:
        application_svd_params = SVDParameters(
            max_bond_dim=100, rel_tol=1e-10, total_tol=1e-12, renorm=False,
        )
    projected = np.empty((m, m), dtype=complex)
    K = np.empty_like(projected) if full_residuals else None
    h_vectors = []
    for j, vector in enumerate(vectors):
        h_vector = apply_and_compress_native(vector, H, application_svd_params)
        for i, bra in enumerate(vectors):
            projected[i, j] = _krylov_overlap(bra, h_vector)
        if full_residuals:
            h_vectors.append(h_vector)
            for i in range(j + 1):
                value = _krylov_overlap(h_vectors[i], h_vector)
                K[i, j], K[j, i] = value, np.conjugate(value)
        del h_vector
    if not np.all(np.isfinite(projected)) or (K is not None and not np.all(np.isfinite(K))):
        raise ValueError('Nonfinite projected Hamiltonian contractions.')
    raw = X.conj().T @ projected @ X
    defect = float(np.linalg.norm(raw - raw.conj().T) / max(np.linalg.norm(raw), 1e-30))
    energies, Q = np.linalg.eigh((raw + raw.conj().T) / 2)
    coefficients = X @ Q
    residuals = np.full(energies.shape, np.nan)
    if full_residuals:
        norms = np.einsum('ik,ij,jk->k', coefficients.conj(), S, coefficients).real
        h_mean = np.einsum('ik,ij,jk->k', coefficients.conj(), projected, coefficients).real
        h_norm = np.einsum('ik,ij,jk->k', coefficients.conj(), K, coefficients).real
        if np.any(norms <= 0):
            raise ValueError('A Ritz vector has nonpositive norm.')
        r2 = h_norm - 2 * energies * h_mean + energies ** 2 * norms
        scale = np.maximum(1.0, abs(h_norm) + 2 * abs(energies * h_mean) + energies ** 2 * abs(norms))
        if np.any(r2 < -1e-8 * scale):
            raise ValueError('Negative squared Ritz residual exceeds roundoff.')
        residuals = np.sqrt(np.maximum(r2, 0) / norms)
    relative_residuals = residuals / np.maximum(1.0, abs(energies))
    if not Keep_all:
        keep = residuals <= tol
        energies, Q, coefficients = energies[keep], Q[:, keep], coefficients[:, keep]
        residuals, relative_residuals = residuals[keep], relative_residuals[keep]
    print('Ritz values and residuals:')
    for index, energy in enumerate(energies):
        print(f'{index:2d}: E = {energy: .10f}, residual = {residuals[index]:.3e}, relative = {relative_residuals[index]:.3e}')
    result = (energies, Q, coefficients, residuals, relative_residuals)
    if return_diagnostics:
        return result + (dict(gram_matrix=S, projected_hamiltonian=projected,
                             hamiltonian_norm_matrix=K,
                             whitening_residual=float(white_error),
                             hermiticity_defect=defect),)
    return result

def Complex_time_evolution(state, TTNO, time_step_size, final_time,
                           max_bond_dim, operators, M,
                           time_evo_mode=TimeEvoMode.RK45, *,
                           rel_tol=1e-5, abs_tol=1e-5, solver_options=None):
    """Collect t=0 plus M snapshots separated by final_time.

    One BUG integrator is retained across all samples. RK45 is now accepted
    explicitly; its default and positional argument match the uploaded helper.
    BUG truncation defaults and M+1 snapshot semantics are unchanged.
    """
    if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)) or M < 0:
        raise ValueError('M must be a nonnegative integer number of intervals.')
    if not np.isfinite(time_step_size) or time_step_size <= 0 or not np.isfinite(final_time) or final_time <= 0:
        raise ValueError('time_step_size and final_time must be finite and positive.')
    steps_per_sample = round(final_time / time_step_size)
    if steps_per_sample < 1 or not np.isclose(steps_per_sample * time_step_size, final_time):
        raise ValueError('final_time must be an integer multiple of time_step_size.')
    if M == 0:
        return [deepcopy(state)]
    config = BUGConfig(max_bond_dim=max_bond_dim, rel_tol=rel_tol,
                       total_tol=abs_tol, time_evo_mode=time_evo_mode,
                       record_max_bdim=False)
    options = {'rtol': 1e-8, 'atol': 1e-10}
    if solver_options is not None:
        options.update(dict(solver_options))
    bug = BUG(initial_state=state, hamiltonian=TTNO,
              time_step_size=time_step_size, final_time=M * final_time,
              operators=operators, config=config, solver_options=options)
    snapshots = [deepcopy(bug.state)]
    for step in range(1, M * steps_per_sample + 1):
        bug.run_one_time_step()
        if step % steps_per_sample == 0:
            snapshots.append(deepcopy(bug.state))
    return snapshots

def Gram_schmidt_TTN(states, tolerance=1e-10, Keep_all=False):
    """A Gram Schmidt orthonormalisation without TTNS contraction"""
    m = len(states)
    S = np.zeros((m, m), dtype=np.complex128)
    if Keep_all:
        tolerance = 0.0
    for i in range(m):
        for j in range(m):
            S[i, j] = states[j].scalar_product(states[i])
    eigenvalues, U = np.linalg.eigh(S)
    keep = eigenvalues > tolerance
    X = U[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep]))
    return (X, eigenvalues[keep])

def Greens_function_lehman(c, E, psi, n, w, E_0=None, psi_0=None):
    """
    Calculate a zero-temperature retarded Green's function from a
    complete or truncated set of eigenpairs.

    Parameters
    ----------
    c
        Fermionic annihilation operator in the full Hilbert space.

    E : array_like, shape (n_ritz,)
        Exact energies or truncated Ritz energies.

    psi : array_like, shape (hilbert_dim, n_ritz)
        Corresponding eigenvectors or Ritz vectors as columns.
        For truncated Krylov results, these must first be transformed
        back into the full Hilbert space.

    n : float
        Positive broadening parameter eta.

    w : float or array_like
        Frequency or frequency grid.

    E_0 : float, optional
        Ground-state energy. If omitted, the lowest supplied energy
        is used.

    psi_0 : array_like, optional
        Ground-state vector in the full Hilbert space. If omitted,
        the vector corresponding to the lowest supplied energy is used.

    Returns
    -------
    G
        Approximate retarded Green's function.
    """
    E = np.asarray(E)
    psi = np.asarray(psi, dtype=np.complex128)
    w = np.asarray(w, dtype=np.float64)
    if n <= 0.0:
        raise ValueError('The broadening parameter n must be positive.')
    E = np.real_if_close(E)
    if np.iscomplexobj(E):
        raise ValueError('The supplied energies contain significant imaginary parts.')
    E = np.asarray(E, dtype=np.float64).reshape(-1)
    if psi.ndim != 2:
        raise ValueError('psi must have shape (Hilbert-space dimension, number of states).')
    if psi.shape[1] != len(E):
        raise ValueError(f'psi contains {psi.shape[1]} states, but len(E)={len(E)}.')
    if c.shape[0] != psi.shape[0]:
        raise ValueError('The Ritz vectors are not represented in the full Hilbert space. Reconstruct them from the Krylov coefficients first.')
    order = np.argsort(E)
    E = E[order]
    psi = psi[:, order]
    if psi_0 is None:
        psi_0 = psi[:, 0]
    else:
        psi_0 = np.asarray(psi_0, dtype=np.complex128).reshape(-1)
    if psi_0.shape[0] != psi.shape[0]:
        raise ValueError('psi_0 and the Ritz vectors must belong to the same Hilbert space.')
    norm_0 = np.linalg.norm(psi_0)
    if norm_0 == 0.0:
        raise ValueError('psi_0 has zero norm.')
    psi_0 = psi_0 / norm_0
    if E_0 is None:
        E_0 = E[0]
    else:
        E_0 = float(np.real(E_0))
    c_dag = c.conj().T
    particle_state = c_dag @ psi_0
    hole_state = c @ psi_0
    particle_amplitudes = psi.conj().T @ particle_state
    hole_amplitudes = psi.conj().T @ hole_state
    particle_weights = np.abs(particle_amplitudes) ** 2
    hole_weights = np.abs(hole_amplitudes) ** 2
    excitation_energies = E - E_0
    z = w[..., None] + 1j * n
    particle_part = np.sum(particle_weights / (z - excitation_energies), axis=-1)
    hole_part = np.sum(hole_weights / (z + excitation_energies), axis=-1)
    G = particle_part + hole_part
    if w.ndim == 0:
        return complex(G)
    return G

def spectral_function(G):
    """Calculate A(w) = -Im[G^R(w)] / pi."""
    return -np.imag(np.asarray(G)) / np.pi

def Greens_function_lehman_TTN(E, Ritz_coefficients, Krylo_vectors, psi_0, E_0, eta, w, node_key=None, *, parity_nodes=()):
    """Legacy full-basis Lehmann helper with explicit native charge transport.

    The basis must include the N and N+/-1 sectors needed by the Lehmann sum.
    psi_0 is the ground state's coefficient vector in the original snapshot
    basis. For separate fixed-N CTKS sectors, prefer create_impurity_excitations
    and lehmann_sector_TTN. Supply the preceding JW sites as parity_nodes if
    node_key is not the first mode in the model's physical ordering.
    """
    vectors = list(Krylo_vectors)
    if not vectors:
        raise ValueError('Krylo_vectors must be nonempty.')
    if node_key is None:
        node_key = 'Impurity_1_up' if 'Impurity_1_up' in vectors[0].nodes else next((i for i, n in vectors[0].nodes.items() if n.open_dimension() == 2))
    annihilate = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    native = is_lasyten_tensor(vectors[0].tensors[vectors[0].root_id])
    matrix = np.zeros((len(vectors), len(vectors)), dtype=complex)
    if native:
        canonical = []
        charges = []
        for state in vectors:
            state = deepcopy(state)
            state.ensure_root_orth_center(mode=SplitMode.REDUCED)
            charges.append(check_native_u1(state))
            canonical.append(state)
        for j, state in enumerate(canonical):
            operated = apply_local_charge_operator(state, annihilate, node_id=node_key, delta=-1, parity_nodes=parity_nodes)
            for i, bra in enumerate(canonical):
                if charges[i] == charges[j] - 1:
                    matrix[i, j] = operated.scalar_product(bra)
    else:
        for j, state in enumerate(vectors):
            operated = deepcopy(state)
            for site in parity_nodes:
                operated.apply_operator(TensorProduct({site: np.diag([1.0, -1.0])}))
            operated.apply_operator(TensorProduct({node_key: annihilate}))
            for i, bra in enumerate(vectors):
                matrix[i, j] = operated.scalar_product(bra)
    energies = np.asarray(E)
    coefficients = np.asarray(Ritz_coefficients)
    ground = np.asarray(psi_0)
    z = np.asarray(w)[..., None] + 1j * eta
    creation_weight = np.abs(ground.conj() @ matrix @ coefficients) ** 2
    removal_weight = np.abs(ground.conj() @ matrix.conj().T @ coefficients) ** 2
    return np.sum(creation_weight / (z + E_0 - energies) + removal_weight / (z - E_0 + energies), axis=-1)

def lehmann_sector_TTN(E, Ritz_coefficients, Krylov_vectors, psi_exc,
                       E_0, eta, omega, sign, *, overlaps=None,
                       excitation_norm=None):
    """Normalized-sector response; multiply by physical excitation Z once.

    overlaps may be the cached Gram row <psi_exc|snapshot_j>, and
    excitation_norm may be its diagonal entry when snapshot zero is psi_exc.
    """
    E = np.asarray(E)
    B = np.asarray(Ritz_coefficients, dtype=complex)
    vectors = list(Krylov_vectors)
    if E.ndim != 1 or E.size == 0 or B.shape != (len(vectors), E.size):
        raise ValueError('Ritz coefficients must have shape (snapshots, nonzero_poles).')
    if sign not in (-1, 1) or not np.isfinite(eta) or eta <= 0:
        raise ValueError('sign must be +/-1 and eta must be finite and positive.')
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(B)):
        raise ValueError('Nonfinite Ritz poles or coefficients.')
    if np.max(abs(E.imag)) > 1e-10 * max(1.0, float(np.max(abs(E.real)))):
        raise ValueError('Physical Lehmann poles must be real.')
    E = E.real
    E_0 = complex(E_0)
    if not np.isfinite(E_0) or abs(E_0.imag) > 1e-10 * max(1.0, abs(E_0.real)):
        raise ValueError('The physical ground-state energy must be real and finite.')
    if overlaps is None:
        overlaps = np.array([_krylov_overlap(psi_exc, v) for v in vectors])
    overlaps = np.asarray(overlaps, dtype=complex)
    if overlaps.shape != (len(vectors),) or not np.all(np.isfinite(overlaps)):
        raise ValueError('Invalid excitation-snapshot overlaps.')
    norm = _krylov_overlap(psi_exc, psi_exc) if excitation_norm is None else complex(excitation_norm)
    if not np.isfinite(norm) or norm.real <= 0 or abs(norm.imag) > 1e-10 * max(1.0, norm.real):
        raise ValueError('The excitation must have positive finite norm.')
    order = np.argsort(E)
    E, B = E[order], B[:, order]
    weights = abs(overlaps @ B) ** 2 / norm.real
    z = np.asarray(omega)[..., None] + 1j * eta
    G = np.sum(weights / (z - sign * (E - E_0.real)), axis=-1)
    return G, float(weights.sum())

def _bug_step(state, duration, TTNO, operators, max_bond_dim, time_step_size=0.1, rel_tol=1e-12, abs_tol=1e-12):
    """Evolve a TTNS by exp(-i H duration) for duration >= 0."""
    duration = float(duration)
    if duration < 0.0:
        raise ValueError('BUG evolution requires a non-negative duration.')
    if duration == 0.0:
        return state
    tree_bug, *_ = BUG_time_evolution(Tree_state=state, TTNO_Hamil=TTNO, time_step_size=min(time_step_size, duration), final_time=duration, operators=operators, max_bond_dim=max_bond_dim, rel_tol=rel_tol, abs_tol=abs_tol)
    return tree_bug.state

def real_time_evolution_TTN(t_end, steps, psi_ket, bra_vectors, TTNO, operators, max_bond_dim, eta, E_0=0.0, time_step_size=0.1):
    """
    Calculate

        C_i(t) = exp(i E_0 t) <bra_i|exp(-i H t)|psi_ket>

    on a uniform time grid.

    `steps` denotes the number of time intervals, so the returned arrays
    contain `steps + 1` points, including t=0.
    """
    if t_end <= 0.0:
        raise ValueError('t_end must be positive.')
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer.')
    bra_vectors = list(bra_vectors)
    time_grid = np.linspace(0.0, t_end, steps + 1)
    correlation = np.empty((len(time_grid), len(bra_vectors)), dtype=np.complex128)
    psi_t = deepcopy(psi_ket)
    previous_time = 0.0
    for n, t in enumerate(time_grid):
        delta_t = t - previous_time
        psi_t = _bug_step(state=psi_t, duration=delta_t, TTNO=TTNO, operators=operators, max_bond_dim=max_bond_dim, time_step_size=time_step_size)
        phase = np.exp(1j * (E_0 + 1j * eta) * t)
        correlation[n, :] = phase * np.asarray([psi_t.scalar_product(bra) for bra in bra_vectors], dtype=np.complex128)
        previous_time = t
    return (time_grid, correlation)

def Greens_function_CTKS_time_domain(psi_A, psi_B, Vectors, Q, X, E, t_end, steps, interval, TTNO_minus_H, operators, max_bond_dim, eta, E_0=0.0, time_step_size=0.1):
    """
    CTKS approximation to

        C_AB(t) = exp(i E_0 t) <psi_A|exp(-i H t)|psi_B>.

    Parameters
    ----------
    interval:
        CTKS construction interval T.

    TTNO_minus_H:
        TTNO representing -H. It permits bra-side propagation using only
        positive BUG time steps.

    E:
        Absolute Ritz energies of the projected Hamiltonian.
    """
    if t_end <= 0.0:
        raise ValueError('t_end must be positive.')
    if interval <= 0.0:
        raise ValueError('interval must be positive.')
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer.')
    Vectors = list(Vectors)
    X = np.asarray(X, dtype=np.complex128)
    Q = np.asarray(Q, dtype=np.complex128)
    E = np.asarray(E, dtype=float)
    number_of_vectors = len(Vectors)
    if X.ndim != 2 or X.shape[0] != number_of_vectors:
        raise ValueError('X must be a matrix with X.shape[0] == len(Vectors).')
    if Q.ndim != 2 or Q.shape[0] != X.shape[1]:
        raise ValueError('Q must be a matrix with Q.shape[0] == X.shape[1].')
    if Q.shape[1] != len(E):
        raise ValueError('Q.shape[1] must equal len(E).')
    time_grid = np.linspace(0.0, t_end, steps + 1)
    interval_number = np.floor(time_grid / interval).astype(int)
    remainder = time_grid - interval_number * interval
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, t_end, interval)
    near_upper_boundary = np.abs(remainder - interval) < tolerance
    interval_number[near_upper_boundary] += 1
    remainder[near_upper_boundary] = 0.0
    remainder[np.abs(remainder) < tolerance] = 0.0
    basis_to_ritz = X @ Q
    M = np.asarray([psi_B.scalar_product(vector) for vector in Vectors], dtype=np.complex128)
    right_ritz = basis_to_ritz.conj().T @ M
    unique_remainders, remainder_indices = np.unique(remainder, return_inverse=True)
    Y = np.empty((len(unique_remainders), number_of_vectors), dtype=np.complex128)
    psi_A_tau = deepcopy(psi_A)
    previous_tau = 0.0
    for n, tau in enumerate(unique_remainders):
        delta_tau = tau - previous_tau
        psi_A_tau = _bug_step(state=psi_A_tau, duration=delta_tau, TTNO=TTNO_minus_H, operators=operators, max_bond_dim=max_bond_dim, time_step_size=time_step_size)
        Y[n, :] = np.exp(1j * (E_0 + 1j * eta) * tau) * np.asarray([vector.scalar_product(psi_A_tau) for vector in Vectors], dtype=np.complex128)
        previous_tau = tau
    correlation = np.empty(len(time_grid), dtype=np.complex128)
    for n, number_of_intervals in enumerate(interval_number):
        complete_time = number_of_intervals * interval
        ritz_phase = np.exp(-1j * (E - E_0 - 1j * eta) * complete_time)
        coefficients = basis_to_ritz @ (ritz_phase * right_ritz)
        correlation[n] = np.dot(Y[remainder_indices[n]], coefficients)
    return (time_grid, correlation)

def _copy_ttn_state(state):
    """Use the TTN library's dedicated copy method if one exists."""
    if hasattr(state, 'copy'):
        return state.copy()
    return deepcopy(state)

def sample_backward_correlators(psi_0, vectors, TTNO_backward, operators, max_bond_dim, T, n_steps, bug_time_step=1.0, rel_tol=1e-12, abs_tol=1e-12):
    """
    Sequentially calculate backward-time correlators.

    TTNO_backward must represent -H. Since BUG performs

        exp(-i * TTNO_backward * dt),

    this produces

        exp(+i * H * dt).

    BUG is only called with positive time intervals.
    """
    if T <= 0:
        raise ValueError('T must be positive.')
    if n_steps < 2:
        raise ValueError('n_steps must be at least 2.')
    if n_steps % 2 != 0:
        n_steps += 1
    vectors = list(vectors)
    times = np.linspace(0.0, T, n_steps + 1)
    correlators = np.empty((n_steps + 1, len(vectors)), dtype=np.complex128)
    state = _copy_ttn_state(psi_0)
    correlators[0] = [vector.scalar_product(state) for vector in vectors]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if dt <= 0:
            raise RuntimeError('The time grid must be strictly increasing.')
        internal_step = min(bug_time_step, dt)
        tree_bug, _, _ = BUG_time_evolution(Tree_state=state, TTNO_Hamil=TTNO_backward, time_step_size=internal_step, final_time=dt, operators=operators, max_bond_dim=max_bond_dim, rel_tol=rel_tol, abs_tol=abs_tol)
        state = tree_bug.state
        correlators[index] = [vector.scalar_product(state) for vector in vectors]
    return (times, correlators)

def integrate_correlators(times, correlators, eta, w, E_0=0.0):
    """Integrate the previously sampled time-domain correlators."""
    phase = np.exp((1j * (E_0 + w) - eta) * times)
    integrand = correlators * phase[:, None]
    return simpson(integrand, x=times, axis=0)

def Greens_function_CTKS_from_samples(times, correlators, Vectors, Q, X, E, eta, T, w, E_0=0.0):
    """
    Calculate the Green's function from an existing time evolution.

    Expected column ordering:

        correlators[:, 0]  -> overlap with psi_B
        correlators[:, 1:] -> overlaps with Vectors
    """
    integrated_correlators = integrate_correlators(times=times, correlators=correlators, eta=eta, w=w, E_0=E_0)
    G_tilde = integrated_correlators[0]
    boost_integrals = integrated_correlators[1:]
    M = np.asarray([Vectors[0].scalar_product(vector) for vector in Vectors], dtype=np.complex128)
    delta = np.asarray(E) - E_0 - w - 1j * eta
    K_diag = 1.0 / np.expm1(1j * delta * T)
    right = Q.conj().T @ X.conj().T @ M
    right = K_diag * right
    A = X @ Q @ right
    boost = boost_integrals @ A
    return G_tilde + boost

def Greens_function_CTKS(psi_A, psi_B, Vectors, Q, X, E, eta, T, w, TTNO_backward, operators, max_bond_dim, n_steps, E_0=0.0, bug_time_step=1.0, rel_tol=1e-12, abs_tol=1e-12):
    """
    Calculate the CTKS Green's function using sequential positive-time
    BUG evolution with a TTNO representing -H.
    """
    probe_vectors = [psi_B, *Vectors]
    times, correlators = sample_backward_correlators(psi_0=psi_A, vectors=probe_vectors, TTNO_backward=TTNO_backward, operators=operators, max_bond_dim=max_bond_dim, T=T, n_steps=n_steps, bug_time_step=bug_time_step, rel_tol=rel_tol, abs_tol=abs_tol)
    return Greens_function_CTKS_from_samples(times=times, correlators=correlators, Vectors=Vectors, Q=Q, X=X, E=E, eta=eta, T=T, w=w, E_0=E_0)

def greens_function_CTKS_grid(psi_A, psi_B, Krylov_vectors, Q, X, energies, omega_grid, eta, T, TTNO_backward, operators, max_bond_dim, n_steps, E_0=0.0, bug_time_step=None, rel_tol=1e-08, abs_tol=1e-10, frequency_chunk_size=256, sector_weight=1.0, return_auxiliary=False):
    """
    Calculate one particle-addition or particle-removal CTKS resolvent.

    Conventions
    -----------
    TTNO_backward represents -H, so BUG generates exp(+i H t).

    `energies` and `E_0` must use the same Hamiltonian convention.
    For example:

        energies = absolute Ritz energies
        E_0      = absolute ground-state energy

    or:

        energies = Ritz energies - energy_shift
        E_0      = ground energy - energy_shift

    The returned result is the physical retarded sector resolvent,
    including the factor -i relating the time integral to the
    resolvent.

    If psi_A and psi_B are normalized excitation states, sector_weight
    must contain the discarded physical norm, e.g.

        sector_weight = ||d†|psi_0>||²

    Q may be rectangular after Ritz-state truncation.
    """
    if T <= 0.0:
        raise ValueError('T must be positive.')
    if eta <= 0.0:
        raise ValueError("eta must be positive for a retarded Green's function.")
    if n_steps < 2:
        raise ValueError('n_steps must be at least 2.')
    if n_steps % 2 != 0:
        n_steps += 1
    dt = T / n_steps
    if bug_time_step is None:
        bug_time_step = dt
    if bug_time_step <= 0.0:
        raise ValueError('bug_time_step must be positive.')
    internal_step = min(bug_time_step, dt)
    E_0 = float(np.real(E_0))
    sector_weight = complex(sector_weight)
    omega_input = np.asarray(omega_grid)
    scalar_frequency = omega_input.ndim == 0
    omega_grid = np.atleast_1d(np.asarray(omega_grid, dtype=np.float64))
    if omega_grid.ndim != 1:
        raise ValueError('omega_grid must be a scalar or one-dimensional array.')
    n_frequencies = omega_grid.size
    if frequency_chunk_size is None:
        frequency_chunk_size = n_frequencies
    frequency_chunk_size = int(frequency_chunk_size)
    if frequency_chunk_size < 1:
        raise ValueError('frequency_chunk_size must be positive.')
    Krylov_vectors = list(Krylov_vectors)
    if len(Krylov_vectors) == 0:
        raise ValueError('Krylov_vectors must contain at least one state.')
    X = np.asarray(X, dtype=np.complex128)
    Q = np.asarray(Q, dtype=np.complex128)
    energies = np.real_if_close(np.asarray(energies))
    if np.iscomplexobj(energies):
        raise ValueError('The retained Ritz energies contain significant imaginary parts.')
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    if Q.ndim == 1:
        Q = Q[:, None]
    if X.ndim != 2:
        raise ValueError('X must be two-dimensional.')
    if Q.ndim != 2:
        raise ValueError('Q must be one- or two-dimensional.')
    n_krylov = len(Krylov_vectors)
    orthogonal_dimension = X.shape[1]
    if X.shape[0] != n_krylov:
        raise ValueError(f'X and Krylov_vectors have incompatible dimensions:\n    len(Krylov_vectors) = {n_krylov}\n    X.shape             = {X.shape}')
    if Q.shape[0] != orthogonal_dimension:
        raise ValueError(f'Q and X have incompatible dimensions:\n    X.shape = {X.shape}\n    Q.shape = {Q.shape}')
    n_retained_ritz = Q.shape[1]
    if n_retained_ritz == 0:
        raise ValueError('At least one Ritz state must be retained.')
    if len(energies) != n_retained_ritz:
        raise ValueError(f'The number of Ritz energies must equal the number of retained Ritz vectors:\n    len(energies) = {len(energies)}\n    Q.shape       = {Q.shape}')
    B = X @ Q
    M = np.asarray([vector.scalar_product(psi_B) for vector in Krylov_vectors], dtype=np.complex128)
    projected_M = B.conj().T @ M.conj()
    times = np.linspace(0.0, T, n_steps + 1)
    simpson_weights = np.ones(n_steps + 1, dtype=np.float64)
    simpson_weights[1:-1:2] = 4.0
    simpson_weights[2:-1:2] = 2.0
    simpson_weights *= dt / 3.0
    channels = np.empty((n_steps + 1, n_retained_ritz + 1), dtype=np.complex128)
    raw_overlaps = np.empty(n_krylov, dtype=np.complex128)
    if hasattr(psi_A, 'copy'):
        state = psi_A.copy()
    else:
        state = deepcopy(psi_A)
    channels[0, 0] = psi_B.scalar_product(state)
    for index, vector in enumerate(Krylov_vectors):
        raw_overlaps[index] = vector.scalar_product(state)
    channels[0, 1:] = raw_overlaps @ B
    n_inner = max(1, int(np.ceil(dt / internal_step)))
    h = dt / n_inner
    bug = BUG(initial_state=psi_A, hamiltonian=TTNO_backward, time_step_size=h, final_time=T, operators={}, config=BUGConfig(max_bond_dim=max_bond_dim, rel_tol=rel_tol, total_tol=abs_tol, time_evo_mode=TimeEvoMode.RK45, record_max_bdim=False))
    for step_index in range(1, n_steps + 1):
        for _ in range(n_inner):
            bug.run_one_time_step()
        state = bug.state
        channels[step_index, 0] = psi_B.scalar_product(state)
        for index, vector in enumerate(Krylov_vectors):
            raw_overlaps[index] = vector.scalar_product(state)
        channels[step_index, 1:] = raw_overlaps @ B
    green_function = np.empty(n_frequencies, dtype=np.complex128)
    for chunk_start in range(0, n_frequencies, frequency_chunk_size):
        chunk_stop = min(chunk_start + frequency_chunk_size, n_frequencies)
        omega_chunk = omega_grid[chunk_start:chunk_stop]
        phase = np.exp((1j * (E_0 + omega_chunk[:, None]) - eta) * times[None, :])
        integrated_channels = phase * simpson_weights[None, :] @ channels
        finite_integral = integrated_channels[:, 0]
        integrated_ritz = integrated_channels[:, 1:]
        delta = energies[None, :] - E_0 - omega_chunk[:, None] - 1j * eta
        exponent = -1j * delta * T
        u = np.exp(exponent)
        K_diagonal = u / -np.expm1(exponent)
        boost_coefficients = K_diagonal * projected_M[None, :]
        boost_integral = np.sum(integrated_ritz * boost_coefficients, axis=1)
        green_function[chunk_start:chunk_stop] = -1j * sector_weight * (finite_integral + boost_integral)
    if scalar_frequency:
        green_function = complex(green_function[0])
    if return_auxiliary:
        auxiliary = {'times': times, 'projected_correlators': channels, 'simpson_weights': simpson_weights, 'projected_basis': B, 'projected_M': projected_M, 'projected_seed_weight': np.sum(np.abs(projected_M) ** 2), 'orthogonal_dimension': orthogonal_dimension, 'n_retained_ritz': n_retained_ritz, 'n_bug_calls': n_steps, 'dt': dt, 'sector_weight': sector_weight}
        return (green_function, auxiliary)
    return green_function

def build_u1_t3ns(*, norb, nb, virtual_bond_dimension, bath_bond_dimension,
                  total_particles=None, seed=11, symmetry=None, total_particles_by_spin=None):
    """Build single-U1 or spin-resolved U1xU1 states on the same topology."""
    from pytreenet_lasyten.util.hk_u1_model import build_u1_t3ns as build
    return build(norb=norb, nb=nb, virtual_bond_dimension=virtual_bond_dimension,
                 bath_bond_dimension=bath_bond_dimension, total_particles=total_particles,
                 seed=seed, symmetry=symmetry, total_particles_by_spin=total_particles_by_spin)

def check_native_u1(state, operator=None):
    """Validate all native charge components and return total particle number."""
    from pytreenet_lasyten.util.hk_u1_model import check_native_u1 as check
    return check(state, operator)

def apply_and_compress_native(state, operator, params):
    """Native DIRECT application followed by native block SVD.

    HALF_DENSITY_MATRIX is not ported in this package. DIRECT temporarily
    multiplies state/operator bond spaces before compression; this is not
    a hard memory bound. Projection residuals use the compressed H|psi>.
    """
    result = apply_ttno_to_ttns(state, operator, ApplicationMethod.DIRECT)
    svd_truncation(result, params)
    return result

def timed_peak_rss(function):
    """Per-call sampled process RSS, also works under Linux/WSL."""
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        value = function()
    finally:
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        thread.join()
    return (value, elapsed, peak[0] / 1024 ** 3)

def construct_fermionic_order(norb, n_bath):
    """
    Global JW ordering.

    Impurities come first so that the usual local
    Hubbard-Kanamori representation remains string-free.

    Bath modes are ordered as spinful bath levels:
    Bath_m_up_k, Bath_m_down_k
    """
    order = []
    for orbital in range(1, norb + 1):
        order.extend([f'Impurity_{orbital}_up', f'Impurity_{orbital}_down'])
    for orbital in range(1, norb + 1):
        for bath_index in range(n_bath):
            order.extend([f'Bath_{orbital}_up_{bath_index}', f'Bath_{orbital}_down_{bath_index}'])
    node_order = ['node1', 'node2', 'node3', 'node4']
    contraction_order = order + node_order
    return (order, contraction_order)

def construct_Hubbard_Kanamori_hamiltonian(norb, U, U_prime, J, mu=0.0, angle_factor=0.0):
    """Construct the Hubbard–Kanamori impurity Hamiltonian."""
    contour_factor = np.exp(1j * angle_factor)
    annihilate = np.array([[0, 1], [0, 0]], dtype=complex)
    create = annihilate.conj().T
    number = create @ annihilate
    pz = np.array([[1, 0], [0, -1]], dtype=complex)
    conversion_dictionary = {'c': annihilate, 'c_dag': create, 'n': number, 'I1': np.eye(1, dtype=complex), 'I2': np.eye(2, dtype=complex), 'pz': pz}
    contour_factor = np.real_if_close(np.exp(1j * angle_factor))
    coefficient_dict = {'U': contour_factor * U, 'U_prime': contour_factor * U_prime, 'U_prime_minus_J': contour_factor * (U_prime - J), 'J': contour_factor * J, 'minus_mu': -contour_factor * mu, 'contour_factor': contour_factor}
    hamiltonian = Hamiltonian(terms=[], conversion_dictionary=conversion_dictionary, coeffs_mapping=coefficient_dict)
    for i in range(1, norb + 1):
        i_up = f'Impurity_{i}_up'
        i_down = f'Impurity_{i}_down'
        hamiltonian.add_term((Fraction(1), 'minus_mu', TensorProduct({i_up: 'n'})))
        hamiltonian.add_term((Fraction(1), 'minus_mu', TensorProduct({i_down: 'n'})))
        hamiltonian.add_term((Fraction(1), 'U', TensorProduct({i_up: 'n', i_down: 'n'})))
    for i, j in combinations(range(1, norb + 1), 2):
        i_up = f'Impurity_{i}_up'
        i_down = f'Impurity_{i}_down'
        j_up = f'Impurity_{j}_up'
        j_down = f'Impurity_{j}_down'
        hamiltonian.add_term((Fraction(1), 'U_prime', TensorProduct({i_up: 'n', j_down: 'n'})))
        hamiltonian.add_term((Fraction(1), 'U_prime', TensorProduct({i_down: 'n', j_up: 'n'})))
        hamiltonian.add_term((Fraction(1), 'U_prime_minus_J', TensorProduct({i_up: 'n', j_up: 'n'})))
        hamiltonian.add_term((Fraction(1), 'U_prime_minus_J', TensorProduct({i_down: 'n', j_down: 'n'})))
        hamiltonian.add_term((Fraction(-1), 'J', TensorProduct({i_up: 'c_dag', i_down: 'c', j_up: 'c', j_down: 'c_dag'})))
        hamiltonian.add_term((Fraction(-1), 'J', TensorProduct({i_up: 'c', i_down: 'c_dag', j_up: 'c_dag', j_down: 'c'})))
        hamiltonian.add_term((Fraction(1), 'J', TensorProduct({i_up: 'c_dag', i_down: 'c_dag', j_up: 'c', j_down: 'c'})))
        hamiltonian.add_term((Fraction(1), 'J', TensorProduct({i_up: 'c', i_down: 'c', j_up: 'c_dag', j_down: 'c_dag'})))
    return hamiltonian

def jw_bilinear(fermionic_order, creation_site, annihilation_site):
    """
    JW representation of

        f_creation_site^dagger f_annihilation_site.
    """
    positions = {site: index for index, site in enumerate(fermionic_order)}
    if creation_site not in positions:
        raise KeyError(f'{creation_site!r} is absent from fermionic_order.')
    if annihilation_site not in positions:
        raise KeyError(f'{annihilation_site!r} is absent from fermionic_order.')
    p = positions[creation_site]
    q = positions[annihilation_site]
    if p == q:
        return TensorProduct({creation_site: 'n'})
    local_operators = {creation_site: 'c_dag', annihilation_site: 'c'}
    for index in range(min(p, q) + 1, max(p, q)):
        intermediate_site = fermionic_order[index]
        local_operators[intermediate_site] = 'pz'
    return TensorProduct(local_operators)

def add_bath_interaction(Hamiltonian, n_bath, v, e):
    """Add Interactions with a fermionic bath to the Impurity Hamiltonian"""
    coeff_dict = Hamiltonian.coeffs_mapping
    contour_factor = coeff_dict['contour_factor']
    norb = 3
    fermionic_order, _ = construct_fermionic_order(norb=norb, n_bath=n_bath)
    for orbital in range(1, norb + 1):
        for spin in ('up', 'down'):
            impurity_id = f'Impurity_{orbital}_{spin}'
            for bath_index in range(n_bath):
                bath_id = f'Bath_{orbital}_{spin}_{bath_index}'
                energy_key = f'e_{orbital}_{spin}_{bath_index}'
                forward_key = f'v_forward_{orbital}_{spin}_{bath_index}'
                backward_key = f'v_backward_{orbital}_{spin}_{bath_index}'
                coeff_dict[energy_key] = contour_factor * e[bath_index]
                coeff_dict[forward_key] = contour_factor * v[bath_index]
                coeff_dict[backward_key] = contour_factor * np.conjugate(v[bath_index])
                Hamiltonian.add_term((Fraction(1), energy_key, TensorProduct({bath_id: 'n'})))
                Hamiltonian.add_term((Fraction(1), forward_key, jw_bilinear(fermionic_order=fermionic_order, creation_site=impurity_id, annihilation_site=bath_id)))
                Hamiltonian.add_term((Fraction(1), backward_key, jw_bilinear(fermionic_order=fermionic_order, creation_site=bath_id, annihilation_site=impurity_id)))
    return

def construct_HK_with_bath(TTNS, norb, U, U_prime, J, mu_ph, angle_factor, nb, v, e):
    """Same symbolic HK/JW model; charge-labelled native U(1) TTNO."""
    hamiltonian = construct_Hubbard_Kanamori_hamiltonian(norb=norb, U=U, U_prime=U_prime, J=J, mu=mu_ph, angle_factor=angle_factor)
    add_bath_interaction(hamiltonian, n_bath=nb, v=v, e=e)
    operator = u1_ttno_from_hamiltonian(hamiltonian, TTNS)
    check_native_u1(TTNS, operator)
    return (hamiltonian, operator)

def ctks_time_parts(psi_A, psi_B, Vectors, Q, X, E, t_end, steps, interval, TTNO_minus_H, operators, max_bond_dim, eta=0.0, E_0=0.0, time_step_size=0.1, *, bug_step):
    """Return (time, C_real, K_time, C_total) as complex-valued time series.

    C_total approximates exp((i*E_0 - eta)*t) <psi_A|exp(-i*H*t)|psi_B>.
    C_real contains the direct correlation for 0 <= t < interval, and zeros
    afterward. K_time contains the CTKS-extrapolated tail for t >= interval,
    and zeros before it. Thus C_total = C_real + K_time. K_time is the
    time-domain tail whose Fourier transform gives the long-time correction;
    it is not the frequency-dependent geometric boost evaluated at time t.

    H must be Hermitian; TTNO_minus_H represents -H. E contains absolute Ritz
    energies, X has shape (number of vectors, orthonormal rank), and Q has
    shape (orthonormal rank, number of retained Ritz states).

    This uses the PyTreeNet convention ket.scalar_product(bra) = <bra|ket>.
    Pass your existing propagation helper as bug_step=_bug_step. It must
    evolve for the requested duration without normalizing the state.

    interval must be an integer multiple of dt = t_end / steps. Only one
    bra-side trajectory up to interval - dt is propagated. The original
    states are preserved. The retarded prefactor -i and sector weights are
    not included. Damping is already applied; do not apply eta a second time.
    """
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError('t_end must be finite and positive.')
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer.')
    if not np.isfinite(interval) or not 0 < interval <= t_end:
        raise ValueError('interval must satisfy 0 < interval <= t_end.')
    if not np.isfinite(eta) or eta < 0:
        raise ValueError('eta must be finite and nonnegative.')
    if not np.isfinite(E_0) or not np.isfinite(time_step_size) or time_step_size <= 0:
        raise ValueError('E_0 must be finite and time_step_size positive.')
    dt = t_end / steps
    ratio = interval / dt
    n_interval = int(round(ratio))
    tolerance = max(1e-10, 64 * np.finfo(float).eps * abs(ratio))
    if n_interval < 1 or abs(ratio - n_interval) > tolerance:
        raise ValueError('Choose interval as an integer multiple of t_end / steps.')
    T = n_interval * dt
    Vectors = list(Vectors)
    X = np.asarray(X, dtype=complex)
    Q = np.asarray(Q, dtype=complex)
    E = np.real_if_close(np.asarray(E))
    if np.iscomplexobj(E) or E.ndim != 1 or (not np.all(np.isfinite(E))):
        raise ValueError('E must be a finite, real, one-dimensional energy array.')
    E = E.astype(float)
    if not Vectors or X.ndim != 2 or X.shape[0] != len(Vectors):
        raise ValueError('X.shape[0] must match a nonempty Vectors list.')
    if Q.ndim != 2 or Q.shape != (X.shape[1], len(E)) or len(E) == 0:
        raise ValueError('Q must have shape (X.shape[1], len(E)), with len(E) > 0.')
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Q)):
        raise ValueError('X and Q must have finite entries.')
    time = np.linspace(0.0, t_end, steps + 1)
    C_real = np.zeros(steps + 1, dtype=complex)
    K_time = np.zeros(steps + 1, dtype=complex)
    R = X @ Q
    M = np.asarray([psi_B.scalar_product(v) for v in Vectors], dtype=complex)
    right_ritz = R.conj().T @ M
    Y_ritz = np.empty((n_interval, len(E)), dtype=complex)
    bra_state = deepcopy(psi_A)
    for j in range(n_interval):
        if j > 0:
            bra_state = bug_step(state=bra_state, duration=dt, TTNO=TTNO_minus_H, operators=operators, max_bond_dim=max_bond_dim, time_step_size=time_step_size)
        tau = j * dt
        phase = np.exp((1j * E_0 - eta) * tau)
        C_real[j] = phase * psi_B.scalar_product(bra_state)
        Y = np.asarray([v.scalar_product(bra_state) for v in Vectors], dtype=complex)
        Y_ritz[j] = phase * (Y @ R)
    for n in range(n_interval, steps + 1):
        p, j = divmod(n, n_interval)
        phases = np.exp((-1j * (E - E_0) - eta) * (p * T))
        K_time[n] = Y_ritz[j] @ (phases * right_ritz)
    return (time, C_real, K_time, C_real + K_time)

def spectral_from_time(t, C, omega):
    G = np.array([-1j * np.trapezoid(np.exp(1j * w * t) * C, x=t) for w in omega])
    A = -G.imag / np.pi
    return (G, A)

def benchmark_bug_case(nb, virtual_bond, bath_bond, max_bond, *, norb=3, D=1.0, U=2.0, U_prime=None, J=0.2, mu_ph=None, complex_angle=-np.pi / 100, discretisation_function=None):
    """Optional fresh-state benchmark; preserve notebook model variables."""
    if discretisation_function is None:
        from utils import Discretisation_function
        discretisation_function = Discretisation_function
    if U_prime is None:
        U_prime = U - 2 * J
    if mu_ph is None:
        mu_ph = (U + (norb - 1) * (2 * U_prime - J)) / 2
    v, e = discretisation_function(D, nb)
    state = build_u1_t3ns(norb=norb, nb=nb, virtual_bond_dimension=virtual_bond, bath_bond_dimension=bath_bond, seed=11)
    _, operator = construct_HK_with_bath(state, norb, U, U_prime, J, mu_ph, complex_angle, nb, np.asarray(v, dtype=complex), np.asarray(e, dtype=float))
    _, elapsed, peak = timed_peak_rss(lambda: BUG_time_evolution(state, operator, 0.1, 1.0, {}, max_bond, rel_tol=0.001, abs_tol=0.001, time_evo_mode=TimeEvoMode.RK23))
    print(f'nb={nb}, cap={max_bond}: {elapsed:.2f} s, sampled peak RSS {peak:.3f} GiB')
    return (elapsed, peak)

__all__ = ['BUG_time_evolution', 'BUG_time_evolution_fixed', 'imaginary_time_evolution_TTN', 'Grams_schmidt', 'pivoted_cholesky', 'krylov_projection', 'krylov_projection_TTN', 'Complex_time_evolution', 'Gram_schmidt_TTN', 'Greens_function_lehman', 'spectral_function', 'Greens_function_lehman_TTN', 'lehmann_sector_TTN', '_bug_step', 'real_time_evolution_TTN', 'Greens_function_CTKS_time_domain', '_copy_ttn_state', 'sample_backward_correlators', 'integrate_correlators', 'Greens_function_CTKS_from_samples', 'Greens_function_CTKS', 'greens_function_CTKS_grid', 'build_u1_t3ns', 'check_native_u1', 'apply_and_compress_native', 'timed_peak_rss', 'construct_fermionic_order', 'construct_Hubbard_Kanamori_hamiltonian', 'jw_bilinear', 'add_bath_interaction', 'construct_HK_with_bath', 'ctks_time_parts', 'spectral_from_time', 'benchmark_bug_case', 'TimeEvoMode', 'SVDParameters', 'SplitMode', 'BUG', 'BUGConfig', 'FixedBUG', 'FixedBUGConfig', 'DMRGAlgorithm', 'DMRGAlgorithm_sparse', 'create_impurity_excitations']

# Public construction check used by the notebook model cell.
__all__ += ["check_u1_state"]

def _krylov_overlap(bra, ket):
    """Return <bra|ket>; PyTreeNet conjugates scalar_product's second argument."""
    return complex(ket.scalar_product(bra, use_orthogonal_center=False))

def _krylov_gram(vectors):
    m = len(vectors)
    if m == 0:
        raise ValueError('Krylov vectors must be nonempty.')
    S = np.empty((m, m), dtype=np.complex128)
    for j in range(m):
        for i in range(j + 1):
            value = _krylov_overlap(vectors[i], vectors[j])
            if i == j:
                if value.real <= 0 or abs(value.imag) > 1e-10 * max(1.0, abs(value.real)):
                    raise ValueError('A Krylov snapshot has an invalid norm.')
                value = value.real
            S[i, j], S[j, i] = value, np.conjugate(value)
    if not np.all(np.isfinite(S)):
        raise ValueError('Nonfinite Krylov Gram matrix.')
    return S

def broadcast_ttn(network, comm, *, root=0, network_class=None):
    """Broadcast a native TTNS as sparse blocks, never a pickled LASyTen object."""
    from native_mpi_transport import _pack_network, _unpack_network
    rank = comm.Get_rank()
    record = _pack_network(network) if rank == root else None
    record = comm.bcast(record, root=root)
    if rank == root:
        return network
    cls = TreeTensorNetworkState if network_class is None else network_class
    return _unpack_network(cls, record)
