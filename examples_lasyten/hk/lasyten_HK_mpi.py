"""Two-rank U1xU1 HK simulation using helper API 2026.11.04."""

from copy import deepcopy
import sys

import numpy as np
from mpi4py import MPI

from Complex_bug_impurity_solver_lasyten import (
    Complex_time_evolution,
    HK_HELPER_API_VERSION,
    broadcast_ttn,
    construct_fermionic_order,
    construct_HK_with_bath,
    krylov_projection_TTN,
    lehmann_sector_TTN,
    pivoted_cholesky,
    spectral_function,
)
from pytreenet_lasyten.dmrg.dmrg_sparse import DMRGAlgorithm_sparse as DMRG
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
from pytreenet_lasyten.util.abelian_charges import fixed_charge
from pytreenet_lasyten.util.hk_u1_model import build_u1_t3ns, check_native_u1
from pytreenet_lasyten.util.tensor_splitting import SplitMode, SVDParameters
from pytreenet_lasyten.util.u1_excitations import create_impurity_excitations
from pytreenet_lasyten.util.u1_t3ns import check_u1_state
from utils import Discretisation_function


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if "--check-imports" in sys.argv:
    print(
        f"[rank {rank}] Imports OK; helper API {HK_HELPER_API_VERSION}; "
        f"helper: {Complex_time_evolution.__code__.co_filename}",
        flush=True,
    )
    raise SystemExit(0)

if size != 2:
    raise RuntimeError("This script requires exactly two MPI ranks (one per sector).")

# ============================================================
# Model and TTNS parameters
# ============================================================

norb = 3
nb = 5
D = 1.0

v, e = Discretisation_function(D, nb)

v = np.asarray(v, dtype=np.complex128)
e = np.asarray(e, dtype=np.float64)

if v.size != nb or e.size != nb:
    raise ValueError(
        "The bath discretization must return exactly "
        "nb hybridizations and nb bath energies."
    )

U = 2.0
J = 0.2
U_prime = U - 2.0 * J

mu_ph = (
    U
    + (norb - 1) * (2.0 * U_prime - J)
) / 2.0

print("Particle-hole symmetric chemical potential:", mu_ph)

virtual_bond = 15
bath_bond_dimension = 10
max_bond_dim = 15

complex_angle = -np.pi / 100

# ============================================================
# Fixed physical/Jordan-Wigner ordering
# ============================================================

physical_order, full_order = construct_fermionic_order(
    norb=norb,
    n_bath=nb,
)

print("Physical/JW ordering:")
for index, node_id in enumerate(physical_order):
    print(index, node_id)


# Construct the fixed-(N_up, N_down) TTNS directly in allowed blocks.
# None chooses half filling: N = norb * (nb + 1), split equally by spin.
total_particles = None
total_particles_by_spin = None
tree_with_bath_state = build_u1_t3ns(
    norb=norb, nb=nb, virtual_bond_dimension=virtual_bond,
    bath_bond_dimension=bath_bond_dimension,
    total_particles=total_particles, seed=11,
    symmetry="u1xu1", total_particles_by_spin=total_particles_by_spin,
)
assert set(tree_with_bath_state.nodes) == set(full_order)
assert len(tree_with_bath_state.nodes) == 10 + 6 * nb
assert check_u1_state(tree_with_bath_state) < 1e-10
total_particles = check_native_u1(tree_with_bath_state)
particles_by_spin = fixed_charge(tree_with_bath_state)
assert len(particles_by_spin) == 2
print('Spin populations (N_up, N_down):', particles_by_spin)
print('Particle sector:', total_particles, 'of', len(physical_order), 'modes')
print('Normalized norm:', tree_with_bath_state.norm())
print('Number of TTNS nodes:', len(tree_with_bath_state.nodes))
print(
    'Root / arms:',
    tree_with_bath_state.root_id,
    tree_with_bath_state.nodes[tree_with_bath_state.root_id].children,
)
print(
    'Root tensor:',
    type(tree_with_bath_state.tensors[tree_with_bath_state.root_id].native).__name__,
)
print('U(1) × U(1) bond sectors ((N_up, N_down): degeneracy), first six bonds:')
for i, sectors in list(tree_with_bath_state.u1_info['bond_sectors'].items())[:6]:
    print(' ', i, sectors)

# ============================================================
# Physical, backward, and complex-time Hamiltonians
# ============================================================

H_physical, ttno_bath_physical = construct_HK_with_bath(
    tree_with_bath_state, norb, U, U_prime, J, mu_ph, 0.0, nb, v, e,
)
H_backwards, ttno_bath_physical_backwards = construct_HK_with_bath(
    tree_with_bath_state, norb, U, U_prime, J, mu_ph, np.pi, nb, v, e,
)
H_imaginary, ttno_bath_imaginary = construct_HK_with_bath(
    tree_with_bath_state, norb, U, U_prime, J, mu_ph, -np.pi / 2, nb, v, e,
)
H_complex, ttno_bath = construct_HK_with_bath(
    tree_with_bath_state, norb, U, U_prime, J, mu_ph, complex_angle, nb, v, e,
)

# Optional compatibility with code that previously called this H.
H = H_complex
operators = {
    "energy": ttno_bath_physical,
}
# DMRG ground-state search
if rank == 0:
    svd_params = SVDParameters(
        max_bond_dim=max_bond_dim,
        rel_tol=1e-10,
        total_tol=1e-12,
    )

    dmrg = DMRG(
        initial_state=deepcopy(tree_with_bath_state),
        hamiltonian=ttno_bath_physical,
        num_sweeps=20,
        max_iter=100,
        svd_params=svd_params,
        site="one-site",
    )

    energies_per_sweep = dmrg.run()

    ground_state = dmrg.state
    E_0 = float(np.real(np.asarray(energies_per_sweep[-1]).squeeze()))

    print("DMRG ground-state energy:", E_0)
    print("Sweep energies:", np.asarray(energies_per_sweep).squeeze())
    ground_state.ensure_root_orth_center(mode=SplitMode.REDUCED)
    ground_state.normalise()
    assert check_native_u1(ground_state, ttno_bath_physical) == total_particles
    assert fixed_charge(ground_state) == particles_by_spin
    # A finite sweep count is not itself a convergence criterion.
    print(
        'Physical energy check:',
        ground_state.operator_expectation_value(ttno_bath_physical),
    )

    ground_state.normalise()
    (psi_removal, psi_addition, Z_removal, Z_addition) = create_impurity_excitations(
        ground_state=ground_state,
        impurity_node="Impurity_1_up",
        parity_nodes=(),
    )
    print(psi_removal)
    print("Z addition:", Z_addition)
    print("Z removal: ", Z_removal)
    print("Sum:       ", Z_addition + Z_removal)
    assert check_native_u1(psi_removal, ttno_bath_physical) == total_particles - 1
    assert check_native_u1(psi_addition, ttno_bath_physical) == total_particles + 1
    assert fixed_charge(psi_removal) == (particles_by_spin[0] - 1, particles_by_spin[1])
    assert fixed_charge(psi_addition) == (particles_by_spin[0] + 1, particles_by_spin[1])
    assert np.isclose(Z_addition + Z_removal, 1.0, atol=1e-10)
elif rank == 1:
    ground_state = None
    E_0 = None
    psi_removal = None
    psi_addition = None
    Z_removal = None
    Z_addition = None

ground_state = broadcast_ttn(ground_state, comm, root=0)
E_0 = comm.bcast(E_0, root=0)
psi_removal = broadcast_ttn(psi_removal, comm, root=0)
psi_addition = broadcast_ttn(psi_addition, comm, root=0)
Z_removal = comm.bcast(Z_removal, root=0)
Z_addition = comm.bcast(Z_addition, root=0)

comm.Barrier()
# Directly in frequency space
omega = np.linspace(-5 * U, 5 * U, 2001)

eta = 0.05 * U
T = 100.0
dt = 0.1
n_steps = int(round(T / dt))
time_step_size = 0.1
krylov_interval = 1.0
M = 10
operators = {}
if rank == 0:
    # Original residual filter; Keep_all=True is an option for a pipeline test.
    # ============================================================
    # Complex-time Krylov snapshots
    # ============================================================


    current_state = deepcopy(
        tree_with_bath_state
    )

    Krylov_states_r = Complex_time_evolution(
        psi_removal, ttno_bath, time_step_size, krylov_interval,
        max_bond_dim, operators, M, time_evo_mode=TimeEvoMode.RK45,
    )


    # Orthogonalise Krylov space

    L_r, perm_r, diag_r, error_r, X_r, S_r = pivoted_cholesky(
        Krylov_states_r, tol=1e-8, svd_tol=1e-10, return_gram=True,
    )
    print(np.shape(X_r))

    print(
        "Number of snapshots:",
        len(Krylov_states_r),
    )

    print(
        "Orthogonalization shape:",
        X_r.shape,
    )


    # Physical Hamiltonian


    (
        E_krylov_r, Q_r, coefficients_r, residuals_r, relative_residuals_r,
    ) = krylov_projection_TTN(
        ttno_bath_physical, X_r, Krylov_states_r, tol=1e-3, Keep_all=True,
        gram_matrix=S_r,
    )

    sorting = np.argsort(np.real(E_krylov_r))

    E_krylov_r = E_krylov_r[sorting]
    Q_r = Q_r[:, sorting]
    coefficients_r = coefficients_r[:, sorting]
    residuals_r = residuals_r[sorting]
    relative_residuals_r = relative_residuals_r[sorting]

    print("\nKrylov:")
    print(E_krylov_r)
    data = np.column_stack([
        E_krylov_r,
        residuals_r,
        relative_residuals_r,
    ])

    np.savetxt(
        "Ritz_values_removal.csv",
        data,
        delimiter=",",
        header="Energies,Residual_values,Relative_residuals",
        comments="",
    )
    # Removal: negative-frequency poles
    B_removal = coefficients_r

    G_omega_removal, captured_weight_removal = lehmann_sector_TTN(
        E_krylov_r, B_removal, Krylov_states_r, psi_removal,
        E_0, eta, omega, sign=-1,
        overlaps=S_r[0, :], excitation_norm=S_r[0, 0],
    )
    A_krylov_removal = spectral_function(G_omega_removal)
    G_omega_addition = None
    A_krylov_addition = None


elif rank == 1:
    # Original residual filter; Keep_all=True is an option for a pipeline test.
    # ============================================================
    # Complex-time Krylov snapshots
    # ============================================================

    current_state = deepcopy(
        tree_with_bath_state
    )

    Krylov_states_a = Complex_time_evolution(
        psi_addition, ttno_bath, time_step_size, krylov_interval,
        max_bond_dim, operators, M, time_evo_mode=TimeEvoMode.RK45,
    )


    # Orthogonalise Krylov space

    L_a, perm_a, diag_a, error_a, X_a, S_a = pivoted_cholesky(
        Krylov_states_a, tol=1e-8, svd_tol=1e-10, return_gram=True,
    )
    print(np.shape(X_a))

    print(
        "Number of snapshots:",
        len(Krylov_states_a),
    )

    print(
        "Orthogonalization shape:",
        X_a.shape,
    )


    # Physical Hamiltonian


    (
        E_krylov_a, Q_a, coefficients_a, residuals_a, relative_residuals_a,
    ) = krylov_projection_TTN(
        ttno_bath_physical, X_a, Krylov_states_a, tol=1e-3, Keep_all=True,
        gram_matrix=S_a,
    )

    sorting = np.argsort(np.real(E_krylov_a))

    E_krylov_a = E_krylov_a[sorting]
    Q_a = Q_a[:, sorting]
    coefficients_a = coefficients_a[:, sorting]
    residuals_a = residuals_a[sorting]
    relative_residuals_a = relative_residuals_a[sorting]

    print("\nKrylov:")
    print(E_krylov_a)


    data = np.column_stack([
        E_krylov_a,
        residuals_a,
        relative_residuals_a,
    ])

    np.savetxt(
        "Ritz_values_addition.csv",
        data,
        delimiter=",",
        header="Energies,Residual_values,Relative_residuals",
        comments="",
    )


    # Addition: positive-frequency poles
    B_addition = coefficients_a

    G_omega_addition, captured_weight_addition = lehmann_sector_TTN(
        E_krylov_a, B_addition, Krylov_states_a, psi_addition,
        E_0, eta, omega, sign=1,
        overlaps=S_a[0, :], excitation_norm=S_a[0, 0],
    )
    A_krylov_addition = spectral_function(G_omega_addition)


comm.Barrier()
A_krylov_addition = comm.bcast(A_krylov_addition, root=1)
G_omega_addition = comm.bcast(G_omega_addition, root=1)

comm.Barrier()


if rank == 0:

    G_total = Z_removal * G_omega_removal + Z_addition * G_omega_addition
    A_krylov = spectral_function(G_total)


    data = np.column_stack([
        omega,
        A_krylov,
        Z_addition * A_krylov_addition,
        Z_removal * A_krylov_removal,
    ])

    np.savetxt(
        "spectral_function.csv",
        data,
        delimiter=",",
        header="omega,A_total,A_addition,A_removal",
        comments="",
    )

    data = np.column_stack([
        omega,
        G_total.real,
        G_total.imag,
    ])

    np.savetxt(
        "greens_function.csv",
        data,
        delimiter=",",
        header="omega,G_real,G_imag",
        comments="",
    )
