"""Compiled U1xU1 checks; never replace LASyTen with a test double here."""
from collections import Counter

import numpy as np
import pytest

pytestmark = pytest.mark.native


@pytest.mark.parametrize("nb", [5, 7, 11])
@pytest.mark.parametrize("chi", [8, 16, 50])
def test_pair_charge_initializer_keeps_actual_qr_spaces(nb, chi):
    pytest.importorskip("lasyten")
    from pytreenet_lasyten.util.hk_u1_model import build_u1_t3ns
    from pytreenet_lasyten.util.u1u1_t3ns import check_u1u1_state
    from pytreenet_lasyten.util.abelian_charges import fixed_charge

    target = (3 * (nb + 1) // 2,) * 2
    state = build_u1_t3ns(
        norb=3, nb=nb, virtual_bond_dimension=chi, bath_bond_dimension=chi,
        total_particles_by_spin=target, symmetry="u1xu1", seed=11,
    )
    assert fixed_charge(state) == target
    assert check_u1u1_state(state) < 1e-10
    for node_id in state.nodes:
        if node_id == state.root_id:
            continue
        space = state.tensors[node_id].legs[0].dual
        counts = Counter(tuple(map(int, q)) for q in space.sectors_of_basis)
        assert dict(counts) == state.u1_info["bond_sectors"][node_id]
        assert 0 < int(space.dim) <= chi
    np.testing.assert_allclose(
        state.scalar_product(use_orthogonal_center=False), 1, atol=1e-10,
    )


def test_pair_charge_native_algorithms_against_exact_four_mode_model():
    pytest.importorskip("lasyten")
    from examples.native_symmetry_check import main
    main()
