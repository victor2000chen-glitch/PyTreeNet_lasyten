"""Run on LRZ after installing the compiled dependencies; no substitutes here."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing

import numpy as np
import pytest

pytestmark = pytest.mark.native


@pytest.fixture
def native_dependencies():
    pytest.importorskip("lasyten")
    import hdf5_io  # Required when LASyTen is installed; a missing I/O build is an error.


def test_native_process_workers_and_checkpoint_round_trip(native_dependencies, tmp_path):
    from examples.mpi_bug_smoke import make_problem, agreement
    from pytreenet_lasyten.time_evolution.bug import BUG, BUGConfig
    from pytreenet_lasyten.time_evolution.mpi_bug import MPIBUG
    from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
    from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
    from pytreenet_lasyten.ttno.ttno_class import TreeTensorNetworkOperator as TTNO
    state, operator, _ = make_problem("lasyten")
    state.save(tmp_path / "state.h5")
    operator.save(tmp_path / "operator.h5")
    restored = TTNS.load(tmp_path / "state.h5")
    restored_operator = TTNO.load(tmp_path / "operator.h5")
    assert agreement(restored, state) > 1-1e-12
    config = BUGConfig(max_bond_dim=4, rel_tol=0, total_tol=0, time_evo_mode=TimeEvoMode.RK45)
    options = dict(rtol=1e-10, atol=1e-12)
    args = (restored, restored_operator, .01, .02, {})
    serial = BUG(*args, config=deepcopy(config), solver_options=options)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as executor:
        parallel = MPIBUG(*args, config=deepcopy(config), solver_options=options,
                           executor=executor, subtree_tasks=3)
        for _ in range(2):
            parallel.run_one_time_step()
            serial.run_one_time_step()
            assert agreement(parallel.state, serial.state) > 1-1e-9


def test_native_excitation_accepts_nonroot_charge_centre(native_dependencies):
    from examples.mpi_bug_smoke import make_problem
    from pytreenet_lasyten.util.u1_excitations import create_impurity_excitations
    from pytreenet_lasyten.util.tensor_splitting import SplitMode
    state, _, _ = make_problem("lasyten")
    state.move_orthogonalization_center("a", mode=SplitMode.REDUCED)
    removal, addition, zr, za = create_impurity_excitations(state, impurity_node="a")
    np.testing.assert_allclose(zr+za, 1, atol=1e-10)
    np.testing.assert_allclose(removal.scalar_product(), 1, atol=1e-10)
    np.testing.assert_allclose(addition.scalar_product(), 1, atol=1e-10)
