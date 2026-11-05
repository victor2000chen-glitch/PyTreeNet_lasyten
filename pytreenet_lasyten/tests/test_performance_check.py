"""Exercise the reusable optimization check on available backends."""

import os

import pytest

from examples.bug_performance_check import check_case
from pytreenet_lasyten.time_evolution.time_evolution import TimeEvoMode
from pytreenet_lasyten.util import bug_native_speedups


@pytest.mark.parametrize("mode", [TimeEvoMode.RK23, TimeEvoMode.RK45])
@pytest.mark.parametrize("dt", [0.02, 0.02 - 0.004j])
def test_numpy_optimized_steps_and_settings_restoration(mode, dt):
    environment = dict(os.environ)
    settings = bug_native_speedups.settings()
    error, _ = check_case("numpy", mode, dt)
    assert error < 2e-8
    assert dict(os.environ) == environment
    assert bug_native_speedups.settings() == settings


@pytest.mark.native
@pytest.mark.parametrize("mode", [TimeEvoMode.RK23, TimeEvoMode.RK45])
@pytest.mark.parametrize("dt", [0.02, 0.02 - 0.004j])
def test_compiled_native_optimized_steps(mode, dt):
    pytest.importorskip("lasyten")
    check_case("lasyten", mode, dt)
