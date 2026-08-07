import pytest

from .conftest import run_with_reference

base_folder = "darcy"
fields = ["MBF"]


def test_pressure_dirichlet(n_proc):
    test_folder = "pressure_dirichlet"
    run_with_reference(base_folder, test_folder, fields, n_proc, 2)
