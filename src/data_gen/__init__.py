"""
Data generation modules for demo families.
"""

from .phase_family import generate_phase_family_trajectory, generate_phase_family_dataset
from .screened_poisson import generate_screened_poisson_trajectory, generate_screened_poisson_dataset
from .helmholtz import generate_helmholtz_trajectory, generate_helmholtz_dataset
from .wave_equation import generate_wave_equation_trajectory, generate_wave_equation_dataset

__all__ = [
    "generate_phase_family_trajectory",
    "generate_phase_family_dataset",
    "generate_screened_poisson_trajectory",
    "generate_screened_poisson_dataset",
    "generate_helmholtz_trajectory",
    "generate_helmholtz_dataset",
    "generate_wave_equation_trajectory",
    "generate_wave_equation_dataset",
]