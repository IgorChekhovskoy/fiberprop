from numpy.typing import NDArray
import numpy as np
from juliacall import Main
import os

package_path = os.path.dirname(__file__)
file_path = os.path.join(package_path, "src", "computing_module.jl")
Main.include(file_path)

def make_iteration_julia(initial_psi: NDArray[np.complex128],
                         N: np.int64, M: np.int64, L2: np.float64, T: np.float64,
                         size: np.int16, beta1: NDArray[np.float64], beta2: NDArray[np.float64], gamma: NDArray[np.float64], 
                         E_sat: NDArray[np.float64], alpha: NDArray[np.float64], g_0: NDArray[np.float64], 
                         noise_amplitude: np.float64,
                         D: NDArray[np.complex128]) -> NDArray[np.complex128]:
    new_psi = Main.ComputingJuliaModule.make_iteration_for_python(initial_psi, N, M, L2, T, size, 
                                                                  beta1, beta2, gamma,  E_sat, alpha, g_0, 
                                                                  noise_amplitude, D)
    return np.array(new_psi, dtype=np.complex128)

def make_iteration_dcalc_julia(initial_psi: NDArray[np.complex128],
                               N: np.int64, M: np.int64, L2: np.float64, T: np.float64,
                               size: np.int16, beta1: NDArray[np.float64], beta2: NDArray[np.float64], gamma: NDArray[np.float64], 
                               E_sat: NDArray[np.float64], alpha: NDArray[np.float64], g_0: NDArray[np.float64], 
                               noise_amplitude: np.float64,
                               linear_coeffs_array: NDArray[np.float64],
                               self_coupling: NDArray[np.float64]) -> NDArray[np.complex128]:
    new_psi = Main.ComputingJuliaModule.make_iteration_dcalc_for_python(initial_psi, N, M, L2, T, size, 
                                                                        beta1, beta2, gamma,  E_sat, alpha, g_0, 
                                                                        noise_amplitude, linear_coeffs_array, self_coupling)
    return np.array(new_psi, dtype=np.complex128)
