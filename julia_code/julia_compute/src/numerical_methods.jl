import Random
using Printf
include("calculation_nonlinear_step.jl")

"""
Один шаг численного моделирования с расчётом матрицы D (симметричное расщепление N-D-N с учётом усиления и шума).
*глубокое копирование массивов происходит при передаче julia управления памятью
- initial_psi: начальное поле, комплексный массив размера (n, M)
- linear_coeffs_array: матрица связи (n, n)
- self_coupling: массив (n) коэффициентов самосвязи

Возвращает следующий шаг решения.
"""
function make_ndn_iteration_dcalc(initial_psi::Array{ComplexF64, 2},
                                  eq::EquationParameters,
                                  com::ComputationalParameters,
                                  linear_coeffs_array::Matrix{Float64},
                                  self_coupling::Vector{Float64})::Array{ComplexF64, 2}
    psi = initial_psi  # переданный объект изменяется в результате нелинейного шага
    n, M = size(psi)

    omega = 2π * fftfreq(M, 1.0/com.tau)
    E_sat = eq.E_sat
    gamma_h = 0.5*com.h * eq.gamma
    g0_h = 0.5*com.h * eq.g_0
    exp_g0h = exp.(g0_h)
    exp_2g0h = exp.(com.h * eq.gamma)

    # расчёт матрицы для линейного шага
    D = calculate_D_matrix(eq, com, 
                           linear_coeffs_array, self_coupling,
                           com.h, omega)

    # Половина нелинейного шага
    nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, com.tau)

    # Линейный шаг
    psi = linear_step(psi, D)

    # Вторая половина нелинейного шага
    nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, com.tau)

    # Добавление шума
    if eq.noise_amplitude != 0.0
        # равномерный шум на [-1,1) для действительной и мнимой частей
        noise_real = 2.0 * rand(n, M) .- 1.0
        noise_imag = 2.0 * rand(n, M) .- 1.0
        noise = eq.noise_amplitude * (noise_real .+ 1im .* noise_imag)
        psi .+= noise
    end

    return psi
end

"""
Один шаг численного моделирования без расчёта матрицы D (симметричное расщепление N-D-N с учётом усиления и шума).
*глубокое копирование массивов происходит при передаче julia управления памятью
- initial_psi: начальное поле, комплексный массив размера (n, M)
- linear_coeffs_array: матрица связи (n, n)
- D: массив (n, n, M) или (n, n) экспонента матрицы для линейного шага

Возвращает следующий шаг решения.
"""
function make_ndn_iteration(initial_psi::Array{ComplexF64, 2},
                            eq::EquationParameters,
                            com::ComputationalParameters,
                            D::Array{ComplexF64})::Array{ComplexF64, 2}
    psi = initial_psi  # переданный объект изменяется в результате первого нелинейного шага полушага
    n, M = size(psi)

    E_sat = eq.E_sat
    gamma_h = 0.5*com.h * eq.gamma
    g0_h = 0.5*com.h * eq.g_0
    exp_g0h = exp.(g0_h)
    exp_2g0h = exp.(com.h * eq.g_0)
    
    # Половина нелинейного шага
    nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, com.tau)

    # Линейный шаг
    psi = linear_step(psi, D)

    # Вторая половина нелинейного шага
    nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, com.tau)

    # Добавление шума
    if eq.noise_amplitude != 0.0
        # равномерный шум на [-1,1) для действительной и мнимой частей
        noise_real = 2.0 * rand(n, M) .- 1.0
        noise_imag = 2.0 * rand(n, M) .- 1.0
        noise = eq.noise_amplitude * (noise_real .+ 1im .* noise_imag)
        psi .+= noise
    end

    return psi
end
