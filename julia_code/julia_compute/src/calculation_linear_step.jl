using FFTW: fft, ifft, fftfreq, PATIENT, plan_fft!, plan_fft, plan_ifft!, plan_ifft
using LinearAlgebra: mul!
include("calculation_dmatrix.jl")

# ── Module-level FFTW plan cache ─────────────────────────────────────────────
# Key: (n::Int64, M::Int64)
# Value: NamedTuple (fwd=..., inv=...)
const _plan_cache = Dict{Tuple{Int64,Int64}, NamedTuple}()

# ─────────────────────────────────────────────────────────────────────────────
# get_plans: retrieve (or create) pre-planned in-place FFT transforms.
# Julia convention: psi is (n, M) — column-major, time on dim 2.
# ─────────────────────────────────────────────────────────────────────────────
function get_plans(n::Int64, M::Int64)
    key = (n, M)
    if !haskey(_plan_cache, key)
        buf = zeros(ComplexF64, n, M)
        fwd! = plan_fft!(similar(buf), 2; flags=PATIENT)
        fwd = plan_fft(similar(buf), 2; flags=PATIENT)
        inv! = plan_ifft!(similar(buf), 2; flags=PATIENT)
        inv = plan_ifft(similar(buf), 2; flags=PATIENT)
        _plan_cache[key] = (fwd! = fwd!, inv! = inv!, fwd = fwd, inv = inv)
    end
    return _plan_cache[key]
end

"""
Линейный шаг без выделения новой памяти, параллельная реализация тратит часть ресурсов на создание потоков
- D::Array{ComplexF64, 2} или D::Array{ComplexF64, 3}
- plans::NamedTuple план преобразования Фурье
без параллелизма (иначе медленно)
"""
function parallel_linear_step!(psi::Array{ComplexF64, 2}, D::Array{ComplexF64}, plans::NamedTuple)
    n, M = size(psi)
    if ndims(D) == 3
        # D должен быть размера (n, n, M)
        mul!(psi, plans.fwd!, psi)  # БПФ на месте по второму измерению (время)

        new_psi = similar(psi)
        # Для каждой частоты m: new_psi[:, m] = D[:, :, m] * psi_f[:, m]
        Threads.@threads for m in 1:M
            # Умножение матрицы на вектор-столбец
            # @views psi[:, m] = D[:, :, m] * psi[:, m]
            mul!(view(new_psi, :, m), view(D, :, :, m), view(psi, :, m))
        end

        mul!(psi, plans.inv, new_psi)  # обратное БПФ на месте по второму измерению (время)
    elseif ndims(D) == 2
        # D — матрица (n, n)
        # Умножение строк матрицы psi на элементы матрицы D; (n×M)
        psi = D * psi
    else
        throw(ArgumentError("Unexpectide D size"))
    end
end

"""
Линейный шаг без выделения новой памяти, однопоточная реализация
- D::Array{ComplexF64, 2} или D::Array{ComplexF64, 3}
- plans::NamedTuple план преобразования Фурье
без параллелизма (иначе медленно)
"""
function linear_step!(psi::Array{ComplexF64, 2}, D::Array{ComplexF64}, plans::NamedTuple)
    n, M = size(psi)
    if ndims(D) == 3
        # D должен быть размера (n, n, M)
        mul!(psi, plans.fwd!, psi)  # БПФ на месте по второму измерению (время)

        new_psi = similar(psi)
        # Для каждой частоты m: new_psi[:, m] = D[:, :, m] * psi_f[:, m]
        for m in 1:M
            # Умножение матрицы на вектор-столбец
            # @views psi[:, m] = D[:, :, m] * psi[:, m]
            mul!(view(new_psi, :, m), view(D, :, :, m), view(psi, :, m))
        end

        mul!(psi, plans.inv, new_psi)  # обратное БПФ на месте по второму измерению (время)
    elseif ndims(D) == 2
        # D — матрица (n, n)
        # Умножение строк матрицы psi на элементы матрицы D; (n×M)
        psi = D * psi
    else
        throw(ArgumentError("Unexpectide D size"))
    end
end
