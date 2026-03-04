using LinearAlgebra: exp
include("structures.jl")

"""
   Вычисляет матрицу D размера (n, n, M) или (n, n) если производные от константы распространения нулевые.
- linear_coeffs_array: матрица связи (n, n) (Float64)
- self_coupling: набор (n, N) поправок на диагонали матрицы связи (self coupling) при фиксированном z
- h: шаг по z (в зависимости от схемы может отличаться от com.h)
- omega: частотная сетка
Возвращает массив комплексных чисел размером (n, n, M).
"""
function calculate_D_matrix(eq::EquationParameters,
                            com::ComputationalParameters,
                            linear_coeffs_array::Matrix{Float64},
                            self_coupling::AbstractVector{Float64},
                            h_coef::Float64,
                            omega::AbstractVector{Float64})::Array{ComplexF64}
    n = eq.size
    M = com.M

    # Извлекаем векторы параметров (все Float64)
    C = linear_coeffs_array               # (n, n)
    α = eq.alpha                          # (n)
    g0 = eq.g_0                           # (n)
    β2 = eq.beta2                         # (n)
    β1 = eq.beta1                         # (n)
    sc = self_coupling                    # (n)

    if all(iszero, β1) && all(iszero, β2)
        return calculate_coupling_matrix_exponent(eq, linear_coeffs_array, self_coupling, h_coef)
    end

    # Частотная сетка (циклические частоты)
    ω = omega  # (M)

    # Итоговая матрица D (n, n, M)
    D = Array{ComplexF64, 3}(undef, n, n, M)

    # Цикл по всем частотам
    Threads.@threads for m in 1:M
        w = ω[m]
        # Матрица A = h * (1im*C + diag_contrib + offdiag_contrib)
        A = zeros(ComplexF64, n, n)

        # Вклад от матрицы связи (1im * C)
        for i in 1:n, j in 1:n
            A[i, j] += 1im * C[i, j]
        end

        # Диагональный вклад (члены, зависящие от частоты)
        for i in 1:n
            diag_term = (2im * sc[i] + 1im * β2[i] * w^2 - (α[i] + g0[i])) * 0.5
            A[i, i] += diag_term
        end

        # Недиагональный вклад от beta1 (там, где C не равен нулю)
        for i in 1:n, j in 1:n
            if C[i, j] != 0.0
                A[i, j] += 1im * β1[j] * w
            end
        end

        # Умножаем на шаг
        A .*= h_coef

        # Экспонента от матрицы
        D[:, :, m] = exp(A)
    end

    return D
end

"""
   Вычисляет матрицу D размера (n, n).
- linear_coeffs_array: матрица связи (n, n) (Float64)
- self_coupling: набор (n, N) поправок на диагонали матрицы связи (self coupling)
- h: шаг по z
- omega: частотная сетка
Возвращает массив комплексных чисел размером (n, n).
"""
function calculate_coupling_matrix_exponent(eq::EquationParameters,
                                            linear_coeffs_array::Matrix{Float64},
                                            self_coupling::AbstractVector{Float64},
                                            h::Float64)::Array{ComplexF64}
    n = eq.size

    # Извлекаем векторы параметров (все Float64)
    C = linear_coeffs_array               # (n, n)
    α = eq.alpha                          # (n)
    g0 = eq.g_0                           # (n)
    sc = self_coupling                    # (n)

    # Матрица A = h * (1im*C + diag_contrib + offdiag_contrib)
    A = zeros(ComplexF64, n, n)

    # Вклад от матрицы связи (1im * C)
    for i in 1:n, j in 1:n
        A[i, j] += 1im * C[i, j]
    end

    # Диагональный вклад (члены, зависящие от частоты)
    for i in 1:n
        diag_term = (2im * sc[i] - (α[i] + g0[i])) * 0.5
        A[i, i] += diag_term
    end

    # Умножаем на шаг h
    A .*= h
    return exp(A)
end
