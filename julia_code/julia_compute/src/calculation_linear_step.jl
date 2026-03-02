using FFTW: fft, ifft, fftfreq
using LinearAlgebra: mul!
include("calculation_dmatrix.jl")

"""
Линейный шаг 
- D::Array{ComplexF64, 2} или D::Array{ComplexF64, 3}
параллелизм возможен для выполнения цикла по частотам в случае D::Array{ComplexF64, 3}
"""
function linear_step(psi::Array{ComplexF64, 2}, D::Array{ComplexF64})::Array{ComplexF64, 2}
    n, M = size(psi)
    if ndims(D) == 3
        # D должен быть размера (n, n, M)
        psi_f = fft(psi, 2)   # БПФ по второму измерению (время)
        psi_f_new = similar(psi_f)

        # Для каждой частоты m: psi_f_new[:, m] = D[:, :, m] * psi_f[:, m]
        Threads.@threads for m in 1:M
            # Умножение матрицы на вектор-столбец
            mul!(view(psi_f_new, :, m), view(D, :, :, m), view(psi_f, :, m))
        end

        return ifft(psi_f_new, 2)
    elseif ndims(D) == 2
        # D — матрица (n, n)
        # Умножение строк матрицы psi на элементы матрицы D; (n×M)
        return D * psi
    else
        throw(ArgumentError("Unexpectide D size"))
    end
end
