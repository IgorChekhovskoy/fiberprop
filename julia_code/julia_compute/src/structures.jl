# Структура для вычислительных параметров
@kwdef struct ComputationalParameters
    N::Int64 = 0
    M::Int64 = 0
    L1::Float64 = 0.0
    L2::Float64 = 0.0
    T1::Float64 = 0.0
    T2::Float64 = 0.0
    # Вычисляемые поля
    h::Float64 = (N > 0) ? (L2 - L1) / N : 0.0
    tau::Float64 = (M > 0) ? (T2 - T1) / M : 0.0
end

# Изменяемая структура для физических параметров уравнения
@kwdef mutable struct EquationParameters
    size::Int64 = 1
    beta1::Vector{Float64} = zeros(Float64, size)  # длина size
    beta2::Vector{Float64} = zeros(Float64, size)  # длина size
    gamma::Vector{Float64} = zeros(Float64, size)  # длина size
    E_sat::Vector{Float64} = zeros(Float64, size)  # длина size
    alpha::Vector{Float64} = zeros(Float64, size)  # длина size
    g_0::Vector{Float64} = zeros(Float64, size)    # длина size
    noise_amplitude::Float64 = 0.0
end
