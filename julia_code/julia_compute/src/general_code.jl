using PythonCall
include("numerical_methods.jl")

function make_ndn_iteration_for_python(initial_psi::PyArray{ComplexF64, 2},
                                       N::Int64, M::Int64, L2::Float64, T::Float64,
                                       size::Int64, beta1::PyArray{Float64, 1}, beta2::PyArray{Float64, 1}, gamma::PyArray{Float64, 1}, 
                                       E_sat::PyArray{Float64, 1}, alpha::PyArray{Float64, 1}, g_0::PyArray{Float64, 1}, 
                                       noise_amplitude::Float64,
                                       D::PyArray{ComplexF64})::Matrix{ComplexF64}
    initial_psi = pyconvert(Matrix{ComplexF64}, initial_psi)
    beta1 = pyconvert(Vector{Float64}, beta1)
    beta2 = pyconvert(Vector{Float64}, beta2)
    gamma = pyconvert(Vector{Float64}, gamma)
    E_sat = pyconvert(Vector{Float64}, E_sat)
    alpha = pyconvert(Vector{Float64}, alpha)
    g_0 = pyconvert(Vector{Float64}, g_0)
    D = pyconvert(Array{ComplexF64}, D)

    eq = EquationParameters(size=size, beta1=beta1, beta2=beta2, gamma=gamma, E_sat=E_sat, alpha=alpha, g_0=g_0, noise_amplitude=noise_amplitude)
    com = ComputationalParameters(N=N, M=M, L2=L2, T1=-T/2, T2=T/2)
    return make_ndn_iteration(initial_psi, eq, com, D)
end

function make_ndn_iteration_dcalc_for_python(initial_psi::PyArray{ComplexF64, 2},
                                             N::Int64, M::Int64, L2::Float64, T::Float64,
                                             size::Int64, beta1::PyArray{Float64, 1}, beta2::PyArray{Float64, 1}, gamma::PyArray{Float64, 1}, 
                                             E_sat::PyArray{Float64, 1}, alpha::PyArray{Float64, 1}, g_0::PyArray{Float64, 1}, 
                                             noise_amplitude::Float64,
                                             linear_coeffs_array::PyArray{Float64, 2},
                                             self_coupling::PyArray{Float64, 1})::Matrix{ComplexF64}
    initial_psi = pyconvert(Matrix{ComplexF64}, initial_psi)
    beta1 = pyconvert(Vector{Float64}, beta1)
    beta2 = pyconvert(Vector{Float64}, beta2)
    gamma = pyconvert(Vector{Float64}, gamma)
    E_sat = pyconvert(Vector{Float64}, E_sat)
    alpha = pyconvert(Vector{Float64}, alpha)
    g_0 = pyconvert(Vector{Float64}, g_0)
    linear_coeffs_array = pyconvert(Matrix{Float64}, linear_coeffs_array)
    self_coupling = pyconvert(Vector{Float64}, self_coupling)

    eq = EquationParameters(size=size, beta1=beta1, beta2=beta2, gamma=gamma, E_sat=E_sat, alpha=alpha, g_0=g_0, noise_amplitude=noise_amplitude)
    com = ComputationalParameters(N=N, M=M, L2=L2, T1=-0.5*T, T2=0.5*T)
    return make_ndn_iteration_dcalc(initial_psi, eq, com, linear_coeffs_array, self_coupling)
end
