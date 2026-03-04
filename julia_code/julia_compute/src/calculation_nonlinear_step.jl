include("calculation_linear_step.jl")

"""
in-place обновление psi (n, M).
"""

function get_power(psi_slice::ComplexF64)::Float64
    return real(psi_slice)^2 + imag(psi_slice)^2
end

# """
# in-place обновление psi (n, M).
# *параллелизм по времени
# """
# function nonlinear_step!(psi::AbstractMatrix{ComplexF64},
#                          gamma_h::Array{Float64, 1},
#                          g0_h::Array{Float64, 1},
#                          exp_g0h::Array{Float64, 1},
#                          exp_2g0h::Array{Float64, 1},
#                          E_sat::Array{Float64, 1}, 
#                          tau::Float64)
#     n, M = size(psi)
    
#     no_gain = findall(iszero, g0_h)
#     gamh_ng = view(gamma_h, no_gain)
#     Threads.@threads for j in 1:M
#         psi[no_gain, j] .*= exp.(1im .* gamh_ng .* get_power.(view(psi, no_gain, j)))
#     end

#     gain = findall(!iszero, g0_h)
#     e_arr = zeros(Float64, n)
#     Threads.@threads for j in 1:M
#         e_arr[gain] .+= get_power.(view(psi, gain, j))
#     end
#     e_arr .*= tau

#     ek = view(e_arr, gain)
#     esat = view(E_sat, gain)
#     g0hk = view(g0_h, gain)
#     eg1 = view(exp_g0h, gain)
#     eg2 = view(exp_2g0h, gain)
#     gamh = view(gamma_h, gain)

#     Threads.@threads for j in 1:M
#         pk = get_power.(view(psi, gain, j))
#         phik = angle.(view(psi, gain, j))

#         new_ek = @. sqrt((ek^2 + 2 * ek * esat) * eg2 + esat^2) - esat

#         c_k = @. -gamh * (ek + esat - esat * log(ek + 2*esat)) / (g0hk * ek) * pk + phik

#         new_pk = @. eg1 * sqrt((ek + 2*esat) / ek) * sqrt(new_ek / (new_ek + 2*esat)) * pk

#         new_phik = @. gamh * (new_ek + esat - esat * log(new_ek + 2*esat)) / (g0hk * ek) * pk + c_k

#         psi[gain, j] = @. sqrt(new_pk) * exp(1im * new_phik)
#     end

#     return nothing
# end

"""
in-place обновление psi (n, M).
*параллелизм по сердцевинам
"""
function nonlinear_step!(psi::AbstractMatrix{ComplexF64},
                         gamma_h::Array{Float64, 1},
                         g0_h::Array{Float64, 1},
                         exp_g0h::Array{Float64, 1},
                         exp_2g0h::Array{Float64, 1},
                         E_sat::Array{Float64, 1}, 
                         tau::Float64)

    # Индексы сердцевин без усиления (g0_h == 0)
    no_gain = findall(iszero, g0_h)
    Threads.@threads for i in no_gain
        gamh = gamma_h[i]
        psi[i, :] .*= exp.(1im .* gamh .* get_power.(view(psi, i, :)))
    end

    # Индексы сердцевин с усилением (g0_h != 0)
    gain = findall(!iszero, g0_h)
    Threads.@threads for i in gain
        ek = sum(get_power, view(psi, i, :)) * tau
        esat = E_sat[i]
        g0hk = g0_h[i]
        eg1 = exp_g0h[i]
        eg2 = exp_2g0h[i]
        gamh = gamma_h[i]

        pk = get_power.(view(psi, i, :))
        phik = angle.(view(psi, i, :))

        new_ek = sqrt((ek^2 + 2 * ek * esat) * eg2 + esat^2) - esat

        c_k = (-gamh * (ek + esat - esat * log(ek + 2*esat)) / (g0hk * ek)) .* pk .+ phik

        new_pk = (eg1 * sqrt((ek + 2*esat) / ek) * sqrt(new_ek / (new_ek + 2*esat))) .* pk

        new_phik = (gamh * (new_ek + esat - esat * log(new_ek + 2*esat)) / (g0hk * ek)) .* pk .+ c_k

        psi[i, :] = sqrt.(new_pk) .* exp.(1im .* new_phik)
    end

    return nothing
end
