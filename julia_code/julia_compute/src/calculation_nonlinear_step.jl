include("calculation_linear_step.jl")

"""
in-place обновление psi (n, M).
"""
function nonlinear_step!(psi::AbstractMatrix{ComplexF64},
                         gamma_h::Array{Float64, 1},
                         g0_h::Array{Float64, 1},
                         exp_g0h::Array{Float64, 1},
                         exp_2g0h::Array{Float64, 1},
                         E_sat::Array{Float64, 1}, 
                         tau::Float64)
    n, M = size(psi)

    # Индексы сердцевин без усиления (g0_h == 0)
    no_gain = findall(iszero, g0_h)
    Threads.@threads for i in no_gain
        gamh = gamma_h[i]
        for j in 1:M
            psi[i, j] *= exp(1im * gamh * (real(psi[i, j])^2 + imag(psi[i, j])^2))
        end
    end

    gain = findall(!iszero, g0_h)
    Threads.@threads for i in gain
        ek = 0.0
        for j in 1:M
            ek += real(psi[i, j])^2 + imag(psi[i, j])^2
        end
        ek *= tau
        esat = E_sat[i]
        g0hk = g0_h[i]
        eg1 = exp_g0h[i]
        eg2 = exp_2g0h[i]
        gamh = gamma_h[i]

        for j in 1:M
            pk = real(psi[i, j])^2 + imag(psi[i, j])^2
            phik = angle(psi[i, j])

            new_ek = sqrt((ek^2 + 2 * ek * esat) * eg2 + esat^2) - esat

            c_k = -gamh * (ek + esat - esat * log(ek + 2*esat)) / (g0hk * ek) * pk + phik

            new_pk = eg1 * sqrt((ek + 2*esat) / ek) * sqrt(new_ek / (new_ek + 2*esat)) * pk

            phi = gamh * (new_ek + esat - esat * log(new_ek + 2*esat)) / (g0hk * ek) * pk + c_k

            psi[i, j] = sqrt(new_pk) * exp(1im * phi)
        end
    end

    return nothing
end
