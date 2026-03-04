module FiberpropSSFM

using FFTW
using PythonCall

# ── Module-level FFTW plan cache ─────────────────────────────────────────────
# Key: (n_cores::Int, M::Int)
# Value: NamedTuple (fwd=..., inv=...)
const _plan_cache = Dict{Tuple{Int,Int}, NamedTuple}()

# ─────────────────────────────────────────────────────────────────────────────
# get_plans: retrieve (or create) pre-planned in-place FFT transforms.
# Julia convention: psi is (M, n_cores) — column-major, time on dim 1.
# ─────────────────────────────────────────────────────────────────────────────
function get_plans(n_cores::Int, M::Int)
    key = (n_cores, M)
    if !haskey(_plan_cache, key)
        buf = zeros(ComplexF64, M, n_cores)
        fwd = plan_fft!(similar(buf), 1; flags = FFTW.PATIENT)
        inv = plan_ifft!(similar(buf), 1; flags = FFTW.PATIENT)
        _plan_cache[key] = (fwd = fwd, inv = inv)
    end
    return _plan_cache[key]
end

# ─────────────────────────────────────────────────────────────────────────────
# nonlinear_step!
# psi    : (M, n_cores)  ComplexF64  in-place
# gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat : (n_cores,) Float64
# energy_in : (n_cores,) Float64
# P     : (M, n_cores)  Float64   |psi|²
# ─────────────────────────────────────────────────────────────────────────────
function nonlinear_step!(psi::Matrix{ComplexF64},
                          gamma_h::Vector{Float64},
                          g0_h::Vector{Float64},
                          exp_g0h::Vector{Float64},
                          exp_2g0h::Vector{Float64},
                          E_sat::Vector{Float64},
                          energy_in::Vector{Float64},
                          P::Matrix{Float64})
    n_cores = size(psi, 2)
    for i in 1:n_cores
        if g0_h[i] == 0.0
            # gain-free: pure Kerr phase rotation
            @views psi[:, i] .*= exp.(1im .* gamma_h[i] .* P[:, i])
        else
            # saturated gain core — replicate Python formulas
            Ek   = energy_in[i]
            esat = E_sat[i]
            g0h  = g0_h[i]
            eg1  = exp_g0h[i]
            eg2  = exp_2g0h[i]
            gamh = gamma_h[i]

            @views begin
                Pk  = P[:, i]
                psi_i = psi[:, i]

                E   = sqrt((Ek^2 + 2*Ek*esat) * eg2 + esat^2) - esat
                C   = -gamh .* Pk .* (Ek + esat - esat*log(Ek + 2*esat)) ./ (g0h*Ek) .+ angle.(psi_i)
                Pn  = Pk .* eg1 .* sqrt((Ek + 2*esat)/Ek) .* sqrt(E / (E + 2*esat))
                phi = gamh .* Pk .* (E + esat - esat*log(E + 2*esat)) ./ (g0h*Ek) .+ C
                psi[:, i] = sqrt.(Pn) .* exp.(1im .* phi)
            end
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# py_nonlinear_step: Python-callable entry point.
# Note: function names ending in ! map to _b in juliacall; use plain names here
# so Python can call them as jl.FiberpropSSFM.py_nonlinear_step(...).
# ─────────────────────────────────────────────────────────────────────────────
function py_nonlinear_step(psi_py, gamma_h_py, g0_h_py, exp_g0h_py, exp_2g0h_py,
                            E_sat_py, energy_in_py, P_py)
    # Convert Python arrays → native Julia (copy-in, compute, copy-out)
    psi_np = pyconvert(Array{ComplexF64, 2}, psi_py)   # (n_cores, M) C-order → copy
    # Python psi shape is (n_cores, M); we work in (M, n_cores) Julia layout
    psi    = permutedims(psi_np, (2, 1))                # (M, n_cores)
    P_np   = pyconvert(Array{Float64, 2}, P_py)
    P      = permutedims(P_np, (2, 1))                  # (M, n_cores)

    gamma_h   = pyconvert(Vector{Float64}, gamma_h_py)
    g0_h      = pyconvert(Vector{Float64}, g0_h_py)
    exp_g0h   = pyconvert(Vector{Float64}, exp_g0h_py)
    exp_2g0h  = pyconvert(Vector{Float64}, exp_2g0h_py)
    E_sat     = pyconvert(Vector{Float64}, E_sat_py)
    energy_in = pyconvert(Vector{Float64}, energy_in_py)

    nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy_in, P)

    # Write back: psi is (M, n_cores), psi_py expects (n_cores, M)
    psi_py .= permutedims(psi, (2, 1))
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# linear_step!
# psi      : (M, n_cores)  ComplexF64  in-place
# has_beta : Bool
# D        : (M, n_cores, n_cores) if has_beta; (n_cores, n_cores) otherwise
# plans    : NamedTuple(fwd, inv)
# ─────────────────────────────────────────────────────────────────────────────
function linear_step!(psi::Matrix{ComplexF64}, has_beta::Bool,
                       D_full, plans::NamedTuple)
    if has_beta
        plans.fwd * psi   # in-place FFT along dim 1 (time axis)

        M = size(psi, 1)

        if ndims(D_full) == 3
            # D_full is (M, n_cores, n_cores) — apply per-frequency matrix multiply
            for m in 1:M
                @views psi[m, :] = D_full[m, :, :] * psi[m, :]
            end
        else
            # D_full is (n_cores, n_cores) — same matrix every frequency
            for m in 1:M
                @views psi[m, :] = D_full * psi[m, :]
            end
        end

        plans.inv * psi   # in-place IFFT along dim 1
    else
        # coupling-only: D is (n_cores, n_cores); apply to every time point
        psi .= psi * D_full'
    end
    return nothing
end

# Python-callable entry point for linear_step
function py_linear_step(psi_py, has_beta_py, D_py, plans)
    has_beta = pyconvert(Bool, has_beta_py)
    psi_np   = pyconvert(Array{ComplexF64, 2}, psi_py)   # (n_cores, M)
    psi      = permutedims(psi_np, (2, 1))               # (M, n_cores)

    if has_beta
        # Python D shape: (n_cores, n_cores, M) → Julia (M, n_cores, n_cores)
        D_np = pyconvert(Array{ComplexF64, 3}, D_py)     # (n_cores, n_cores, M)
        D    = permutedims(D_np, (3, 1, 2))              # (M, n_cores, n_cores)
    else
        D = pyconvert(Array{ComplexF64, 2}, D_py)        # (n_cores, n_cores)
    end

    linear_step!(psi, has_beta, D, plans)

    result = permutedims(psi, (2, 1))   # (n_cores, M)
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# py_ssfm_order2_ndn: Python-callable full N-D-N split-step entry point
# ─────────────────────────────────────────────────────────────────────────────
function py_ssfm_order2_ndn(psi_py, energy_py,
                              gamma_h_py, g0_h_py,
                              exp_g0h_py, exp_2g0h_py,
                              E_sat_py,
                              D_py, has_beta_py,
                              taper_py,
                              tau_py, damp_length_py,
                              noise_amplitude_py)

    has_beta        = pyconvert(Bool, has_beta_py)
    tau             = pyconvert(Float64, tau_py)
    damp_length     = pyconvert(Float64, damp_length_py)
    noise_amplitude = pyconvert(Float64, noise_amplitude_py)

    psi_np = pyconvert(Array{ComplexF64, 2}, psi_py)   # (n_cores, M)
    psi    = permutedims(psi_np, (2, 1))               # (M, n_cores)

    energy = pyconvert(Vector{Float64}, energy_py)

    gamma_h  = pyconvert(Vector{Float64}, gamma_h_py)
    g0_h     = pyconvert(Vector{Float64}, g0_h_py)
    exp_g0h  = pyconvert(Vector{Float64}, exp_g0h_py)
    exp_2g0h = pyconvert(Vector{Float64}, exp_2g0h_py)
    E_sat    = pyconvert(Vector{Float64}, E_sat_py)

    if has_beta
        D_np = pyconvert(Array{ComplexF64, 3}, D_py)   # (n_cores, n_cores, M)
        D    = permutedims(D_np, (3, 1, 2))            # (M, n_cores, n_cores)
    else
        D = pyconvert(Array{ComplexF64, 2}, D_py)      # (n_cores, n_cores)
    end

    taper_vec = pyconvert(Vector{Float64}, taper_py)
    has_taper = length(taper_vec) > 0

    n_cores = size(psi, 2)
    M       = size(psi, 1)
    plans   = get_plans(n_cores, M)

    # ── Compute initial power ─────────────────────────────────────────────────
    P_real = real.(psi .* conj.(psi))

    # Update energy for gain cores before first half NL step
    gain = g0_h .!= 0.0
    if any(gain)
        for i in 1:n_cores
            if g0_h[i] != 0.0
                energy[i] = sum(P_real[:, i]) * tau
            end
        end
    end

    # ── ½ Nonlinear step ─────────────────────────────────────────────────────
    nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy, P_real)

    # ── (optional) absorbing boundary ────────────────────────────────────────
    if has_taper && damp_length > 0.0
        for i in 1:n_cores
            @views psi[:, i] .*= taper_vec
        end
    end

    # ── Linear step ──────────────────────────────────────────────────────────
    linear_step!(psi, has_beta, D, plans)

    # ── (optional) absorbing boundary ────────────────────────────────────────
    if has_taper && damp_length > 0.0
        for i in 1:n_cores
            @views psi[:, i] .*= taper_vec
        end
    end

    # ── Recompute power after linear step ────────────────────────────────────
    P_real = real.(psi .* conj.(psi))

    # Update energy for gain cores before second half NL step
    if any(gain)
        for i in 1:n_cores
            if g0_h[i] != 0.0
                energy[i] = sum(P_real[:, i]) * tau
            end
        end
    end

    # ── ½ Nonlinear step ─────────────────────────────────────────────────────
    nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy, P_real)

    # ── (optional) absorbing boundary ────────────────────────────────────────
    if has_taper && damp_length > 0.0
        for i in 1:n_cores
            @views psi[:, i] .*= taper_vec
        end
    end

    # ── (optional) additive noise ────────────────────────────────────────────
    if noise_amplitude > 0.0
        noise = noise_amplitude .* (
            (rand(Float64, M, n_cores) .* 2 .- 1) .+
            1im .* (rand(Float64, M, n_cores) .* 2 .- 1)
        )
        psi .+= noise
    end

    # ── Write back to Python buffer ───────────────────────────────────────────
    psi_py .= permutedims(psi, (2, 1))   # (n_cores, M)
    energy_py .= energy
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Module __init__: pre-warm FFTW plans and trigger JIT compilation
# ─────────────────────────────────────────────────────────────────────────────
function __init__()
    # Pre-create plans for common fiberprop shapes
    for (nc, M) in [(1, 512), (7, 1024), (19, 2048)]
        get_plans(nc, M)
    end

    # Dummy call to pre-compile nonlinear_step! for ComplexF64 arrays
    _dummy_M, _dummy_nc = 8, 1
    _psi   = zeros(ComplexF64, _dummy_M, _dummy_nc)
    _P     = zeros(Float64,    _dummy_M, _dummy_nc)
    _gamma = zeros(Float64, _dummy_nc)
    _g0    = zeros(Float64, _dummy_nc)
    _eg1   = ones(Float64,  _dummy_nc)
    _eg2   = ones(Float64,  _dummy_nc)
    _esat  = ones(Float64,  _dummy_nc)
    _ein   = zeros(Float64, _dummy_nc)
    nonlinear_step!(_psi, _gamma, _g0, _eg1, _eg2, _esat, _ein, _P)

    # Dummy call to pre-compile linear_step! for has_beta=true
    _plans = get_plans(_dummy_nc, _dummy_M)
    _D3 = zeros(ComplexF64, _dummy_M, _dummy_nc, _dummy_nc)
    linear_step!(_psi, true, _D3, _plans)
    _D2 = zeros(ComplexF64, _dummy_nc, _dummy_nc)
    linear_step!(_psi, false, _D2, _plans)

    precompile(get_plans, (Int, Int))
    precompile(nonlinear_step!,
               (Matrix{ComplexF64}, Vector{Float64}, Vector{Float64},
                Vector{Float64}, Vector{Float64}, Vector{Float64},
                Vector{Float64}, Matrix{Float64}))
end

end # module FiberpropSSFM
