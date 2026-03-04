# Research: Julia SSFM Solver Integration

**Branch**: `001-julia-ssfm-solver` | **Date**: 2026-03-04

---

## Decision 1: Python-Julia Bridge Library

**Decision**: Use `juliacall` (Python) + `PythonCall.jl` (Julia) as the interop bridge.

**Rationale**: `juliacall` is the standard, actively maintained Python-side library for calling Julia from Python. It is paired with `PythonCall.jl` on the Julia side and is the recommended approach by the Julia community for scientific computing interop. The alternative `PyCall.jl` / `pyjulia` pairing is older, less actively maintained, and has known issues with complex initialization.

**Alternatives considered**:
- `pyjulia` + `PyCall.jl`: Older ecosystem, initialization is more fragile, requires a separate `python-jl` launcher in some configurations. Rejected.
- Subprocess / IPC (e.g., calling Julia as a subprocess): Too high overhead per step call in a tight integration loop. Rejected.

---

## Decision 2: Array Data Passing Strategy

**Decision**: Pass NumPy `complex128` arrays to Julia as zero-copy `PyArray{ComplexF64, N}` views (the default `juliacall` behavior). At the Julia function boundary, copy into a native Julia `Array` once for computation, then write results back via element-wise assignment to the Python buffer.

**Rationale**:
- `juliacall` passes C-contiguous NumPy arrays as zero-copy `PyArray` wrappers by default — no copy at the Python→Julia call boundary.
- **However**: NumPy uses row-major (C-order) layout; Julia uses column-major (Fortran-order). For 2D arrays of shape `(n_cores, M)`, the memory layout is physically transposed. Working directly on `PyArray` with Julia's column-major FFT routines requires `permutedims` at the boundary.
- **Recommended pattern**: At the Julia function boundary, call `Array(psi_py)` once (one copy), perform all computation in native Julia arrays (enabling full FFTW optimization), then write results back via `psi_py .= result`. This "copy-in, compute, copy-out" pattern is standard for Julia/Python interop in hot loops and gives better overall performance than operating on `PyArray` directly.
- The `(n_cores, M)` shape in NumPy corresponds to `(M, n_cores)` in Julia's column-major storage. The `linear_step` applies FFT along the time axis (axis=1 in Python = dim 1 in Julia's column-major `(M, n_cores)` matrix), which aligns naturally.

**Alternatives considered**:
- Operate entirely on `PyArray` without copying: Avoids copy but loses FFTW plan optimization and requires careful indexing. Profiling expected to be slower for large M. Deferred to optimization phase if needed.

---

## Decision 3: FFT Implementation in Julia

**Decision**: Use `FFTW.jl` with pre-planned transforms (`plan_fft!` / `plan_ifft!` with `FFTW.PATIENT` flag). Plans are created once per unique `(n_cores, M)` shape and cached in a module-level dictionary.

**Rationale**:
- `FFTW.jl` wraps the same FFTW3 library used by SciPy's `scipy.fft`. Performance is equivalent or better because Julia enables `FFTW.PATIENT` planning (longer measurement during plan creation, better kernel selection for repeated calls) more naturally than SciPy's API.
- Pre-planned in-place transforms (`plan_fft!(buf, dim)`) avoid all allocations per step — critical for a hot simulation loop.
- `FFTW.set_num_threads(N)` controls FFTW's internal thread count independently of Julia's `Threads.nthreads()`.

**Alternatives considered**:
- Julia's standard `LinearAlgebra.fft`: Backed by FFTW.jl anyway; `FFTW.jl` gives direct access to plan caching and flags. No reason to use the lower-level wrapper.

---

## Decision 4: Julia Dependency Management

**Decision**: Two-file approach:
1. `julia/Project.toml` + `julia/Manifest.toml` — the Julia environment definition (committed to repo, `Manifest.toml` for full reproducibility).
2. `fiberprop/juliapkg.json` — consumed by `juliapkg` on the Python side; automatically activates the correct Julia environment at startup.

**Rationale**:
- `juliapkg` (installed automatically with `juliacall`) reads `juliapkg.json` from the Python package directory and calls `Pkg.resolve()` automatically before the Julia session starts. This is the recommended approach for distributing Python packages with Julia dependencies.
- `Project.toml` provides human-readable version constraints. `Manifest.toml` provides exact lockfile reproducibility across machines.

**juliapkg.json structure**:
```json
{
    "julia": "1.9",
    "packages": {
        "FFTW": {"uuid": "7a1cc6ca-52ef-59f5-83cd-3a7055c09341", "version": "1"},
        "PythonCall": {"uuid": "6099a3de-0909-46bc-b1f4-468b9a2dfc0d", "version": "0.9"}
    }
}
```

**Alternatives considered**:
- User manages Julia environment manually: Not reproducible, poor onboarding experience. Rejected.
- `CondaPkg.jl` (juliacall's default conda-based management): More complex for a purely Julia-extension use case. Rejected in favor of direct `juliapkg.json`.

---

## Decision 5: JIT Warmup Strategy

**Decision**: Julia module-level `__init__()` function pre-warms all hot functions with representative array sizes (e.g., `(M=1024, n_cores=7)`) and pre-creates FFTW plans. On the Python side, the Julia module is loaded lazily (on first call to `ssfm_order2_ndn_julia`) using a module-level flag, and the user is warned once about first-call initialization time.

**Rationale**:
- Julia compiles functions on first call for each unique set of argument types. For `ComplexF64` arrays with typical fiberprop sizes, compilation takes 0.5–3 seconds.
- Running warmup in Julia's `__init__()` ensures compilation happens once when the module is first loaded, not on the first simulation step (where it would be invisible and confusing).
- `precompile()` directives are also added for type-dispatch caching across sessions, reducing future startup time.

**Alternatives considered**:
- No warmup (accept first-call latency): Creates silent slowdown on first simulation step. Rejected.
- Warmup at Python import time (top-level `import juliacall`): Forces Julia startup cost even when the Julia method is never used. Rejected in favor of lazy loading.

---

## Decision 6: Constitution Principle II (Dual Backend Parity)

**Decision**: The Julia backend is treated as an **alternative CPU implementation** of an existing method (`ssfm_order2_ndn`), not a new SSFM feature. Principle II ("New SSFM features MUST be implemented in both NumPy and PyTorch backends") applies to new physical models or algorithms, not to alternative implementations of existing algorithms. The existing `ssfm_order2_ndn_pytorch` satisfies the GPU requirement independently.

**Rationale**: If a PyTorch-Julia equivalent were required, the scope of this feature would double (PyTorch tensors cannot be passed to Julia via `juliacall`; a copy-based bridge would be needed). The primary motivation is performance benchmarking and code quality, not adding a new physical capability. This reasoning is documented as a Complexity Tracking entry in the plan.

---

## Decision 7: Module Placement and Naming

**Decision**:
- Julia source: `julia/FiberpropSSFM.jl` (Julia module `FiberpropSSFM`)
- Python wrapper: `fiberprop/ssfm_julia.py` (mirrors `ssfm_mcf.py` and `ssfm_mcf_pytorch.py`)
- Public API in Python wrapper: `ssfm_order2_ndn_julia`, `nonlinear_step_julia`, `linear_step_julia`

**Rationale**: Follows the established naming pattern in the codebase. `ssfm_mcf.py` → `ssfm_mcf_pytorch.py` → `ssfm_julia.py` is a natural progression. The `julia/` directory at repo root co-locates all Julia source and package files.

---

## Technical Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Julia initialization fails silently in some environments | Medium | High | Lazy init with clear error messages; graceful fallback |
| Row-major / column-major axis confusion causes incorrect FFT results | High | Critical | Explicit axis assertion in tests; compare against Python baseline at every shape |
| `juliapkg` auto-downloads Julia binary (large, slow first install) | Medium | Low | Document in README; optional: point to pre-installed Julia via `JULIA` env var |
| FFTW plan cache grows unbounded for varied grid sizes | Low | Low | Cache is keyed by `(n_cores, M)` — number of distinct shapes is bounded in practice |
| First-call JIT compilation introduces silent latency in CI | Medium | Medium | Run warmup call in test fixture setup |
