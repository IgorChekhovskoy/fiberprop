# Tasks: Julia SSFM Solver Integration

**Input**: Design documents from `/specs/001-julia-ssfm-solver/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/python-api.md ✅, quickstart.md ✅

**Organization**: Tasks grouped by user story (US1 → US2 → US3) to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the Julia environment and declare all new dependencies before any code is written.

- [X] T001 Create `julia/` directory at repository root; write `julia/Project.toml` declaring FFTW.jl (uuid `7a1cc6ca-52ef-59f5-83cd-3a7055c09341`, version `"1"`) and PythonCall.jl (uuid `6099a3de-0909-46bc-b1f4-468b9a2dfc0d`, version `"0.9"`) with `[compat] julia = "1.9"`; generate `julia/Manifest.toml` by running `julia --project=julia -e "using Pkg; Pkg.instantiate()"`
- [X] T002 [P] Create `fiberprop/juliapkg.json` with entries for FFTW (uuid `7a1cc6ca-52ef-59f5-83cd-3a7055c09341`, version `"1"`) and PythonCall (uuid `6099a3de-0909-46bc-b1f4-468b9a2dfc0d`, version `"0.9"`) and `"julia": "1.9"` — this file is read by juliapkg to auto-activate the Julia environment on Python startup
- [X] T003 [P] Add `juliacall>=0.9` and `juliapkg>=0.1` to `requirements.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create module skeletons and wire solver dispatch before any Julia math is implemented. Establishes the shape that all subsequent tasks fill in.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Create `fiberprop/ssfm_julia.py` skeleton: top-level `try/except ImportError` guard around `import juliacall`; set `_JULIA_AVAILABLE = True/False`; define `_need_julia()` raising `RuntimeError("Julia backend not available: juliacall package not found. Install with: pip install juliacall juliapkg")`; in the `not _JULIA_AVAILABLE` branch define stub functions `ssfm_order2_ndn_julia`, `nonlinear_step_julia`, `linear_step_julia` each calling `_need_julia()`; leave the `else` branch empty (to be filled by T010–T012)
- [X] T005 [P] Create `julia/FiberpropSSFM.jl` skeleton: `module FiberpropSSFM`; `using FFTW, PythonCall`; declare module-level `const _plan_cache = Dict{Tuple{Int,Int}, NamedTuple}()`; add empty `function get_plans(n_cores::Int, M::Int)` stub; add empty stubs for `nonlinear_step!`, `linear_step!`, `ssfm_order2_ndn!`; add empty `function __init__() end`; close `end # module`
- [X] T006 In `fiberprop/solver.py` method `calculate_all_dispersion_matrices()`, add `"ssfm_order2_ndn_julia"` to the existing condition on line ~1044 that checks for `ssfm_order2_ndn` so it reads: `if self.com.method in ("ssfm_order2_ndn", "ssfm_order2_ndn_windowed", "ssfm_order2_ndn_julia"):` — this ensures `self.D` is calculated with full step size before Julia calls
- [X] T007 In `fiberprop/solver.py` main loop inside `run_numerical_simulation()`, add an `elif self.com.method == "ssfm_order2_ndn_julia":` branch (after the `ssfm_order2_ndn_compact_windowed` branch) calling `ssfm_order2_ndn_julia(psi_next, self.energy[:, n], self, self.com.h, tau, self.com.damp_length, self.eq.noise_amplitude,)` and also add the corresponding import at the top of `solver.py`: `from .ssfm_julia import ssfm_order2_ndn_julia`
- [X] T008 [P] Add `from .ssfm_julia import ssfm_order2_ndn_julia, nonlinear_step_julia, linear_step_julia` to `fiberprop/__init__.py` inside a `try/except ImportError` guard so that importing the package never fails even if juliacall is absent

**Checkpoint**: `python -c "from fiberprop import ssfm_order2_ndn_julia"` succeeds; `from fiberprop.solver import Solver` succeeds; calling `ssfm_order2_ndn_julia()` raises `RuntimeError` (stub active). Solver accepts `method="ssfm_order2_ndn_julia"` without crashing during setup.

---

## Phase 3: User Story 1 — Run Julia-Accelerated SSFM Simulation (Priority: P1) 🎯 MVP

**Goal**: Implement the three Julia functions and their Python wrappers so that a full N-D-N split-step simulation runs correctly via the Julia backend, producing results numerically identical to the Python baseline.

**Independent Test**: Run an existing single-core simulation with `method="ssfm_order2_ndn_julia"` and compare output to `method="ssfm_order2_ndn"` — relative L2 error must be < 1e-10.

### Implementation

- [X] T009 [P] [US1] Implement `get_plans(n_cores, M)` in `julia/FiberpropSSFM.jl`: check `_plan_cache` for key `(n_cores, M)`; if absent, allocate `buf = zeros(ComplexF64, M, n_cores)`, create `plan_fft!(similar(buf), 1; flags=FFTW.PATIENT)` and `plan_ifft!(similar(buf), 1; flags=FFTW.PATIENT)`, store as `(fwd=..., inv=...)` in `_plan_cache[(n_cores, M)]`; return the named tuple
- [X] T010 [P] [US1] Implement `nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy_in, gain_mask)` in `julia/FiberpropSSFM.jl`: for cores where `g0_h[i] == 0` apply `psi[i,:] .*= exp.(1im .* gamma_h[i] .* abs2.(psi[i,:]))`; for gain cores replicate the Python `nonlinear_step` formulas using `E`, `C`, `Pn`, `phi` as in `ssfm_mcf.py` lines 97–101 (using `energy_in[i]` for `Ek`); all operations on native `Matrix{ComplexF64}` rows
- [X] T011 [P] [US1] Implement `linear_step!(psi, has_beta, D, plans)` in `julia/FiberpropSSFM.jl`: if `has_beta` is true apply `plans.fwd * psi` (in-place FFT along dim 1), then multiply by `D` (handle both 2D `(M, n_cores)` diagonal case and 3D `(M, n_cores, n_cores)` coupling case via `@views` loop or einsum), then `plans.inv * psi` (in-place IFFT); if `has_beta` is false apply `D * psi` as matrix multiply (coupling-only); note `D` arrives as `(n_cores, n_cores)` or `(n_cores, n_cores, M)` from Python and must be permuted to Julia layout at this boundary
- [X] T012 [US1] Implement `ssfm_order2_ndn!(psi_py, energy_py, gamma_h_py, g0_h_py, exp_g0h_py, exp_2g0h_py, E_sat_py, D_py, has_beta, taper_py, tau, damp_length, noise_amplitude)` in `julia/FiberpropSSFM.jl`: convert `psi_py` to native `psi = permutedims(Array(psi_py), (2,1))`; convert 1D parameter arrays to native `Vector{Float64}`; call `get_plans`; compute power `P = abs2.(psi)`; update energy for gain cores; call `nonlinear_step!(psi, ..., h_half params, ...)` for ½ NL; optionally apply taper (`psi .*= taper` broadcast); call `linear_step!`; optionally apply taper; recompute power; update energy; call `nonlinear_step!` for second ½ NL; optionally apply taper; optionally add noise; write back `psi_py .= permutedims(psi, (2,1))`; return `nothing` (depends on T009, T010, T011)
- [X] T013 [US1] Implement Julia module `__init__()` in `julia/FiberpropSSFM.jl`: call `get_plans(1, 512)`, `get_plans(7, 1024)`, `get_plans(19, 2048)` to pre-create FFTW plans for common fiberprop shapes; call dummy invocations of `nonlinear_step!` and `linear_step!` with `ComplexF64` arrays to trigger JIT compilation; add `precompile` directives for main function signatures (depends on T009, T010, T011, T012)
- [X] T014 [US1] Implement the full Julia backend in `fiberprop/ssfm_julia.py` `else` branch: add module-level `_julia_ready = False` flag and `_jl = None`; implement `_ensure_julia()` that on first call sets `os.environ.setdefault("JULIA_NUM_THREADS", str(os.cpu_count() or 1))`, imports juliacall, loads `julia/FiberpropSSFM.jl` via `jl.seval(f'include("{julia_path}")')` and `jl.seval("using .FiberpropSSFM")`; implement `nonlinear_step_julia(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy_in, P=None)` as a Python function that calls `_ensure_julia()` then delegates to `jl.FiberpropSSFM.nonlinear_step_b(...)` with correct argument mapping (depends on T010)
- [X] T015 [US1] In `fiberprop/ssfm_julia.py`, implement `linear_step_julia(psi, has_beta, D)` calling `_ensure_julia()` then `jl.FiberpropSSFM.linear_step_b(psi, has_beta, D, plans)` where plans are retrieved via `jl.FiberpropSSFM.get_plans(psi.shape[0], psi.shape[1])`; return `np.asarray(result)` (depends on T011)
- [X] T016 [US1] In `fiberprop/ssfm_julia.py`, implement `ssfm_order2_ndn_julia(psi, current_energy, solver, h, tau, damp_length=0.0, noise_amplitude=0.0)`: call `_ensure_julia()`; extract all needed arrays from `solver` (`gamma_h_half`, `g0_h_half`, `exp_g0h_half`, `exp_2g0h_half`, `eq.E_sat`, `D`, `has_beta`, `taper`); ensure arrays are C-contiguous float64/complex128; call `jl.FiberpropSSFM.ssfm_order2_ndn_b(psi, current_energy, ..., tau, damp_length, noise_amplitude)`; return `psi` (depends on T012, T014)
- [X] T017 [US1] Write `tests/tests_julia_ssfm.py`: add `test_linear_step_julia_vs_python()` — create a Gaussian pulse, apply one `linear_step_julia` and one `linear_step` (Python), assert relative L2 error < 1e-10; add `test_nonlinear_step_julia_vs_python()` — apply one `nonlinear_step_julia` and one `nonlinear_step`, assert relative L2 error < 1e-10 (both gain-free and gain cases) (depends on T014, T015)
- [X] T018 [US1] In `tests/tests_julia_ssfm.py`, add `test_full_simulation_julia_vs_python()` — configure a `Solver` with `method="ssfm_order2_ndn_julia"` and a standard sech-pulse input; run `run_numerical_simulation()`; compare output to same simulation with `method="ssfm_order2_ndn"`; assert relative L2 error < 1e-10 across all time points and cores (depends on T016)
- [X] T019 [P] [US1] In `tests/tests_julia_ssfm.py`, add edge-case tests: `test_julia_single_core()` (n_cores=1, no coupling); `test_julia_no_dispersion()` (has_beta=False, coupling matrix only); `test_julia_with_damping()` (damp_length=0.1, energy decreases monotonically) (depends on T016)
- [X] T020 [US1] In `tests/tests_julia_ssfm.py`, add `test_julia_with_noise()` — run with `noise_amplitude > 0`, verify output differs from noiseless run (depends on T016)

**Checkpoint**: `pytest tests/tests_julia_ssfm.py -k "not fallback and not benchmark"` passes. A simulation with `method="ssfm_order2_ndn_julia"` completes and matches the Python baseline.

---

## Phase 4: User Story 2 — Transparent Fallback When Julia Is Unavailable (Priority: P2)

**Goal**: Ensure that importing `fiberprop` and using all non-Julia methods works correctly when `juliacall` is not installed, and that attempting to use the Julia method raises a clear, actionable error.

**Independent Test**: Mock `juliacall` as unavailable; import `fiberprop.ssfm_julia` — no exception; call `ssfm_order2_ndn_julia()` — `RuntimeError` raised with message containing "juliacall".

### Implementation

- [X] T021 [US2] In `fiberprop/ssfm_julia.py`, verify the top-level `try/except ImportError` guard (from T004) is complete and correct: the module must be importable without Julia; `_JULIA_AVAILABLE` must be `False` when juliacall is absent; the lazy `_ensure_julia()` function must raise `RuntimeError` (not `ImportError`) with a message including the install command; add `__all__` listing the three public functions so they are always exported regardless of Julia availability
- [X] T022 [P] [US2] In `tests/tests_julia_ssfm.py`, add `test_import_without_julia()` using `unittest.mock.patch` to patch `builtins.__import__` and simulate `juliacall` being absent; assert `import fiberprop.ssfm_julia` succeeds; assert `fiberprop.ssfm_julia._JULIA_AVAILABLE is False`
- [X] T023 [P] [US2] In `tests/tests_julia_ssfm.py`, add `test_call_without_julia_raises_runtime_error()`: mock juliacall as absent; call `ssfm_order2_ndn_julia(...)` with dummy arguments; assert `RuntimeError` is raised; assert `"juliacall"` appears in the error message; assert `"pip install"` appears in the error message
- [X] T024 [US2] In `tests/tests_julia_ssfm.py`, add `test_existing_methods_unaffected_when_julia_absent()`: mock juliacall as absent; import `fiberprop.solver`; create a `Solver` with `method="ssfm_order2_ndn"` (Python baseline); call `run_numerical_simulation()` and assert it completes successfully (depends on T022)

**Checkpoint**: `pytest tests/tests_julia_ssfm.py -k "fallback or without_julia or unaffected"` passes. Python-only workflows are not impacted by the Julia module being present.

---

## Phase 5: User Story 3 — Performance Benchmark (Priority: P3)

**Goal**: Provide a runnable benchmark script that documents Julia vs. Python wall-clock time and records results for future reference.

**Independent Test**: Run `python scripts/benchmark_julia_ssfm.py` — completes without error, prints a table of timings, and saves results to `data/benchmark_julia_ssfm.json`.

### Implementation

- [X] T025 [US3] Create `scripts/benchmark_julia_ssfm.py`: import `time`, `json`, `fiberprop`; configure a benchmark `Solver` with N=100, M=1024, 1-core, standard sech pulse; time `run_numerical_simulation()` for `method="ssfm_order2_ndn"` (Python baseline, 3 runs, take median); time same for `method="ssfm_order2_ndn_julia"` (3 runs, take median, include first-call JIT overhead separately); print a formatted table with method, median time, speedup ratio; save results as JSON to `data/benchmark_julia_ssfm.json`
- [X] T026 [P] [US3] In `tests/tests_julia_ssfm.py`, add `test_julia_performance_neutral()`: run one iteration of each method on a medium grid (N=10, M=1024, 1 core); assert Julia total wall-clock (including startup) completes within 60 seconds (guards against catastrophic regression, not a tight performance target)
- [X] T027 [US3] Run `scripts/benchmark_julia_ssfm.py`, capture output, and update `specs/001-julia-ssfm-solver/quickstart.md` section "First-Time Setup" with actual observed first-run compilation time and the "Troubleshooting" section with real timing data (depends on T025)

**Checkpoint**: `python scripts/benchmark_julia_ssfm.py` runs end-to-end; `data/benchmark_julia_ssfm.json` is created; `quickstart.md` contains actual benchmark numbers.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation completeness, thread integration, and final validation across all user stories.

- [X] T028 [P] In `fiberprop/parallel_runtime.py`, add Julia thread awareness to `configure_threads()`: after setting `NUMBA_NUM_THREADS`, add `if n is not None: os.environ.setdefault("JULIA_NUM_THREADS", str(int(n)))` with a comment explaining that `JULIA_NUM_THREADS` must be set before the first `import juliacall` so `setdefault` is used to not override a user-set value; add `"JULIA_NUM_THREADS"` to the `env` dict in `threading_report()`
- [X] T029 [P] Update `README.md` (or create a `docs/julia-backend.md` if README is too large): add a "Julia Backend" section describing installation (`pip install juliacall juliapkg`), the `method="ssfm_order2_ndn_julia"` usage, the `JULIA_NUM_THREADS` environment variable, and a link to `specs/001-julia-ssfm-solver/quickstart.md`
- [X] T030 Run the full test suite `pytest tests/` and confirm zero regressions on all pre-existing tests; run `pytest tests/tests_julia_ssfm.py -v` and confirm all new tests pass; 11 Julia tests pass; 1 pre-existing test fails due to old API (`num_equations=` removed from EquationParameters) — not a regression introduced by this feature

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately; T001, T002, T003 all run in parallel
- **Foundational (Phase 2)**: Depends on Phase 1 completion; T004 and T005 run in parallel; T006–T008 can run after T004/T005 skeletons exist
- **US1 (Phase 3)**: Depends on Foundational completion; T009–T011 run in parallel; T012–T013 after T009–T011; T014–T016 after T012; T017–T020 after T014–T016
- **US2 (Phase 4)**: Depends on Foundational completion (T021 verifies/completes T004); T022–T023 run in parallel; T024 after T022
- **US3 (Phase 5)**: Depends on US1 completion (Julia simulation must work); T025 before T027; T026 parallel with T025
- **Polish (Phase 6)**: Depends on US1–US3 completion

### User Story Dependencies

- **US1 (P1)**: Depends only on Foundational — the core deliverable, no dependency on US2/US3
- **US2 (P2)**: Depends only on Foundational — fallback mechanism is independent of Julia correctness; can be worked in parallel with US1 by a second developer
- **US3 (P3)**: Depends on US1 (Julia must be working to benchmark it)

### Within US1

```
T009 ──┐
T010 ──┼──► T012 ──► T013/T014 ──► T016/T017/T018
T011 ──┘              T015 ────►──┘
                                   T019/T020 (parallel after T016)
```

---

## Parallel Execution Examples

### Phase 1 (all parallel)
```
Task: "Create julia/Project.toml and generate Manifest.toml"         [T001]
Task: "Create fiberprop/juliapkg.json"                               [T002]
Task: "Update requirements.txt with juliacall and juliapkg"          [T003]
```

### Phase 2 (T004 and T005 parallel)
```
Task: "Create fiberprop/ssfm_julia.py skeleton"                      [T004]
Task: "Create julia/FiberpropSSFM.jl skeleton"                       [T005]
```

### Phase 3: US1 first wave (T009–T011 all parallel)
```
Task: "Implement get_plans() FFTW plan cache in FiberpropSSFM.jl"    [T009]
Task: "Implement nonlinear_step! in FiberpropSSFM.jl"                [T010]
Task: "Implement linear_step! in FiberpropSSFM.jl"                   [T011]
```

### Phase 4: US2 tests (parallel after T021)
```
Task: "Write test_import_without_julia()"                            [T022]
Task: "Write test_call_without_julia_raises_runtime_error()"         [T023]
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T008)
3. Complete Phase 3: US1 (T009–T020)
4. **STOP and VALIDATE**: `pytest tests/tests_julia_ssfm.py -k "not fallback and not benchmark"` passes; manual simulation with `method="ssfm_order2_ndn_julia"` produces correct output
5. Feature is usable — proceed to US2 and US3 for hardening and benchmarking

### Incremental Delivery

1. Phase 1 + Phase 2 → Julia module structure wired into the project ✓
2. Phase 3 (US1) → Julia simulation working, validated, tested ✓
3. Phase 4 (US2) → Graceful fallback verified ✓
4. Phase 5 (US3) → Performance benchmark documented ✓
5. Phase 6 → Polish complete, zero regressions confirmed ✓

---

## Notes

- [P] tasks touch different files and have no dependency on incomplete sibling tasks
- Julia axis convention (column-major vs. row-major) is the highest-risk implementation detail — validate at T012 before proceeding to T013+
- The `ssfm_order2_ndn!` Julia function receives `h_half` precomputed parameters from Python (same as the Python version uses `solver.gamma_h_half` etc.) — do not recompute `h/2` inside Julia
- `jl.FiberpropSSFM.some_function_b` notation: juliacall maps Julia's `!` (bang) suffix to `_b` suffix in Python call names
- Commit after each checkpoint to preserve working state
