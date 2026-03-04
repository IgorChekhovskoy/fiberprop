# Feature Specification: Julia SSFM Solver Integration

**Feature Branch**: `001-julia-ssfm-solver`
**Created**: 2026-03-04
**Status**: Draft
**Input**: User description: "Реализуй модуль на языке julia, который будет содержать функции для расчёта отдельных шагов метода по типу ssfm_order2_ndn. Оберни его в модуль python, чтобы эту функцию можно было вызывать в основном проекте из solver.py, как остальные реализации численного метода. Используй juliacall и pythoncall для налаживания взаимодействия языков"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Julia-Accelerated SSFM Simulation (Priority: P1)

A researcher running fiber optics propagation simulations wants to use a high-performance Julia-based implementation of the `ssfm_order2_ndn` method. They configure their simulation with `method="ssfm_order2_ndn_julia"` in `ComputationalParameters`, then call `solver.run_numerical_simulation()` as usual. The simulation completes with numerically identical results to the Python/Numba baseline.

**Why this priority**: This is the core deliverable. Without it, the feature provides no value.

**Independent Test**: Can be tested by running an existing simulation configuration with `method="ssfm_order2_ndn_julia"` and comparing the output field against `method="ssfm_order2_ndn"`.

**Acceptance Scenarios**:

1. **Given** a valid `Solver` instance with `com.method = "ssfm_order2_ndn_julia"`, **When** `run_numerical_simulation()` is called, **Then** the simulation completes without errors and stores results in `solver.numerical_solution`.
2. **Given** the same initial field and parameters used with `ssfm_order2_ndn` (Python), **When** the same scenario is run with `ssfm_order2_ndn_julia`, **Then** the resulting optical field differs by less than 1e-10 (relative L2 norm) from the Python baseline.
3. **Given** parameters that include nonlinear gain (`g_0 != 0`) and dispersion (`beta2 != 0`), **When** the Julia method is used, **Then** both the nonlinear and linear sub-steps produce correct results matching the reference implementation.

---

### User Story 2 - Transparent Fallback When Julia Is Unavailable (Priority: P2)

A researcher runs the project on a machine without Julia installed. The system gracefully informs the user that the Julia backend is unavailable and does not crash the existing Python-based simulation workflows.

**Why this priority**: Protects the usability of existing functionality when Julia is not available.

**Independent Test**: Can be tested by attempting to import the Julia wrapper module in a Python environment without Julia, and verifying that a clear, informative error is raised only when the Julia method is actually invoked — not at import time.

**Acceptance Scenarios**:

1. **Given** Julia is not installed, **When** `solver.py` is imported, **Then** the import succeeds and all non-Julia methods remain usable.
2. **Given** Julia is not installed, **When** `method="ssfm_order2_ndn_julia"` is specified and simulation is launched, **Then** a descriptive error is raised identifying the missing Julia dependency.

---

### User Story 3 - Performance Benefit for Large-Scale Simulations (Priority: P3)

A researcher running large-scale simulations (many cores, long time windows) expects the Julia implementation to complete the same simulation faster than the Python/Numba baseline, justifying the additional setup complexity.

**Why this priority**: The primary motivation for a Julia backend is computational performance. This scenario validates that the integration delivers measurable benefit at scale.

**Independent Test**: Can be tested by timing both methods on a standardized benchmark configuration and comparing elapsed wall-clock time.

**Acceptance Scenarios**:

1. **Given** a benchmark scenario with 1000 propagation steps and 4096 time grid points, **When** `ssfm_order2_ndn_julia` is timed against `ssfm_order2_ndn`, **Then** the Julia variant completes in no more than the Python variant's time (neutral or faster).
2. **Given** a multi-core system, **When** the Julia implementation runs on a large grid, **Then** the wall-clock time is documented and provides a reference for future optimization.

---

### Edge Cases

- What happens when the Julia runtime takes significant time to initialize (JIT compilation on first call)? The first-call overhead should not cause simulation timeouts or user confusion.
- How does the system handle complex-valued arrays (`complex128`) passed between Python and Julia — specifically preserving full double precision?
- What happens when the input field `psi` contains NaN or Inf values — does the Julia module propagate the failure clearly back to the Python caller?
- How does the system behave when `damp_length > 0` (absorbing boundary conditions) is used in the Julia code path?
- How does the system handle `noise_amplitude > 0` — is additive random noise generated and applied consistently?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a Julia module containing functions that implement the individual computational sub-steps of the `ssfm_order2_ndn` (N-D-N) scheme: the nonlinear step (Kerr nonlinearity and saturable gain), the linear dispersion step (FFT-based), and their composition into a full N-D-N split-step iteration. The `ssfm_order2_dnd` variant and all other SSFM variants are out of scope.
- **FR-002**: The system MUST provide a Python wrapper module that exposes: (a) the composed `ssfm_order2_ndn_julia` function with the same calling interface as the existing Python implementation (`psi, current_energy, solver, h, tau, damp_length, noise_amplitude`), and (b) the individual sub-step functions (`nonlinear_step_julia`, `linear_step_julia`) as independently callable Python functions to enable unit testing.
- **FR-003**: The solver dispatch logic in `solver.py` MUST recognize `method="ssfm_order2_ndn_julia"` and route each propagation step to the Julia-backed implementation, consistent with how other methods are dispatched.
- **FR-004**: Data arrays (complex128, shape `(n_cores, M)`) MUST be passed between Python and Julia without loss of numerical precision, and without requiring manual data copies in user code.
- **FR-005**: The Julia implementation MUST produce results numerically equivalent (relative L2 error < 1e-10) to the reference Python implementation for all valid parameter combinations tested.
- **FR-006**: The Julia module MUST correctly handle both the gain-free case (`g_0 = 0`) and the saturated gain case (`g_0 != 0`) in the nonlinear sub-step.
- **FR-007**: The Python wrapper MUST handle Julia runtime initialization (including JIT warm-up) transparently, with no additional user action required beyond specifying `method="ssfm_order2_ndn_julia"`.
- **FR-008**: If Julia or the required bridge packages are not installed, importing the project MUST NOT break existing Python functionality; the Julia method MUST raise an informative error only when actually invoked.
- **FR-009**: The Julia-backed method MUST support the optional absorbing boundary condition (`damp_length > 0`), consistent with the Python version's behavior.
- **FR-010**: The Julia-backed method MUST support the optional additive noise step (`noise_amplitude > 0`), consistent with the Python version's behavior.

### Key Entities

- **Julia SSFM Module**: A self-contained Julia module containing functions for the nonlinear sub-step, the linear dispersion sub-step, and the full N-D-N split-step composition. Receives and returns arrays exchangeable with Python.
- **Python Julia Wrapper**: A Python module that initializes the Julia runtime, loads the Julia SSFM module, and exposes a `ssfm_order2_ndn_julia` function callable from `solver.py`.
- **Solver Method Dispatch**: The routing logic inside `Solver.run_numerical_simulation()` that selects the correct per-step function based on `com.method`.
- **ComputationalParameters**: The existing dataclass whose `method` string field is extended to accept `"ssfm_order2_ndn_julia"` as a valid value.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The Julia-backed method produces output fields with a relative L2 error below 1e-10 compared to the Python/Numba baseline across a suite of at least 5 test scenarios covering varied `beta2`, `gamma`, `g_0`, and grid sizes.
- **SC-002**: All pre-existing tests in the project pass without modification after the feature is integrated, confirming zero regression on Python-based methods.
- **SC-003**: A researcher can switch from the Python method to the Julia method by changing only the `method` field in `ComputationalParameters` — no other user code changes are required.
- **SC-004**: On a standard benchmark (1000 steps, 4096 time points, 7 fiber cores), the Julia method completes successfully and its wall-clock time is recorded for reference.
- **SC-005**: Attempting to use `method="ssfm_order2_ndn_julia"` without Julia installed produces a clear, user-readable error message within 5 seconds of invoking the simulation.

## Clarifications

### Session 2026-03-04

- Q: Should the Julia module implement only `ssfm_order2_ndn`, or also `ssfm_order2_dnd`? → A: Only `ssfm_order2_ndn` (N-D-N ordering). `ssfm_order2_dnd` and other variants are explicitly out of scope for this feature.
- Q: Should individual sub-steps (`nonlinear_step`, `linear_step`) be exposed as independently callable Python functions, or only the composed function? → A: Both the composed function and individual sub-steps are exposed, enabling independent unit testing of each sub-step.
- Q: How should Julia package dependencies be declared and managed? → A: A dedicated `Project.toml` + `Manifest.toml` stored in the repository (under a `julia/` subdirectory), pinning exact package versions for reproducibility.

## Assumptions

- The existing `Solver` class interface (attributes such as `gamma_h_half`, `g0_h_half`, `exp_g0h_half`, `exp_2g0h_half`, `eq.E_sat`, `D`, `has_beta`, `taper`) remains stable and is accessible to the Python wrapper.
- Julia and the required interoperability bridge packages will be installed as optional dependencies, documented in project setup files, but are not required for core Python functionality. Julia package dependencies are declared via a `Project.toml` + `Manifest.toml` stored in a `julia/` subdirectory of the repository, ensuring reproducible and version-pinned Julia environments across machines.
- Complex arrays use `complex128` (double-precision complex) in Python and the equivalent `ComplexF64` in Julia.
- The Julia module is loaded once per process at first use and reused for all subsequent calls within the same session.
- Thread count for Julia parallel execution is configured via the existing environment variable mechanism (e.g., `JULIA_NUM_THREADS`), consistent with how the project already manages parallel threads for Python.
