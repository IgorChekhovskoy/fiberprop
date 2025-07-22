import torch
import torch.fft as fft
from tqdm import trange


def get_energy_rectangles_pytorch(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле левых прямоугольников) """
    return torch.sum(torch.abs(arr_func) ** 2) * time_step


import torch
from torch import Tensor

# @torch.jit.script
def nonlinear_step_pytorch(psi: Tensor,
                       gamma_h: Tensor, g0_h: Tensor,
                       exp_g0h: Tensor, exp_2g0h: Tensor,
                       E_sat: Tensor, g0: Tensor,
                       energy_in: Tensor) -> None:
    P = psi.abs()**2
    no_gain = (g0 == 0)
    if no_gain.any():
        psi[no_gain] *= torch.exp(1j * gamma_h[no_gain, None] * P[no_gain])

    gain = (~no_gain)
    if gain.any():
        Ek   = energy_in[gain].unsqueeze(1)
        Pk   = P[gain]
        esat = E_sat[gain].unsqueeze(1)
        g0h_ = g0_h[gain].unsqueeze(1)
        eg1  = exp_g0h[gain].unsqueeze(1)
        eg2  = exp_2g0h[gain].unsqueeze(1)
        gamh = gamma_h[gain].unsqueeze(1)

        E  = torch.sqrt((Ek**2 + 2*Ek*esat) * eg2 + esat**2) - esat
        C  = -gamh*Pk*(Ek+esat-esat*torch.log(Ek+2*esat)) / (g0h_ * Ek) + torch.angle(psi[gain])
        Pn = Pk*eg1*torch.sqrt((Ek+2*esat)/Ek)*torch.sqrt(E/(E+2*esat))
        phi= gamh*Pk*(E+esat-esat*torch.log(E+2*esat)) / (g0h_ * Ek) + C
        psi[gain] = torch.sqrt(Pn) * torch.exp(1j*phi)



def linear_step_pytorch(psi: torch.Tensor, has_beta, D) -> torch.Tensor:
    """
    Линейный оператор для PyTorch.

    • D может быть (n², M) либо (n, n, M).  Чтобы не платить за
      view() на каждом шаге, передаём из solver уже трёхмерный
      тензор, но оставляем поддержку старого формата «на всякий».
    """
    if has_beta:
        psi = fft.fft(psi, dim=-1)
        psi = torch.einsum('ijk,jk->ik', D, psi)
        return fft.ifft(psi, dim=-1)
    else:
        return torch.matmul(D, psi)


def apply_absorbing_boundary_pytorch(psi: torch.Tensor, *, solver):
    taper = solver._taper_t
    if taper is None:
        return psi
    psi.mul_(taper)       # in-place, broadcasting
    return psi


# @torch.jit.script
def ssfm_order2_pytorch(
    psi: Tensor,
    current_E,                     # может прийти ndarray -> сконвертируем
    solver,
    h: float,
    tau: float,
    damp_length: float = 0.0,
    noise_amplitude: float = 0.0,
) -> Tensor:

    # ---------- гарантируем Tensor -----------------------------------
    if not torch.is_tensor(current_E):
        current_E = torch.as_tensor(
            current_E,
            dtype=solver.dtype,
            device=psi.device,
        )

    # ---------- константы (уже Tensor) --------------------------------
    if solver.gamma_h_half_t is None:
        solver.prepare_halfstep_constants()        # создаёт все _t

    g0    = solver.g0_t            # (n,) Tensor
    E_sat = solver.E_sat_t

    gain = g0 != 0
    if gain.any():
        current_E[gain] = (psi.abs()**2).sum(-1)[gain] * tau

    # -------- ½ NL ----------------------------------------------------
    nonlinear_step_pytorch(
        psi,
        solver.gamma_h_half_t, solver.g0_h_half_t,
        solver.exp_g0h_half_t, solver.exp_2g0h_half_t,
        E_sat, g0, current_E
    )

    # -------- absorber-1 ---------------------------------------------
    if damp_length:
        psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

    # -------- FFT – L – IFFT -----------------------------------------
    psi = linear_step_pytorch(psi, solver.has_beta, solver.D_pytorch)

    # -------- absorber-2 ---------------------------------------------
    if damp_length:
        psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

    if gain.any():
        current_E[gain] = (psi.abs()**2).sum(-1)[gain] * tau

    # -------- ½ NL (вторая) ------------------------------------------
    nonlinear_step_pytorch(
        psi,
        solver.gamma_h_half_t, solver.g0_h_half_t,
        solver.exp_g0h_half_t, solver.exp_2g0h_half_t,
        E_sat, g0, current_E
    )

    # -------- absorber-3 ---------------------------------------------
    if damp_length:
        psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

    # -------- шум -----------------------------------------------------
    if noise_amplitude:
        noise = noise_amplitude * (
            (torch.rand_like(psi.real) * 2 - 1)
            + 1j * (torch.rand_like(psi.real) * 2 - 1)
        )
        psi += noise.to(dtype=psi.dtype)

    # вернём psi и обновлённый NumPy-энергий,
    # чтобы caller заполнил self.energy[:, n+1]
    return psi


#################################################

def _nonlinear_step_windowed_pytorch(
        psi: torch.Tensor,
        gamma_h: torch.Tensor, g0_h: torch.Tensor,
        exp_g0h: torch.Tensor, exp_2g0h: torch.Tensor,
        E_sat: torch.Tensor, g0: torch.Tensor,
        tau: float, window: int
) -> None:
    """
    In-place Kerr + gain with rectangular windowing along time axis.
    Проходимся с шагом window, чтобы уменьшить перепады энергии
    и исключить OOM на очень длинных записей.
    """
    C, M = psi.shape
    for s in range(0, M, window):
        e = min(M, s + window)
        view = psi[:, s:e]          # (C, w) – общая ссылка, без .clone()
        # энергия в окне для каждого core
        E_slice = (view.abs() ** 2).sum(dim=1) * tau   # (C,)
        # применяем обычный нелинейный шаг
        nonlinear_step_pytorch(
            view, gamma_h, g0_h, exp_g0h, exp_2g0h,
            E_sat, g0, E_slice
        )

# ─────────────────────────────────────────────────────────────
def ssfm_order2_dnd_windowed_torch(
        solver,
        window_size: int,
        damp_length: float = 0.0,
) -> torch.Tensor:
    """
    Полностью GPU-ориентированный вариант схемы «D-N-D»
    с оконным нелинейным шагом. Работает и на CPU, если
    solver.device — 'cpu'.

    Возвращает final-tensor psi    (C, M).
    """
    # исходное поле
    psi = torch.as_tensor(
        solver.numerical_solution[0],
        dtype=solver.ctype,
        device=solver.device
    )

    # --- матрицы линейного шага (½-h, h, −½-h) -------------------
    D_half_t  = torch.as_tensor(solver.D_half,  dtype=solver.ctype,
                                device=solver.device)
    D_full_t  = torch.as_tensor(solver.D,       dtype=solver.ctype,
                                device=solver.device)
    invD_half = torch.as_tensor(solver.invD_half, dtype=solver.ctype,
                                device=solver.device)

    tau = solver.com.tau
    gamma_h  = solver.gamma_h_t
    g0_h     = solver.g0_h_t
    exp_g0h  = solver.exp_g0h_t
    exp_2g0h = solver.exp_2g0h_t
    g0       = solver.g0_t
    E_sat    = solver.E_sat_t

    # ------------- префикс: ½-D -----------------------------------
    psi = linear_step_pytorch(psi, solver.has_beta, D_half_t)
    if damp_length:
        psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

    # ------------- основной цикл по z -----------------------------
    for _ in trange(solver.com.N):

        _nonlinear_step_windowed_pytorch(
            psi, gamma_h, g0_h, exp_g0h, exp_2g0h,
            E_sat, g0, tau, window_size
        )

        psi = linear_step_pytorch(psi, solver.has_beta, D_full_t)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

    # ------------- суффикс: ½-D⁻¹ ----------------------------------
    psi = linear_step_pytorch(psi, solver.has_beta, invD_half)
    if damp_length:
        psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

    return psi
