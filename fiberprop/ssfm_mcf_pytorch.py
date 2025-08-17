from typing import Any

try:
    import torch
    import torch.fft as fft
    from torch import Tensor
    from tqdm import trange
    _TORCH_AVAILABLE = True
except Exception:
    torch = None           # type: ignore
    fft = None             # type: ignore
    Tensor = Any           # type: ignore
    def trange(*args, **kwargs):     # минимальная заглушка tqdm.trange
        return range(kwargs.get("total", args[0] if args else 0))
    _TORCH_AVAILABLE = False


def _need_torch():
    raise RuntimeError(
        "Выбран PyTorch-бэкенд, но torch не установлен. "
        "Установите PyTorch (pip install torch) или используйте NumPy-бэкенд."
    )


if not _TORCH_AVAILABLE:
    # ----------------- Заглушки: вызываются только при попытке использования ----
    def get_energy_rectangles_pytorch(*args, **kwargs): _need_torch()
    def nonlinear_step_pytorch(*args, **kwargs): _need_torch()
    def linear_step_pytorch(*args, **kwargs): _need_torch()
    def apply_absorbing_boundary_pytorch(*args, **kwargs): _need_torch()
    def ssfm_order2_pytorch(*args, **kwargs): _need_torch()
    def _nonlinear_step_windowed_pytorch(*args, **kwargs): _need_torch()
    def ssfm_order2_dnd_windowed_short_torch(*args, **kwargs): _need_torch()

else:
    # ----------------- Полноценная реализация (как у вас), без лишних правок ----

    def get_energy_rectangles_pytorch(arr_func: Tensor, time_step: float) -> Tensor:
        """ Возвращает интеграл энергии (левые прямоугольники) """
        return torch.sum(torch.abs(arr_func) ** 2) * time_step

    # @torch.jit.script
    def nonlinear_step_pytorch(
        psi: Tensor,
        gamma_h: Tensor, g0_h: Tensor,
        exp_g0h: Tensor, exp_2g0h: Tensor,
        E_sat: Tensor, g0: Tensor,
        energy_in: Tensor
    ) -> None:
        P = psi.abs() ** 2
        no_gain = (g0 == 0)
        if no_gain.any():
            psi[no_gain] *= torch.exp(1j * gamma_h[no_gain, None] * P[no_gain])

        gain = (~no_gain)
        if gain.any():
            Ek = energy_in[gain].unsqueeze(1)
            Pk = P[gain]
            esat = E_sat[gain].unsqueeze(1)
            g0h_ = g0_h[gain].unsqueeze(1)
            eg1 = exp_g0h[gain].unsqueeze(1)
            eg2 = exp_2g0h[gain].unsqueeze(1)
            gamh = gamma_h[gain].unsqueeze(1)

            E = torch.sqrt((Ek**2 + 2*Ek*esat) * eg2 + esat**2) - esat
            C = -gamh*Pk*(Ek+esat-esat*torch.log(Ek+2*esat)) / (g0h_ * Ek) + torch.angle(psi[gain])
            Pn = Pk*eg1*torch.sqrt((Ek+2*esat)/Ek)*torch.sqrt(E/(E+2*esat))
            phi = gamh*Pk*(E+esat-esat*torch.log(E+2*esat)) / (g0h_ * Ek) + C
            psi[gain] = torch.sqrt(Pn) * torch.exp(1j*phi)

    def linear_step_pytorch(psi: Tensor, has_beta: bool, D: Tensor) -> Tensor:
        """
        Линейный оператор для PyTorch.

        • D может быть (n², M) либо (n, n, M). Чтобы не платить за view() на каждом шаге,
          лучше передавать сразу трёхмерный тензор, но оставляем поддержку старого формата.
        """
        if has_beta:
            psi = fft.fft(psi, dim=-1)
            psi = torch.einsum('ijk,jk->ik', D, psi)
            return fft.ifft(psi, dim=-1)
        else:
            return torch.matmul(D, psi)

    def apply_absorbing_boundary_pytorch(psi: Tensor, *, solver):
        taper = solver._taper_t
        if taper is None:
            return psi
        psi.mul_(taper)  # in-place, broadcasting
        return psi

    # @torch.jit.script
    def ssfm_order2_pytorch(
        psi: Tensor,
        current_E,
        solver,
        h: float,
        tau: float,
        damp_length: float = 0.0,
        noise_amplitude: float = 0.0,
    ) -> Tensor:
        # ---------- гарантируем Tensor -----------------------------------
        if not torch.is_tensor(current_E):
            current_E = torch.as_tensor(current_E, dtype=solver.dtype, device=psi.device)

        # ---------- константы (уже Tensor) --------------------------------
        if solver.gamma_h_half_t is None:
            solver.prepare_halfstep_constants()  # создаёт все _t

        g0 = solver.g0_t            # (n,) Tensor
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

        return psi

    def _nonlinear_step_windowed_pytorch(
        psi: Tensor,
        gamma_h: Tensor, g0_h: Tensor,
        exp_g0h: Tensor, exp_2g0h: Tensor,
        E_sat: Tensor, g0: Tensor,
        tau: float, window: int
    ) -> None:
        """
        In-place Kerr + gain with rectangular windowing along time axis.
        """
        C, M = psi.shape
        for s in range(0, M, window):
            e = min(M, s + window)
            view = psi[:, s:e]  # (C, w) – общая ссылка, без .clone()
            E_slice = (view.abs() ** 2).sum(dim=1) * tau  # (C,)
            nonlinear_step_pytorch(view, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, g0, E_slice)

    def ssfm_order2_dnd_windowed_short_torch(
        solver,
        window_size: int,
        damp_length: float = 0.0,
        disable_progress_bar=False,
    ):
        """
        Полностью torch-вариант схемы «D-N-D» с оконным нелинейным шагом.
        Возвращает финальный тензор psi (C, M).
        """
        psi = torch.as_tensor(solver.numerical_solution[0], dtype=solver.ctype, device=solver.device)

        # матрицы линейного шага (½-h, h, −½-h)
        D_half_t  = torch.as_tensor(solver.D_half,    dtype=solver.ctype, device=solver.device)
        D_full_t  = torch.as_tensor(solver.D,         dtype=solver.ctype, device=solver.device)
        invD_half = torch.as_tensor(solver.invD_half, dtype=solver.ctype, device=solver.device)

        tau = solver.com.tau
        gamma_h  = solver.gamma_h_t
        g0_h     = solver.g0_h_t
        exp_g0h  = solver.exp_g0h_t
        exp_2g0h = solver.exp_2g0h_t
        g0       = solver.g0_t
        E_sat    = solver.E_sat_t

        # префикс: ½-D
        psi = linear_step_pytorch(psi, solver.has_beta, D_half_t)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        for _ in trange(solver.com.N, disable=disable_progress_bar):
            _nonlinear_step_windowed_pytorch(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, g0, tau, window_size)
            psi = linear_step_pytorch(psi, solver.has_beta, D_full_t)
            if damp_length:
                psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        # суффикс: ½-D⁻¹
        psi = linear_step_pytorch(psi, solver.has_beta, invD_half)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        return psi