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
    def ssfm_order2_ndn_pytorch(*args, **kwargs): _need_torch()
    def ssfm_order2_dnd_pytorch(*args, **kwargs): _need_torch()
    def ssfm_order2_dnd_short_pytorch(*args, **kwargs): _need_torch()
    def nonlinear_step_windowed_pytorch(*args, **kwargs): _need_torch()
    def ssfm_order2_dnd_windowed_short_pytorch(*args, **kwargs): _need_torch()

else:
    # ----------------- Полноценная реализация (как у вас), без лишних правок ----

    def get_energy_rectangles_pytorch(arr_func: Tensor, time_step: float) -> Tensor:
        """ Возвращает интеграл энергии (левые прямоугольники) """
        return torch.sum(torch.abs(arr_func) ** 2) * time_step

    # @torch.jit.script
    def power_abs2_t(psi: Tensor) -> Tensor:
        """|psi|^2 поэлементно, без sqrt/abs. Возвращает Tensor той же формы."""
        return psi.real * psi.real + psi.imag * psi.imag


    def energy_from_power_t(P: Tensor, tau: Tensor | float) -> Tensor:
        """Интеграл энергии по времени для каждого канала: sum_t P * tau  → (n,)"""
        return P.sum(dim=1) * (tau if torch.is_tensor(tau) else torch.tensor(tau, dtype=P.dtype, device=P.device))


    def nonlinear_step_pytorch(
            psi: Tensor,
            gamma_h: Tensor, g0_h: Tensor,
            exp_g0h: Tensor, exp_2g0h: Tensor,
            E_sat: Tensor,
            energy_in: Tensor,
            *,
            P: Tensor | None = None
    ) -> None:

        if P is None:
            P = power_abs2_t(psi)

        no_gain = (g0_h == 0)
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

            E = torch.sqrt((Ek ** 2 + 2 * Ek * esat) * eg2 + esat ** 2) - esat
            C = -gamh * Pk * (Ek + esat - esat * torch.log(Ek + 2 * esat)) / (g0h_ * Ek) + torch.angle(psi[gain])
            Pn = Pk * eg1 * torch.sqrt((Ek + 2 * esat) / Ek) * torch.sqrt(E / (E + 2 * esat))
            phi = gamh * Pk * (E + esat - esat * torch.log(E + 2 * esat)) / (g0h_ * Ek) + C
            psi[gain] = torch.sqrt(Pn) * torch.exp(1j * phi)

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
        taper = solver.taper
        if taper is None:
            return psi
        psi.mul_(taper)  # in-place, broadcasting
        return psi

    # @torch.jit.script
    def ssfm_order2_ndn_pytorch(
            psi: Tensor,
            current_E,
            solver,
            h: float,
            tau: float,
            damp_length: float = 0.0,
            noise_amplitude: float = 0.0,
    ) -> Tensor:
        """
        Полный шаг D–N–D (PyTorch).
        Требуется: solver.D уже рассчитана для шага h; константы half-step подготовлены через solver.prepare_halfstep_constants().
        """
        if not torch.is_tensor(current_E):
            current_E = torch.as_tensor(current_E, dtype=solver.dtype, device=psi.device)

        P = power_abs2_t(psi)

        # энергия/P ДО первой половины NL
        if (solver.eq.g_0 != 0).any():
            current_E[solver.eq.g_0 != 0] = energy_from_power_t(P, solver.tau_t)[solver.eq.g_0 != 0]

        nonlinear_step_pytorch(
            psi,
            solver.gamma_h_half, solver.g0_h_half,
            solver.exp_g0h_half, solver.exp_2g0h_half,
            solver.eq.E_sat, solver.eq.g_0, current_E,
            P=P
        )

        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        psi = linear_step_pytorch(psi, solver.has_beta, solver.D)

        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        P = power_abs2_t(psi)

        if (solver.eq.g_0 != 0).any():
            current_E[solver.eq.g_0 != 0] = energy_from_power_t(P, solver.tau_t)[solver.eq.g_0 != 0]

        nonlinear_step_pytorch(
            psi,
            solver.gamma_h_half, solver.g0_h_half,
            solver.exp_g0h_half, solver.exp_2g0h_half,
            solver.eq.E_sat, current_E,
            P=P
        )

        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        if noise_amplitude:
            noise = noise_amplitude * (
                    (torch.rand_like(psi.real) * 2 - 1)
                    + 1j * (torch.rand_like(psi.real) * 2 - 1)
            )
            psi = psi + noise.to(dtype=psi.dtype)

        return psi


    def ssfm_order2_dnd_pytorch(
            psi: Tensor,
            current_E,
            solver,
            h: float,
            tau: float,
            damp_length: float = 0.0,
            noise_amplitude: float = 0.0,
    ) -> Tensor:
        if not torch.is_tensor(current_E):
            current_E = torch.as_tensor(current_E, dtype=solver.dtype, device=psi.device)

        psi = linear_step_pytorch(psi, solver.has_beta, solver.D_half)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        P = power_abs2_t(psi)

        g0 = solver.eq.g_0
        gain = (g0 != 0)
        if gain.any():
            current_E[gain] = energy_from_power_t(P, solver.tau_t)[gain]

        nonlinear_step_pytorch(
            psi,
            solver.gamma_h, solver.g0_h,
            solver.exp_g0h, solver.exp_2g0h,
            solver.eq.E_sat, current_E,
            P=P
        )

        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        psi = linear_step_pytorch(psi, solver.has_beta, solver.D_half)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        if noise_amplitude:
            noise = noise_amplitude * (
                    (torch.rand_like(psi.real) * 2 - 1)
                    + 1j * (torch.rand_like(psi.real) * 2 - 1)
            )
            psi = psi + noise.to(dtype=psi.dtype)

        return psi


    def ssfm_order2_dnd_short_pytorch(
            solver,
            damp_length: float = 0.0,
            disable_progress_bar: bool = False,
    ):
        psi = torch.as_tensor(solver.numerical_solution[0], dtype=solver.ctype, device=solver.device)

        if solver.gamma_h is None:
            solver.prepare_halfstep_constants()

        psi = linear_step_pytorch(psi, solver.has_beta, solver.D_half)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        energy_full = torch.zeros(psi.shape[0], dtype=solver.dtype, device=solver.device)

        for _ in trange(solver.com.N, disable=disable_progress_bar):
            P = power_abs2_t(psi)
            if (solver.eq.g_0 != 0).any():
                energy_full[solver.eq.g_0 != 0] = energy_from_power_t(P, solver.tau_t)[solver.eq.g_0 != 0]

            nonlinear_step_pytorch(
                psi,
                solver.gamma_h, solver.g0_h,
                solver.exp_g0h, solver.exp_2g0h,
                solver.eq.E_sat, energy_full,
                P=P
            )

            psi = linear_step_pytorch(psi, solver.has_beta, solver.D)
            if damp_length:
                psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        psi = linear_step_pytorch(psi, solver.has_beta, solver.invD_half)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        return psi


    def nonlinear_step_windowed_pytorch(
            psi: Tensor,
            gamma_h: Tensor, g0_h: Tensor,
            exp_g0h: Tensor, exp_2g0h: Tensor,
            E_sat: Tensor,
            tau: float, window: int,
            *, offset_left: int = 0
    ) -> None:
        """Оконный Kerr+gain; константы берём из solver.exp1 / solver.g0zeros; энергия и P считаются один раз."""
        C, M = psi.shape
        offset_right = M - offset_left

        for s in range(0, M, window):
            e = min(M, s + window)
            view = psi[:, s:e]

            l_off = max(0, offset_left - s)
            r_off = max(0, e - offset_right)
            core = view.shape[1] - l_off - r_off

            if l_off:
                sub = view[:, :l_off]
                Psub = power_abs2_t(sub)
                E_slice = energy_from_power_t(Psub, tau)
                nonlinear_step_pytorch(sub, gamma_h, g0_h * 0, exp_g0h, exp_2g0h, E_sat, E_slice, P=Psub)

            if core > 0:
                sub = view[:, l_off:l_off + core]
                Psub = power_abs2_t(sub)
                E_slice = energy_from_power_t(Psub, tau)
                nonlinear_step_pytorch(sub, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, E_slice, P=Psub)

            if r_off:
                sub = view[:, -r_off:]
                Psub = power_abs2_t(sub)
                E_slice = energy_from_power_t(Psub, tau)
                nonlinear_step_pytorch(sub, gamma_h, g0_h * 0, exp_g0h, exp_2g0h, E_sat, E_slice, P=Psub)


    def ssfm_order2_dnd_windowed_short_pytorch(
            solver,
            window_size: int,
            damp_length: float = 0.0,
            disable_progress_bar: bool = False,
    ):
        psi = torch.as_tensor(solver.numerical_solution[0], dtype=solver.ctype, device=solver.device)

        if solver.gamma_h is None:
            solver.prepare_halfstep_constants()

        psi = linear_step_pytorch(psi, solver.has_beta, solver.D_half)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        for _ in trange(solver.com.N, disable=disable_progress_bar):
            nonlinear_step_windowed_pytorch(
                psi,
                solver.gamma_h, solver.g0_h,
                solver.exp_g0h, solver.exp_2g0h,
                solver.eq.E_sat,
                solver.tau_t, window_size,
                offset_left=solver.com.offset_size
            )
            psi = linear_step_pytorch(psi, solver.has_beta, solver.D)
            if damp_length:
                psi = apply_absorbing_boundary_pytorch(psi, solver=solver)

        psi = linear_step_pytorch(psi, solver.has_beta, solver.invD_half)
        if damp_length:
            psi = apply_absorbing_boundary_pytorch(psi, solver=solver)
        return psi
