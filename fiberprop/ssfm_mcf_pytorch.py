import torch
import torch.fft as fft


def get_energy_rectangles_pytorch(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле левых прямоугольников) """
    return torch.sum(torch.abs(arr_func) ** 2) * time_step


def nonlinear_step_pytorch(psi, gamma, E_sat, g_0, current_energy, step):
    """ Нелинейный оператор (Керр и насыщение) """
    P_k = torch.abs(psi) ** 2
    E_k = current_energy

    # Обработка случаев без усиления
    no_gain_indices = (g_0 == 0)
    if no_gain_indices.any():
        psi[no_gain_indices] *= torch.exp(1j * gamma[no_gain_indices].unsqueeze(1) * P_k[no_gain_indices] * step)

    # Обработка случаев с усилением
    gain_indices = ((g_0 != 0) & (E_k != 0))
    if gain_indices.any():
        e_sat = E_sat[gain_indices].unsqueeze(1)
        g0 = g_0[gain_indices].unsqueeze(1)
        E_k = E_k[gain_indices].unsqueeze(1)
        P_k = P_k[gain_indices]
        gamma = gamma[gain_indices].unsqueeze(1)

        E = torch.sqrt((E_k ** 2 + 2 * E_k * e_sat) * torch.exp(2 * g0 * step) + e_sat ** 2) - e_sat
        C = -gamma * P_k * (E_k + e_sat - e_sat * torch.log(E_k + 2 * e_sat)) / (g0 * E_k) + torch.angle(
            psi[gain_indices])
        P = P_k * torch.exp(g0 * step) * torch.sqrt((E_k + 2 * e_sat) / E_k) * torch.sqrt(E / (E + 2 * e_sat))
        phi = gamma * P_k * (E + e_sat - e_sat * torch.log(E + 2 * e_sat)) / (g0 * E_k) + C
        psi[gain_indices] = torch.sqrt(P) * torch.exp(1j * phi)


def linear_step_pytorch(psi, Dmat):
    """ Линейный оператор (связи, дисперсия и потери) """
    n, m = psi.shape
    Dmat = Dmat.view(n, n, m)
    resV = torch.einsum('ijk,jk->ik', Dmat, psi)
    return resV


def ssfm_order2_pytorch(psi, current_energy, D, gamma, E_sat, g_0, h, tau, noise_amplitude=0.0):
    """ Реализация схемы расщепления """
    num = psi.shape[0]
    for i in range(num):
        if g_0[i] != 0:
            current_energy[i] = get_energy_rectangles_pytorch(psi[i], tau)
    nonlinear_step_pytorch(psi, gamma, E_sat, g_0, current_energy, h / 2)

    psi = fft.fft(psi, dim=-1)
    psi = linear_step_pytorch(psi, D)
    psi = fft.ifft(psi, dim=-1)

    for i in range(num):
        if g_0[i] != 0:
            current_energy[i] = get_energy_rectangles_pytorch(psi[i], tau)
    nonlinear_step_pytorch(psi, gamma, E_sat, g_0, current_energy, h / 2)

    if noise_amplitude != 0.0:
        current_noise = (torch.random.uniform(-noise_amplitude, noise_amplitude, psi.shape) +
                         1j*torch.random.uniform(-noise_amplitude, noise_amplitude, psi.shape))
        psi += current_noise
    return psi
