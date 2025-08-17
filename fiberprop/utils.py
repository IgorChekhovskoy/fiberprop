import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

def fft_derivative(arr, dt, axis=-1):
    """
    Производная по времени через БПФ:
        d/dt f(t)  ⟷  i·ω·F(ω)

    Parameters
    ----------
    arr : ndarray (..., M)
        Комплексный сигнал.
    dt : float
        Шаг по времени.
    axis : int
        Ось времени (по умолчанию последняя).

    Returns
    -------
    ndarray (комплекс)
    """
    n = arr.shape[axis]
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)  # [rad/ps]
    F = np.fft.fft(arr, axis=axis)
    dFdt = 1j * omega * F
    return np.fft.ifft(dFdt, axis=axis)


def fft_derivative_torch(arr, dt, axis=-1):
    """
    Torch-аналог fft_derivative. Требует установленный PyTorch.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "fft_derivative_torch требует PyTorch, но он не установлен. "
            "Установите torch или используйте fft_derivative (NumPy)."
        )
    n = arr.shape[axis]
    dtype = arr.dtype
    device = arr.device
    omega = 2.0 * torch.pi * torch.fft.fftfreq(
        n, d=dt, device=device, dtype=dtype.real_dtype
    )
    # reshape omega для корректного broadcasting вдоль нужной оси
    shape = [1] * arr.ndim
    shape[axis] = n
    omega = omega.reshape(shape)

    F = torch.fft.fft(arr, dim=axis)
    dFdt = 1j * omega * F
    return torch.fft.ifft(dFdt, dim=axis)


def gradient4(u: np.ndarray, dx: float, axis: int = -1) -> np.ndarray:
    """
    ∂u/∂x, 4-th-order everywhere (five-point stencil, incl. boundaries).

    Parameters
    ----------
    u    : ndarray
        Field values on an equally spaced grid.
    dx   : float
        Grid step along `axis`.
    axis : int, optional
        Axis along which to differentiate (default last axis).

    Returns
    -------
    du_dx : ndarray
        Same shape as `u`.  All points, incl. edges, are 4-th-order accurate.
    """
    u  = np.moveaxis(u, axis, -1)           # work on last axis
    Nz = u.shape[-1]
    if Nz < 5:
        raise ValueError("Need at least 5 points for 4-th-order stencil")

    du = np.empty_like(u, dtype=u.dtype)

    # ---- interior (central 5-point)  ---------------------------------
    um2 = np.roll(u,  2, axis=-1)
    um1 = np.roll(u,  1, axis=-1)
    up1 = np.roll(u, -1, axis=-1)
    up2 = np.roll(u, -2, axis=-1)
    du_c = (-up2 + 8*up1 - 8*um1 + um2) / (12*dx)
    du[..., 2:-2] = du_c[..., 2:-2]

    # ---- left boundary (one-sided) -----------------------------------
    du[..., 0] = (-25*u[..., 0] + 48*u[..., 1] - 36*u[..., 2] + 16*u[..., 3] -  3*u[..., 4]) / (12*dx)
    du[..., 1] = (-3*u[..., 0]  - 10*u[..., 1] + 18*u[..., 2] - 6*u[..., 3] +  1*u[..., 4]) / (12*dx)

    # ---- right boundary (one-sided, mirrored) ------------------------
    du[..., -2] = ( 3*u[..., -1] + 10*u[..., -2] - 18*u[..., -3] + 6*u[..., -4] -  1*u[..., -5]) / (12*dx)
    du[..., -1] = ( 25*u[..., -1] - 48*u[..., -2] + 36*u[..., -3] - 16*u[..., -4] +  3*u[..., -5]) / (12*dx)

    return np.moveaxis(du, -1, axis)