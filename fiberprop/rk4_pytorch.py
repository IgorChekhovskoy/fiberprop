import torch

# ───────────────────────── явный RK-4 ― PyTorch ──────────────────────────
def rk4_step_torch(u, v, dt, dz, rhs_u, rhs_v,
                   feedback_coefficient, boundary_condition,
                   boundary_condition_derivative):
    """
    u, v                       : torch.Tensor  shape (C, Nz)
    feedback_coefficient       : complex или torch комплекс-скаляр
    boundary_condition         : torch.Tensor shape (C,) или (C, 1)
    boundary_condition_derivative : то же для производной
    """
    k1u = rhs_u(u, v, dz)
    k1v = rhs_v(u, v, dz)

    k2u = rhs_u(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v, dz)
    k2v = rhs_v(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v, dz)

    k3u = rhs_u(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v, dz)
    k3v = rhs_v(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v, dz)

    k4u = rhs_u(u + dt * k3u, v + dt * k3v, dz)
    k4v = rhs_v(u + dt * k3u, v + dt * k3v, dz)

    u_next = u + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
    v_next = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)

    # краевая ячейка (j = 0)
    u_next[:, 0] = u_next[:, -1] * feedback_coefficient + boundary_condition
    v_next[:, 0] = v_next[:, -1] * feedback_coefficient + boundary_condition_derivative
    return u_next, v_next


# ────────────────── неявный 2-стадийный Gauss–RK-4 — PyTorch ─────────────
sqrt3 = 3.0 ** 0.5
A_t  = torch.tensor([[0.25,
                      0.25 - sqrt3 / 6.0],
                     [0.25 + sqrt3 / 6.0,
                      0.25]],
                    dtype=torch.float64)  # или .complex64 при нужде
b_t  = torch.tensor([0.5, 0.5], dtype=A_t.dtype)


def rk4_implicit_step_torch(u, v, dt, dz, rhs_u, rhs_v,
                            feedback_coefficient, bc0, bc0d,
                            max_iter: int = 100, tol: float = 1e-10):
    """
    Двухстадийная схема Gauss–Legendre (4-й порядок, A-stable)
    для PyTorch-тензоров.
    """
    dtype   = u.dtype
    device  = u.device
    A, b = A_t.to(device=device, dtype=dtype), b_t.to(device=device, dtype=dtype)

    # начальный guess: явный RHS
    k_u = [rhs_u(u, v, dz), rhs_u(u, v, dz)]
    k_v = [rhs_v(u, v, dz), rhs_v(u, v, dz)]

    for _ in range(max_iter):
        k_u_old0 = k_u[0].clone()

        for s in (0, 1):
            uu, vv = u.clone(), v.clone()
            uu = uu + dt * (A[s, 0] * k_u[0] + A[s, 1] * k_u[1])
            vv = vv + dt * (A[s, 0] * k_v[0] + A[s, 1] * k_v[1])

            k_u[s] = rhs_u(uu, vv, dz)
            k_v[s] = rhs_v(uu, vv, dz)

        if torch.norm(k_u[0] - k_u_old0) < tol:
            break

    incr_u = dt * (b[0] * k_u[0] + b[1] * k_u[1])
    incr_v = dt * (b[0] * k_v[0] + b[1] * k_v[1])

    u_next = u + incr_u
    v_next = v + incr_v

    # применяем обратную связь / граничные условия
    u_next[:, 0] = u_next[:, -1] * feedback_coefficient + bc0
    v_next[:, 0] = v_next[:, -1] * feedback_coefficient + bc0d
    return u_next, v_next
