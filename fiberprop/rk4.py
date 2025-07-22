import numpy as np

def rk4_step(u, v, dt, dz, rhs_u, rhs_v, feedback_coefficient, boundary_condition, boundary_condition_derivative):
    """Классический RK-4 для системы  du/dt=f , dv/dt=g ."""
    k1u = rhs_u(u, v, dz)
    k1v = rhs_v(u, v, dz)

    k2u = rhs_u(u + 0.5*dt*k1u, v + 0.5*dt*k1v, dz)
    k2v = rhs_v(u + 0.5*dt*k1u, v + 0.5*dt*k1v, dz)

    k3u = rhs_u(u + 0.5*dt*k2u, v + 0.5*dt*k2v, dz)
    k3v = rhs_v(u + 0.5*dt*k2u, v + 0.5*dt*k2v, dz)

    k4u = rhs_u(u + dt*k3u, v + dt*k3v, dz)
    k4v = rhs_v(u + dt*k3u, v + dt*k3v, dz)

    u_next = u + dt/6. * (k1u + 2*k2u + 2*k3u + k4u)
    v_next = v + dt/6. * (k1v + 2*k2v + 2*k3v + k4v)

    u_next[:,0] = u_next[:,-1] * feedback_coefficient + boundary_condition
    v_next[:,0] = v_next[:,-1] * feedback_coefficient + boundary_condition_derivative

    return u_next, v_next


# -------------------- коэффициенты Gauss–Legendre RK4 -----------------
sqrt3 = np.sqrt(3.0)
c1, c2   = 0.5 - sqrt3/6.0, 0.5 + sqrt3/6.0
a11      = 0.25
a12      = 0.25 - sqrt3/6.0
a21      = 0.25 + sqrt3/6.0
a22      = 0.25
b1, b2   = 0.5, 0.5       # весы для итогового шага

A = np.array([[a11, a12],
              [a21, a22]])
c = np.array([c1,  c2])
b = np.array([b1,  b2])

# -------------------- неявный шаг RK4 ----------------------------------
def rk4_implicit_step(u, v, dt, dz, rhs_u, rhs_v,
                 feedback_coefficient, bc0, bc0d,
                 max_iter=20, tol=1e-10):
    """
    Неявная двухстадийная схема 4-го порядка (Gauss–Legendre).
    Все массивы shape = (C, Nz).  Краевые узлы (j=0, -1)
    обновляются после основного шага через feedback/boundary_condition.
    """
    # ---------------- первоначальная инициализация стадий --------------
    # k_SU, k_SV: список из 2 массивов стадий (u̇, v̇)
    k_u = [rhs_u(u, v, dz), rhs_u(u, v, dz)]    # начальный guess
    k_v = [rhs_v(u, v, dz), rhs_v(u, v, dz)]

    for it in range(max_iter):
        k_u_old0 = k_u[0].copy()      # для критерия сходимости

        # пересчитываем две стадии
        for s in (0, 1):
            uu = u.copy()
            vv = v.copy()
            # сумма a_sj * k_j
            for j in (0, 1):
                uu += dt * A[s, j] * k_u[j]
                vv += dt * A[s, j] * k_v[j]

            k_u[s] = rhs_u(uu, vv, dz)
            k_v[s] = rhs_v(uu, vv, dz)

        if np.linalg.norm(k_u[0] - k_u_old0) < tol:
            break
    #else:
    #    print("warning: implicit RK4 did not converge within", max_iter, "iterations")

    # -------------- объединяем стадии в шаг ---------------------------
    incr_u = dt * (b1 * k_u[0] + b2 * k_u[1])
    incr_v = dt * (b1 * k_v[0] + b2 * k_v[1])

    u_next = u + incr_u
    v_next = v + incr_v

    # ----------- краевые условия (feedback / заданные) ----------------
    u_next[:, 0]  = u_next[:, -1] * feedback_coefficient + bc0
    v_next[:, 0]  = v_next[:, -1] * feedback_coefficient + bc0d

    return u_next, v_next