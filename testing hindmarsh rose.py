import numpy as np
import matplotlib.pyplot as plt

USE_SCIPY = True
try:
    from scipy.integrate import solve_ivp
except Exception:
    USE_SCIPY = False


# -----------------------------
# Hindmarsh-Rose parameters
# -----------------------------
a = 1.0
b = 3.0
c = 1.0
d = 5.0
r = 0.006
s = 4.0
x_R = -1.6

# Drive + repulsive coupling (through x)
I = 3.25          # oscillatory regime 
g = 0.10          # repulsive coupling strength 


def hr2_rhs(t, U):
    """
    Two Hindmarsh-Rose neurons with repulsive x-coupling:
      x1' = ... + g*(x1 - x2)
      x2' = ... + g*(x2 - x1)
    """
    x1, y1, z1, x2, y2, z2 = U

    dx1 = y1 - a*x1**3 + b*x1**2 - z1 + I + g*(x1 - x2)
    dy1 = c - d*x1**2 - y1
    dz1 = r * (s*(x1 - x_R) - z1)

    dx2 = y2 - a*x2**3 + b*x2**2 - z2 + I + g*(x2 - x1)
    dy2 = c - d*x2**2 - y2
    dz2 = r * (s*(x2 - x_R) - z2)

    return np.array([dx1, dy1, dz1, dx2, dy2, dz2], dtype=float)


def rk4_integrate(f, t0, t1, dt, U0):
    n = int(np.floor((t1 - t0) / dt)) + 1
    t = t0 + dt * np.arange(n)
    U = np.zeros((n, len(U0)), dtype=float)
    U[0] = U0

    for k in range(n - 1):
        tk = t[k]
        uk = U[k]
        k1 = f(tk, uk)
        k2 = f(tk + 0.5*dt, uk + 0.5*dt*k1)
        k3 = f(tk + 0.5*dt, uk + 0.5*dt*k2)
        k4 = f(tk + dt, uk + dt*k3)
        U[k+1] = uk + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return t, U


def upward_crossings(t, x, threshold=0.0):
    """
    Linear interpolation of upward threshold crossings.
    """
    idx = np.where((x[:-1] < threshold) & (x[1:] >= threshold))[0]
    tc = []
    for i in idx:
        x0, x1 = x[i], x[i+1]
        if x1 == x0:
            tc.append(t[i])
        else:
            alpha = (threshold - x0) / (x1 - x0)
            tc.append(t[i] + alpha * (t[i+1] - t[i]))
    return np.array(tc)


def estimate_phase_offset_from_spikes(t, x1, x2, t_transient=1000.0, threshold=0.0):
    """
    Estimate relative phase using spike timings:
    For each cycle of neuron 1, find a spike of neuron 2 inside the interval and compute
    phase offset = 2*pi*(t2 - t1)/(t1_next - t1). Antiphase => ~ pi.
    """
    mask = t >= t_transient
    tt = t[mask]
    xx1 = x1[mask]
    xx2 = x2[mask]

    s1 = upward_crossings(tt, xx1, threshold=threshold)
    s2 = upward_crossings(tt, xx2, threshold=threshold)

    if len(s1) < 3 or len(s2) < 2:
        return None

    phases = []
    for k in range(len(s1) - 1):
        t_start = s1[k]
        t_end = s1[k+1]
        T = t_end - t_start
        if T <= 0:
            continue
        inside = s2[(s2 > t_start) & (s2 < t_end)]
        if len(inside) == 0:
            continue
        # Use the first spike of neuron 2 in this cycle
        offset = inside[0] - t_start
        phi = 2*np.pi * (offset / T)
        # wrap to [0, 2pi)
        phi = phi % (2*np.pi)
        phases.append(phi)

    if len(phases) == 0:
        return None

    phases = np.array(phases)
    # Circular mean
    mean_angle = np.angle(np.mean(np.exp(1j*phases))) % (2*np.pi)
    # Circular spread 
    R = np.abs(np.mean(np.exp(1j*phases)))
    return {
        "phases": phases,
        "mean_phase": mean_angle,
        "R": R,
        "n_cycles_used": len(phases),
    }


# --
# Initial conditions 
# -----------------------------
U0 = np.array([
    -1.2, -7.0, 3.0,    # neuron 1: x1,y1,z1
     1.2, -7.01, 3.0    # neuron 2: x2,y2,z2 (tiny mismatch helps avoid symmetry artifacts)
], dtype=float)

# -----------------------------
# Integration settings
# ---------------------
t0 = 0.0
t1 = 5000.0
plot_from = 1000.0  # discard transient in plots

if USE_SCIPY:
    # Uniform output grid for clean plotting
    dt_plot = 0.05
    t_eval = np.arange(t0, t1 + dt_plot, dt_plot)
    sol = solve_ivp(
        hr2_rhs, (t0, t1), U0,
        method="DOP853",
        t_eval=t_eval,
        max_step=0.2,
        rtol=1e-7, atol=1e-9
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    t = sol.t
    U = sol.y.T
else:
    # Pure NumPy fallback
    dt = 0.02
    t, U = rk4_integrate(hr2_rhs, t0, t1, dt, U0)

x1 = U[:, 0]
y1 = U[:, 1]
z1 = U[:, 2]
x2 = U[:, 3]
y2 = U[:, 4]
z2 = U[:, 5]

# -----------------------------
# Estimate phase offset after transient
# -----------------------------
phase_info = estimate_phase_offset_from_spikes(t, x1, x2, t_transient=plot_from, threshold=0.0)

if phase_info is None:
    print("Could not estimate phase offset robustly from spike crossings.")
    print("Try changing threshold, increasing runtime, or adjusting I/g slightly.")
else:
    mean_phi = phase_info["mean_phase"]
    mean_phi_deg = np.degrees(mean_phi)
    print(f"Estimated mean relative phase (neuron2 relative to neuron1): {mean_phi:.4f} rad ({mean_phi_deg:.2f} deg)")
    print(f"Cycles used: {phase_info['n_cycles_used']}, circular coherence R: {phase_info['R']:.4f}")
    print("Target antiphase = pi rad = 180 deg")

# -----------------------------
# Plotting
# -----------------------------
mask_tail = t >= plot_from
tt = t[mask_tail]
x1_tail = x1[mask_tail]
x2_tail = x2[mask_tail]

# A shorter zoom window from the tail to visually show many periods but still readable
zoom_window = 400.0  
mask_zoom = t >= (t1 - zoom_window)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# 1) Long tail (post-transient) overlay
axes[0].plot(tt, x1_tail, label="x1(t)")
axes[0].plot(tt, x2_tail, label="x2(t)", alpha=0.85)
axes[0].set_title(f"Two Hindmarsh-Rose neurons with repulsive coupling (post-transient, t >= {plot_from})")
axes[0].set_ylabel("Membrane variable x")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# 2) Zoomed view near the end to see antiphase clearly
axes[1].plot(t[mask_zoom], x1[mask_zoom], label="x1(t)")
axes[1].plot(t[mask_zoom], x2[mask_zoom], label="x2(t)", alpha=0.85)
axes[1].set_title(f"Zoomed tail (last {zoom_window} time units)")
axes[1].set_ylabel("Membrane variable x")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

# 3) Phase portrait x1 vs x2 (antiphase tends to cluster near opposite-sign relation)
axes[2].plot(x1_tail, x2_tail, lw=0.8)
axes[2].set_title("Phase-plane projection (x1 vs x2) after transient")
axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()