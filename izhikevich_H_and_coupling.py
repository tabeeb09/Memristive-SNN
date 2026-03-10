import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


def wrap_phase(phi: float) -> float:
    """Wrap a phase angle to (-pi, pi]."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi



def softplus(x: float, beta: float = 20.0) -> float:
    """softmax google in context of ml activation functions"""
    # Prevent overflow for large positive beta*x
    y = beta * x
    if y > 50.0:
        return x
    if y < -50.0:
        return math.exp(y) / beta
    return math.log1p(math.exp(y)) / beta



# Izhikevich neuron simulator



@dataclass
class IzhikevichParams:
    a: float = 0.02
    b: float = 0.2
    c: float = -0.0650
    d: float = 0.008
    I_ext: float = 10.0


@dataclass
class CycleData:
    t_cycle: np.ndarray
    v_cycle: np.ndarray
    u_cycle: np.ndarray
    T: float
    spike_times: np.ndarray
    dt: float


class IzhikevichOscillator:
    """
    Simple explicit-Euler simulator for a tonic-spiking Izhikevich neuron. Coulda used rk4, but its good practise.

    Equations:
        dv/dt = 0.04 v^2 + 5 v + 140 - u + I_ext + I_inj(t)
        du/dt = a (b v - u)
    Reset when v >= 30:
        v <- c
        u <- u + d
    """

    def __init__(self, params: IzhikevichParams, dt: float = 0.02):
        self.params = params
        self.dt = float(dt)

    def _step(self, v: float, u: float, I_total: float):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I_total
        du = self.params.a * (self.params.b * v - u)
        v_new = v + self.dt * dv
        u_new = u + self.dt * du

        spiked = False
        if v_new >= 30.0:
            spiked = True
            v_new = self.params.c
            u_new = u_new + self.params.d

        return v_new, u_new, spiked

    def simulate(
        self,
        t_final: float,
        v0: Optional[float] = None,
        u0: Optional[float] = None,
        pulse_start: Optional[float] = None,
        pulse_width: float = 0.5,
        pulse_amplitude: float = 0.0,
        stop_after_next_spike: bool = False,
    ):
        """
        Simulate from t=0 to t_final.

        A current pulse can be injected into the voltage equation over
        [pulse_start, pulse_start + pulse_width).
        """
        n_steps = int(np.ceil(t_final / self.dt))
        t = np.arange(n_steps + 1, dtype=float) * self.dt

        if v0 is None:
            v0 = self.params.c
        if u0 is None:
            u0 = self.params.b * v0

        v = np.empty_like(t)
        u = np.empty_like(t)
        v[0] = v0
        u[0] = u0
        spike_times = []

        spike_after_pulse = False

        for k in range(n_steps):
            tk = t[k]
            I_total = self.params.I_ext
            if pulse_start is not None and pulse_start <= tk < pulse_start + pulse_width:
                I_total += pulse_amplitude

            v[k + 1], u[k + 1], spiked = self._step(v[k], u[k], I_total)
            if spiked:
                spike_times.append(t[k + 1])
                if pulse_start is not None and tk >= pulse_start:
                    spike_after_pulse = True
                if stop_after_next_spike and spike_after_pulse:
                    t = t[: k + 2]
                    v = v[: k + 2]
                    u = u[: k + 2]
                    break

        return {
            "t": t,
            "v": v,
            "u": u,
            "spike_times": np.array(spike_times, dtype=float),
        }

    def find_limit_cycle(self, burn_in: float = 1500.0, record_time: float = 500.0) -> CycleData:
        
        # Run long enough to reach a stable orbit, then extract
    
        sim = self.simulate(burn_in + record_time)
        spikes = sim["spike_times"]
        if len(spikes) < 3:
            raise RuntimeError(
                "Not enough spikes to estimate a limit cycle. Increase I_ext, burn_in, or record_time."
            )

        # Use the last cycle for the waveform.
        t_prev = spikes[-2]
        t_last = spikes[-1]
        T = t_last - t_prev

        t = sim["t"]
        mask = (t >= t_prev) & (t < t_last)
        t_cycle = t[mask] - t_prev
        v_cycle = sim["v"][mask]
        u_cycle = sim["u"][mask]

        if len(t_cycle) < 5:
            raise RuntimeError("Cycle extraction failed; dt may be too large.")

        return CycleData(
            t_cycle=t_cycle,
            v_cycle=v_cycle,
            u_cycle=u_cycle,
            T=T,
            spike_times=spikes,
            dt=self.dt,
        )




def periodic_interp(samples_x: np.ndarray, samples_y: np.ndarray, x_query: np.ndarray, period: float):
    """Periodic linear interpolation for 1D samples."""
    xq = np.mod(x_query, period)
    xp = np.asarray(samples_x, dtype=float)
    yp = np.asarray(samples_y, dtype=float)

    # Append endpoint for periodic continuity.
    xp_ext = np.concatenate([xp, [period]])
    yp_ext = np.concatenate([yp, [yp[0]]])
    return np.interp(xq, xp_ext, yp_ext)



# Numerical iPRC estimate by weak current kicks, control theory impulse response estimation, and building H for diffusive coupling, followed by solving for coupling strengths to achieve a target phase-locked state.


@dataclass
class PRCData:
    phases: np.ndarray          # in radians, [0, 2pi)
    phases_time: np.ndarray     # in time units, [0, T)
    Z_v: np.ndarray             # estimated voltage component of iPRC
    pulse_amplitude: float
    pulse_width: float



def estimate_voltage_iprc(
    osc: IzhikevichOscillator,
    cycle: CycleData,
    n_phase_samples: int = 80,
    pulse_amplitude: float = 0.05,
    pulse_width: float = 0.5,
) -> PRCData:
    """
    Estimate the voltage component Z_v of the iPRC using weak square pulses.

    At a phase phi, inject a small current pulse into the voltage equation,
    then measure the shift in the next spike time:
        Z_v(phi) approx -(2*pi/T) * Delta_t_spike / pulse_area
    where pulse_area = amplitude * width.
    """
    T = cycle.T
    dt = cycle.dt

    t_cycle = cycle.t_cycle
    v_cycle = cycle.v_cycle
    u_cycle = cycle.u_cycle

    phases_time = np.linspace(0.0, T, n_phase_samples, endpoint=False)
    phases = 2.0 * np.pi * phases_time / T
    Z_v = np.zeros_like(phases)

    pulse_area = pulse_amplitude * pulse_width
    if pulse_area == 0.0:
        raise ValueError("Pulse area must be nonzero.")

    # Simulate from each sampled point on the cycle until the next spike,
    # once with no perturbation and once with a weak pulse.
    t_horizon = max(2.0 * T + 5.0 * pulse_width, T + 20.0 * dt)

    for i, tau in enumerate(phases_time):
        v0 = periodic_interp(t_cycle, v_cycle, np.array([tau]), T)[0]
        u0 = periodic_interp(t_cycle, u_cycle, np.array([tau]), T)[0]

        base = osc.simulate(
            t_final=t_horizon,
            v0=v0,
            u0=u0,
            stop_after_next_spike=True,
        )
        pert = osc.simulate(
            t_final=t_horizon,
            v0=v0,
            u0=u0,
            pulse_start=0.0,
            pulse_width=pulse_width,
            pulse_amplitude=pulse_amplitude,
            stop_after_next_spike=True,
        )

        if len(base["spike_times"]) == 0 or len(pert["spike_times"]) == 0:
            raise RuntimeError("Failed to capture next spike while estimating iPRC.")

        dt_spike = pert["spike_times"][0] - base["spike_times"][0]
        Z_v[i] = -(2.0 * np.pi / T) * (dt_spike / pulse_area)

    return PRCData(
        phases=phases,
        phases_time=phases_time,
        Z_v=Z_v,
        pulse_amplitude=pulse_amplitude,
        pulse_width=pulse_width,
    )


# Building H for diffusive voltage coupling ????????????????


@dataclass
class InteractionData:
    phases: np.ndarray     # [0, 2pi)
    H_values: np.ndarray
    Hp_values: np.ndarray

    def H(self, phi: float) -> float:
        ph = np.mod(phi, 2.0 * np.pi)
        return np.interp(ph, np.concatenate([self.phases, [2.0 * np.pi]]), np.concatenate([self.H_values, [self.H_values[0]]]))

    def Hp(self, phi: float) -> float:
        ph = np.mod(phi, 2.0 * np.pi)
        return np.interp(ph, np.concatenate([self.phases, [2.0 * np.pi]]), np.concatenate([self.Hp_values, [self.Hp_values[0]]]))



def build_interaction_function_diffusive(cycle: CycleData, prc: PRCData, n_phi: int = 256) -> InteractionData:
    """
    For diffusive voltage coupling I_cpl ~ epsilon * (v_pre - v_post), build
        H(phi) = (1/T) ∫ Z_v(t) [v(t + phi) - v(t)] dt
    sampled on a uniform grid in phi in [0, 2pi).
    """
    T = cycle.T
    t_cycle = cycle.t_cycle
    v_cycle = cycle.v_cycle

    # Resample Z_v onto the same time grid as the stored cycle waveform.
    Z_t = periodic_interp(prc.phases_time, prc.Z_v, t_cycle, T)

    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    H_values = np.zeros_like(phi_grid)

    v_t = v_cycle
    for k, phi in enumerate(phi_grid):
        tau = (phi / (2.0 * np.pi)) * T
        v_shift = periodic_interp(t_cycle, v_cycle, t_cycle + tau, T)
        integrand = Z_t * (v_shift - v_t)
        H_values[k] = np.trapz(integrand, t_cycle) / T

    # Periodic derivative dH/dphi using central differences on the uniform grid.
    dphi = phi_grid[1] - phi_grid[0]
    Hp_values = (np.roll(H_values, -1) - np.roll(H_values, 1)) / (2.0 * dphi)

    return InteractionData(phases=phi_grid, H_values=H_values, Hp_values=Hp_values)





def build_locking_system(a_star: float,
                         b_star: float,
                         H_func: Callable[[float], float],
                         detuning21: float = 0.0,
                         detuning31: float = 0.0):
    a = wrap_phase(a_star)
    b = wrap_phase(b_star)

    A = np.array([
        [-H_func(a), -H_func(b),  H_func(-a), H_func(b - a), 0.0,         0.0],
        [-H_func(a), -H_func(b),  0.0,        0.0,         H_func(-b), H_func(a - b)],
    ], dtype=float)

    rhs = np.array([-detuning21, -detuning31], dtype=float)
    return A, rhs



def affine_solution_space(A: np.ndarray, rhs: np.ndarray, tol: float = 1e-10):
    e0, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(s > tol)
    N = Vt[rank:].T
    return e0, N



def coupling_from_zeta(e0: np.ndarray, N: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    return e0 + N @ zeta



def locking_residual(a_star: float,
                     b_star: float,
                     e: np.ndarray,
                     H_func: Callable[[float], float],
                     detuning21: float = 0.0,
                     detuning31: float = 0.0) -> np.ndarray:
    A, rhs = build_locking_system(a_star, b_star, H_func, detuning21, detuning31)
    return A @ e - rhs



def jacobian_at_target(a_star: float, b_star: float, e: np.ndarray, Hp_func: Callable[[float], float]) -> np.ndarray:
    a = wrap_phase(a_star)
    b = wrap_phase(b_star)

    e12, e13, e21, e23, e31, e32 = e

    J11 = -e21 * Hp_func(-a) - e23 * Hp_func(b - a) - e12 * Hp_func(a)
    J12 =  e23 * Hp_func(b - a) - e13 * Hp_func(b)
    J21 =  e32 * Hp_func(a - b) - e12 * Hp_func(a)
    J22 = -e31 * Hp_func(-b) - e32 * Hp_func(a - b) - e13 * Hp_func(b)

    return np.array([[J11, J12], [J21, J22]], dtype=float)



def stability_metrics(J: np.ndarray):
    tr = float(np.trace(J))
    det = float(np.linalg.det(J))
    eigvals = np.linalg.eigvals(J)
    return tr, det, eigvals



def is_stable_2x2(J: np.ndarray, trace_margin: float = 1e-6, det_margin: float = 1e-6) -> bool:
    tr, det, _ = stability_metrics(J)
    return (tr < -trace_margin) and (det > det_margin)



def penalty(zeta: np.ndarray,
            e0: np.ndarray,
            N: np.ndarray,
            a_star: float,
            b_star: float,
            Hp_func: Callable[[float], float],
            trace_margin: float = 1e-3,
            det_margin: float = 1e-3,
            reg: float = 1e-4,
            coupling_bound: Optional[float] = None) -> float:
    e = coupling_from_zeta(e0, N, zeta)
    J = jacobian_at_target(a_star, b_star, e, Hp_func)
    tr, det, _ = stability_metrics(J)

    p_tr = softplus(tr + trace_margin) ** 2
    p_det = softplus(det_margin - det) ** 2
    p_reg = reg * float(zeta @ zeta)

    p_bound = 0.0
    if coupling_bound is not None:
        excess = np.maximum(np.abs(e) - coupling_bound, 0.0)
        p_bound = 100.0 * float(excess @ excess)

    return p_tr + p_det + p_reg + p_bound



def finite_difference_grad(f: Callable[[np.ndarray], float], z: np.ndarray, h: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(z)
    for k in range(len(z)):
        zp = z.copy()
        zm = z.copy()
        zp[k] += h
        zm[k] -= h
        g[k] = (f(zp) - f(zm)) / (2.0 * h)
    return g



def generate_coupling(a_star: float,
                      b_star: float,
                      H_func: Callable[[float], float],
                      Hp_func: Callable[[float], float],
                      detuning21: float = 0.0,
                      detuning31: float = 0.0,
                      restarts: int = 64,
                      iters_per_restart: int = 200,
                      initial_step: float = 0.1,
                      random_scale: float = 1.0,
                      trace_margin: float = 1e-3,
                      det_margin: float = 1e-3,
                      reg: float = 1e-4,
                      coupling_bound: Optional[float] = None,
                      seed: int = 0):
    A, rhs = build_locking_system(a_star, b_star, H_func, detuning21, detuning31)
    e0, N = affine_solution_space(A, rhs)

    if N.shape[1] == 0:
        e = e0
        J = jacobian_at_target(a_star, b_star, e, Hp_func)
        tr, det, eigvals = stability_metrics(J)
        return {
            "epsilon": e,
            "stable": is_stable_2x2(J, trace_margin, det_margin),
            "locking_residual": locking_residual(a_star, b_star, e, H_func, detuning21, detuning31),
            "trace": tr,
            "det": det,
            "eigvals": eigvals,
            "jacobian": J,
            "e0": e0,
            "N": N,
            "zeta": np.zeros(0),
        }

    rng = np.random.default_rng(seed)

    def obj(z):
        return penalty(
            z, e0, N, a_star, b_star, Hp_func,
            trace_margin=trace_margin,
            det_margin=det_margin,
            reg=reg,
            coupling_bound=coupling_bound,
        )

    best = None
    best_val = np.inf
    starts = [np.zeros(N.shape[1])]
    for _ in range(restarts - 1):
        starts.append(random_scale * rng.standard_normal(N.shape[1]))

    for z0 in starts:
        z = z0.copy()
        step = initial_step
        val = obj(z)

        for _ in range(iters_per_restart):
            g = finite_difference_grad(obj, z)
            if np.linalg.norm(g) < 1e-10:
                break

            accepted = False
            for _ in range(12):
                z_new = z - step * g
                val_new = obj(z_new)
                if val_new < val:
                    z = z_new
                    val = val_new
                    step *= 1.05
                    accepted = True
                    break
                else:
                    step *= 0.5
            if not accepted:
                break

        e = coupling_from_zeta(e0, N, z)
        J = jacobian_at_target(a_star, b_star, e, Hp_func)
        tr, det, eigvals = stability_metrics(J)
        stable = is_stable_2x2(J, trace_margin, det_margin)

        result = {
            "epsilon": e,
            "stable": stable,
            "locking_residual": locking_residual(a_star, b_star, e, H_func, detuning21, detuning31),
            "trace": tr,
            "det": det,
            "eigvals": eigvals,
            "jacobian": J,
            "e0": e0,
            "N": N,
            "zeta": z,
            "objective": val,
        }

        if stable:
            return result
        if val < best_val:
            best_val = val
            best = result

    return best





def find_H_and_generate_coupling(
    a_star: float,
    b_star: float,
    neuron_params: Optional[IzhikevichParams] = None,
    dt: float = 0.02,
    burn_in: float = 1500.0,
    record_time: float = 500.0,
    n_prc_samples: int = 80,
    pulse_amplitude: float = 0.05,
    pulse_width: float = 0.5,
    n_H_samples: int = 256,
    detuning21: float = 0.0,
    detuning31: float = 0.0,
    search_kwargs: Optional[Dict] = None,
):
    """
    Full workflow:
      1) simulate uncoupled Izhikevich neuron to get a stable cycle
      2) estimate Z_v by weak current kicks
      3) build H(phi) for diffusive voltage coupling
      4) feed H and H' into the coupling solver
    """
    if neuron_params is None:
        neuron_params = IzhikevichParams()
    if search_kwargs is None:
        search_kwargs = {}

    osc = IzhikevichOscillator(neuron_params, dt=dt)
    cycle = osc.find_limit_cycle(burn_in=burn_in, record_time=record_time)
    prc = estimate_voltage_iprc(
        osc,
        cycle,
        n_phase_samples=n_prc_samples,
        pulse_amplitude=pulse_amplitude,
        pulse_width=pulse_width,
    )
    interaction = build_interaction_function_diffusive(cycle, prc, n_phi=n_H_samples)

    result = generate_coupling(
        a_star=a_star,
        b_star=b_star,
        H_func=interaction.H,
        Hp_func=interaction.Hp,
        detuning21=detuning21,
        detuning31=detuning31,
        **search_kwargs,
    )

    return {
        "neuron_params": neuron_params,
        "cycle": cycle,
        "prc": prc,
        "interaction": interaction,
        "coupling_result": result,
    }




# g rb, gr b 
if __name__ == "__main__":
    # Target phase differences for the 3-neuron reduced system.
    a_star = 1.0
    b_star = -2.0
    

    out = find_H_and_generate_coupling(
        a_star=a_star,
        b_star=b_star,
        neuron_params=IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0, I_ext=10.0),
        dt=0.02,
        burn_in=1200.0,
        record_time=400.0,
        n_prc_samples=48,
        pulse_amplitude=0.03,
        pulse_width=0.4,
        n_H_samples=192,
        search_kwargs={
            "restarts": 40,
            "iters_per_restart": 150,
            "initial_step": 0.1,
            "random_scale": 1.5,
            "trace_margin": 1e-3,
            "det_margin": 1e-3,
            "reg": 1e-4,
            "coupling_bound": 10.0,
            "seed": 1,
        },
    )

    cycle = out["cycle"]
    prc = out["prc"]
    interaction = out["interaction"]
    result = out["coupling_result"]

    print("Estimated period T:", cycle.T)
    print("First five Z_v samples:", prc.Z_v[:5])
    print("First five H samples:", interaction.H_values[:5])
    print("Order of epsilon: [e12, e13, e21, e23, e31, e32]")
    print("epsilon =", result["epsilon"])
    print("stable  =", result["stable"])
    print("trace   =", result["trace"])
    print("det     =", result["det"])
    print("eigvals =", result["eigvals"])
    print("locking residual =", result["locking_residual"])
    print("jacobian =\n", result["jacobian"])
