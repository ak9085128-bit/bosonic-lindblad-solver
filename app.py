

import math
from itertools import product, combinations
from math import comb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qutip as qt
import streamlit as st
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply

# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Thermal Systems Solver",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Quantum Thermal Systems Solver")

page = st.sidebar.radio(
    "Choose simulation",
    [
        "Bosonic Lindblad Solver",
        "Fermionic Quantum Otto Engine",
    ],
)

# ============================================================
# 2. BOSONIC LINDBLAD SOLVER
# ============================================================

def make_basis(N, M):
    """All occupation states containing exactly N bosons."""
    return [
        tuple(occ)
        for occ in product(range(N + 1), repeat=M)
        if sum(occ) == N
    ]


def build_hamiltonian(basis, omega):
    energies = np.array([
        sum(i * omega * n for i, n in enumerate(occ))
        for occ in basis
    ], dtype=float)
    return qt.Qobj(np.diag(energies)), energies


def build_transitions(basis, M):
    """Adjacent-level bosonic lowering operators."""
    index = {state: i for i, state in enumerate(basis)}
    dim = len(basis)
    operators = []

    for i in range(M - 1):
        matrix = np.zeros((dim, dim), dtype=complex)

        for col, state in enumerate(basis):
            ni, nj = state[i], state[i + 1]

            if nj == 0:
                continue

            new_state = list(state)
            new_state[i] += 1
            new_state[i + 1] -= 1

            matrix[index[tuple(new_state)], col] = math.sqrt(
                (ni + 1) * nj
            )

        operators.append(qt.Qobj(matrix))

    return operators


def thermal_occupation(omega, temperature):
    if temperature <= 0:
        return 0.0

    x = omega / temperature

    if x > 700:
        return 0.0

    return 1.0 / np.expm1(x)


def solve_bosonic_system(N, M, omega, T, gamma, t_final, num_times):
    basis = make_basis(N, M)
    H, energies = build_hamiltonian(basis, omega)
    nbar = thermal_occupation(omega, T)

    c_ops = []

    for A in build_transitions(basis, M):
        c_ops.append(np.sqrt(gamma * (nbar + 1)) * A)

        if nbar > 0:
            c_ops.append(np.sqrt(gamma * nbar) * A.dag())

    ground = (N,) + (0,) * (M - 1)
    rho0 = qt.ket2dm(qt.basis(len(basis), basis.index(ground)))
    times = np.linspace(0, t_final, num_times)

    result = qt.mesolve(H, rho0, times, c_ops=c_ops, e_ops=[])

    return basis, energies, times, result, nbar


def show_bosonic_solver():
    st.header("Bosonic Lindblad Solver")
    st.write(
        "Fixed-particle-number bosons coupled to a thermal Markovian bath. "
        "We use kB = hbar = 1."
    )

    st.sidebar.subheader("Bosonic parameters")

    N = int(st.sidebar.number_input(
        "Number of bosons (N)", 1, 10, 2, key="b_N"
    ))
    M = int(st.sidebar.number_input(
        "Number of energy levels (M)", 2, 8, 3, key="b_M"
    ))
    omega = float(st.sidebar.number_input(
        "Energy spacing (ω)", 0.01, 100.0, 1.0, step=0.1,
        key="b_omega"
    ))
    T = float(st.sidebar.number_input(
        "Bath temperature (T)", 0.0, 100.0, 3.0, step=0.1,
        key="b_T"
    ))
    gamma = float(st.sidebar.number_input(
        "Bath coupling (γ)", 0.001, 100.0, 1.0, step=0.1,
        key="b_gamma"
    ))
    t_final = float(st.sidebar.number_input(
        "Final time", 0.1, 1000.0, 10.0, step=1.0,
        key="b_time"
    ))
    num_times = int(st.sidebar.number_input(
        "Number of time points", 11, 501, 101, step=10,
        key="b_points"
    ))

    dim = comb(N + M - 1, N)
    st.sidebar.caption(f"Hilbert-space dimension: {dim}")

    if dim > 200:
        st.warning("Reduce N or M: this simulation is limited to 200 states.")
        return

    if not st.button("▶ Run Bosonic Simulation", type="primary"):
        st.info("Set the parameters and run the bosonic simulation.")
        return

    try:
        with st.spinner("Solving the Lindblad master equation..."):
            basis, energies, times, result, nbar = solve_bosonic_system(
                N, M, omega, T, gamma, t_final, num_times
            )

            final_rho = result.states[-1]
            populations = np.array([
                np.real(np.diag(rho.full())) for rho in result.states
            ])
            labels = [str(s) for s in basis]

            c1, c2, c3 = st.columns(3)
            c1.metric("Basis states", len(basis))
            c2.metric("Thermal occupation", f"{nbar:.5f}")
            c3.metric("Final trace", f"{np.real(final_rho.tr()):.8f}")

            st.subheader("1. Fock-state populations")
            fig, ax = plt.subplots(figsize=(9, 5))

            chosen = np.argsort(populations[-1])[-min(8, len(basis)):]

            for j in chosen:
                ax.plot(times, populations[:, j], label=labels[j])

            ax.set_xlabel("Time")
            ax.set_ylabel("Population")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("2. Mean occupation of each level")
            mean_occ = np.zeros((len(times), M))

            for j, state in enumerate(basis):
                for level in range(M):
                    mean_occ[:, level] += populations[:, j] * state[level]

            fig, ax = plt.subplots(figsize=(9, 5))

            for level in range(M):
                ax.plot(times, mean_occ[:, level], label=f"Level {level}")

            ax.set_xlabel("Time")
            ax.set_ylabel("Mean boson number")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("3. Final density matrix")
            rho_df = pd.DataFrame(
                final_rho.full(), index=labels, columns=labels
            )
            st.dataframe(rho_df, use_container_width=True)
            st.download_button(
                "Download density matrix CSV",
                rho_df.to_csv().encode("utf-8"),
                "bosonic_density_matrix.csv",
                "text/csv",
            )

            st.subheader("4. Final populations")
            pop_df = pd.DataFrame({
                "Occupation state": labels,
                "Energy": energies,
                "Final population": populations[-1],
            })
            st.dataframe(pop_df, use_container_width=True)
            st.download_button(
                "Download populations CSV",
                pop_df.to_csv(index=False).encode("utf-8"),
                "bosonic_populations.csv",
                "text/csv",
            )

            rho = final_rho.full()
            st.subheader("5. Physical checks")
            st.write(f"Trace: {np.real(final_rho.tr()):.10f}")
            st.write(f"Hermitian: {final_rho.isherm}")
            st.write(
                "Hermiticity error: "
                f"{np.linalg.norm(rho - rho.conj().T):.3e}"
            )
            st.write(f"Total population: {populations[-1].sum():.10f}")

    except Exception as exc:
        st.error("Bosonic simulation failed.")
        st.exception(exc)


# ============================================================
# 3. SPIN-POLARIZED FERMIONIC QUANTUM OTTO ENGINE
# ============================================================

def generate_fermion_states(NF, M):
    """Each spin-polarized fermion level can be empty or occupied."""
    states = []

    for occupied in combinations(range(M), NF):
        state = [0] * M

        for i in occupied:
            state[i] = 1

        states.append(tuple(state))

    return states


def fermion_energy_coefficients(states):
    """Dimensionless energy coefficients q = sum(i * n_i)."""
    return np.array([
        sum(i * n for i, n in enumerate(state))
        for state in states
    ], dtype=float)


def make_fermion_edges(states, state_index, M):
    """Allowed moves: one fermion moves from an occupied level to an empty one."""
    edges = []

    for src, state in enumerate(states):
        for j in range(1, M):
            if state[j] == 0:
                continue

            for i in range(j):
                if state[i] == 1:
                    continue

                target = list(state)
                target[i] = 1
                target[j] = 0

                dst = state_index[tuple(target)]
                edges.append((src, dst, 1.0, j - i))

    return edges


def thermalize_fermion_population(
    p, edges, gamma, omega, T, tau
):
    """Sparse population master equation: dp/dt = G p."""
    if tau <= 0 or gamma <= 0:
        return p.copy()

    D = len(p)
    G = lil_matrix((D, D), dtype=float)
    outgoing = np.zeros(D)

    for src, dst, strength, gap_levels in edges:
        dE = omega * gap_levels

        if T <= 0:
            nbar = 0.0
        else:
            x = dE / T
            nbar = 0.0 if x > 700 else 1.0 / np.expm1(x)

        rate_down = gamma * (nbar + 1.0) * strength
        rate_up = gamma * nbar * strength

        G[dst, src] += rate_down
        G[src, dst] += rate_up

        outgoing[src] += rate_down
        outgoing[dst] += rate_up

    for i in range(D):
        G[i, i] = -outgoing[i]

    p_new = expm_multiply(G.tocsr() * tau, p)
    p_new = np.real(p_new)
    p_new = np.maximum(p_new, 0.0)

    total = p_new.sum()

    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Invalid population normalization.")

    return p_new / total


def one_fermion_otto_cycle(
    p, edges, q, gamma, wc, wh, Tc, Th, tau_c, tau_h
):
    """Cold isochore followed by hot isochore."""
    p_c = thermalize_fermion_population(
        p, edges, gamma, wc, Tc, tau_c
    )
    p_h = thermalize_fermion_population(
        p_c, edges, gamma, wh, Th, tau_h
    )

    mean_q_c = np.dot(q, p_c)
    mean_q_h = np.dot(q, p_h)

    # Work extracted in this cycle; same convention as supplied code.
    Wout = (wh - wc) * (mean_q_h - mean_q_c)

    return p_h, p_c, p_h, Wout


def run_fermion_simulation(
    M, Tc, Th, gamma, wh, wc_values, tau_c, tau_h, Ncycles
):
    all_work = {}
    summary = []

    for NF in range(1, M):
        states = generate_fermion_states(NF, M)
        state_index = {s: i for i, s in enumerate(states)}
        D = comb(M, NF)
        q = fermion_energy_coefficients(states)
        edges = make_fermion_edges(states, state_index, M)

        ground = tuple(1 if i < NF else 0 for i in range(M))
        p_initial = np.zeros(D)
        p_initial[state_index[ground]] = 1.0

        work_values = []

        for wc in wc_values:
            p = p_initial.copy()
            Wout = 0.0

            for _ in range(Ncycles):
                p, p_c, p_h, Wout = one_fermion_otto_cycle(
                    p, edges, q, gamma, float(wc), wh,
                    Tc, Th, tau_c, tau_h
                )

            work_values.append(Wout)

        all_work[NF] = np.asarray(work_values)

        summary.append({
            "Fermion number (NF)": NF,
            "Hilbert dimension": D,
            "Transitions": len(edges),
        })

    return all_work, pd.DataFrame(summary)


def show_fermionic_engine():
    st.header("Fermionic Quantum Otto Engine")
    st.write(
        "Spin-polarized fermions in M equally spaced energy levels. "
        "The app automatically simulates NF = 1, 2, ..., M−1."
    )

    st.info(
        "The plotted quantity is the work output in the final cycle, "
        "matching your original program. The population model uses a sparse "
        "rate matrix to reduce memory use."
    )

    st.sidebar.subheader("Fermionic engine parameters")

    M = int(st.sidebar.number_input(
        "Fermion energy levels (M)", 2, 8, 3, key="f_M"
    ))
    Tc = float(st.sidebar.number_input(
        "Cold temperature (Tc)", 0.01, 100.0, 1.0, step=0.1, key="f_Tc"
    ))
    Th = float(st.sidebar.number_input(
        "Hot temperature (Th)", 0.01, 100.0, 3.0, step=0.1, key="f_Th"
    ))
    gamma = float(st.sidebar.number_input(
        "Bath coupling (γ)", 0.001, 100.0, 1.0, step=0.1, key="f_gamma"
    ))
    wh = float(st.sidebar.number_input(
        "Hot frequency (ωh)", 0.01, 100.0, 2.0, step=0.1, key="f_wh"
    ))
    wc_min = float(st.sidebar.number_input(
        "Minimum cold frequency (ωc)", 0.01, 100.0, 0.2,
        step=0.1, key="f_wc_min"
    ))
    wc_max = float(st.sidebar.number_input(
        "Maximum cold frequency (ωc)", 0.02, 100.0, 1.9,
        step=0.1, key="f_wc_max"
    ))
    wc_points = int(st.sidebar.number_input(
        "Number of frequency points", 5, 100, 30, step=5,
        key="f_wc_points"
    ))
    tau_c = float(st.sidebar.number_input(
        "Cold contact time (τc)", 0.0, 100.0, 2.0,
        step=0.5, key="f_tau_c"
    ))
    tau_h = float(st.sidebar.number_input(
        "Hot contact time (τh)", 0.0, 100.0, 2.0,
        step=0.5, key="f_tau_h"
    ))
    Ncycles = int(st.sidebar.number_input(
        "Number of Otto cycles", 1, 100, 10, key="f_cycles"
    ))

    st.sidebar.caption(
        "Each fermion-number sector has dimension C(M, NF)."
    )

    if not st.button("▶ Run Fermionic Engine", type="primary"):
        st.info("Set the parameters and tap Run Fermionic Engine.")
        return

    if wc_min >= wc_max:
        st.error("Minimum cold frequency must be smaller than maximum.")
        return

    if Th <= Tc:
        st.warning("Th should exceed Tc for the usual heat-engine setup.")

    if wc_max >= wh:
        st.warning(
            "Some cold frequencies are at or above ωh; inspect these points "
            "carefully when interpreting engine operation."
        )

    wc_values = np.linspace(wc_min, wc_max, wc_points)

    try:
        with st.spinner("Calculating all fermion-number sectors..."):
            all_work, summary_df = run_fermion_simulation(
                M, Tc, Th, gamma, wh, wc_values,
                tau_c, tau_h, Ncycles
            )

            st.success("Fermionic simulation completed.")

            st.subheader("1. Hilbert-space summary")
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("2. Work output versus frequency ratio")
            fig, ax = plt.subplots(figsize=(9, 5))

            for NF, work_values in all_work.items():
                ax.plot(
                    wc_values / wh,
                    work_values,
                    marker="o",
                    markersize=3,
                    label=f"{NF} spin-polarized fermion(s)",
                )

            ax.axhline(0, linestyle="--", color="black", linewidth=1)
            ax.set_xlabel(r"$\omega_c/\omega_h$")
            ax.set_ylabel(r"$W_{\mathrm{out}}$ (final cycle)")
            ax.set_title(f"Fermionic Quantum Otto Engine: M = {M}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("3. Numerical results")
            data = {
                "wc": wc_values,
                "wc/wh": wc_values / wh,
            }

            for NF, work_values in all_work.items():
                data[f"Work_NF_{NF}"] = work_values

            work_df = pd.DataFrame(data)
            st.dataframe(work_df, use_container_width=True)
            st.download_button(
                "Download work-output CSV",
                work_df.to_csv(index=False).encode("utf-8"),
                "fermionic_otto_work_output.csv",
                "text/csv",
            )

            st.subheader("4. Maximum work in the scanned range")
            best = []

            for NF, work_values in all_work.items():
                idx = int(np.argmax(work_values))
                best.append({
                    "NF": NF,
                    "Maximum final-cycle work": work_values[idx],
                    "At wc": wc_values[idx],
                    "At wc/wh": wc_values[idx] / wh,
                    "Positive-work points": int(np.sum(work_values > 0)),
                })

            st.dataframe(pd.DataFrame(best), use_container_width=True)
            st.caption(
                "Positive work indicates extracted work under the sign convention "
                "used in this model. A positive value alone does not establish the "
                "efficiency or full thermodynamic performance of the engine."
            )

    except Exception as exc:
        st.error("Fermionic simulation failed.")
        st.exception(exc)


# ============================================================
# 4. DISPLAY THE SELECTED SIMULATION
# ============================================================

if page == "Bosonic Lindblad Solver":
    show_bosonic_solver()
else:
    show_fermionic_engine()
