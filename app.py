
# Paste your COMPLETE Bosonic Lindblad Solver
# Python program below this line.
!pip install qutip streamlit
import math
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qutip as qt
import streamlit as st


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bosonic Lindblad Solver",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Bosonic Lindblad Solver")
st.write(
    "Simulate fixed-particle-number bosonic systems "
    "interacting with a thermal bath."
)

st.info(
    "Model: identical bosons, equally spaced energy levels, "
    "adjacent-level thermal transitions, and a Markovian bath. "
    "Set kB = hbar = 1."
)


# --------------------------------------------------
# 2. BUILD THE FIXED-N BOSONIC BASIS
# --------------------------------------------------

def make_basis(N, M):
    """Occupation-number states with exactly N bosons in M modes."""
    return [
        tuple(occ)
        for occ in product(range(N + 1), repeat=M)
        if sum(occ) == N
    ]


def build_hamiltonian(basis, omega):
    """
    H = sum_i (i * omega) * n_i.
    The zero of energy is the lowest single-particle level.
    """
    energies = [
        sum(i * omega * n for i, n in enumerate(occ))
        for occ in basis
    ]
    return qt.Qobj(np.diag(energies)), np.array(energies)


# --------------------------------------------------
# 3. BUILD PHYSICAL BOSONIC TRANSITION OPERATORS
# --------------------------------------------------

def build_transitions(basis, M):
    """
    A_i = a_i^dagger a_(i+1).

    Matrix elements:
    sqrt((n_i + 1) * n_(i+1))

    A_i moves one boson down by one energy level.
    Its adjoint moves one boson up.
    """
    index = {occ: j for j, occ in enumerate(basis)}
    dim = len(basis)
    operators = []

    for i in range(M - 1):
        matrix = np.zeros((dim, dim), dtype=complex)

        for col, occ in enumerate(basis):
            ni = occ[i]
            nj = occ[i + 1]

            if nj == 0:
                continue

            new_occ = list(occ)
            new_occ[i] += 1
            new_occ[i + 1] -= 1
            new_occ = tuple(new_occ)

            row = index[new_occ]
            matrix[row, col] = math.sqrt((ni + 1) * nj)

        operators.append(qt.Qobj(matrix))

    return operators


# --------------------------------------------------
# 4. THERMAL BATH AND LINDBLAD EQUATION
# --------------------------------------------------

def thermal_occupation(omega, temperature):
    """Bose-Einstein occupation, with kB = hbar = 1."""
    if temperature <= 0:
        return 0.0

    x = omega / temperature

    if x > 700:
        return 0.0

    return 1.0 / np.expm1(x)


def solve_system(N, M, omega, temperature, gamma, t_final, num_times):
    basis = make_basis(N, M)
    dim = len(basis)

    H, energies = build_hamiltonian(basis, omega)
    transitions = build_transitions(basis, M)

    nbar = thermal_occupation(omega, temperature)
    c_ops = []

    for A in transitions:
        # Downward transitions: emission into the bath
        c_ops.append(
            np.sqrt(gamma * (nbar + 1.0)) * A
        )

        # Upward transitions: absorption from the bath
        if nbar > 0:
            c_ops.append(
                np.sqrt(gamma * nbar) * A.dag()
            )

    # Start in the ground configuration:
    # all N bosons occupy the lowest level.
    ground_occ = (N,) + (0,) * (M - 1)
    ground_index = basis.index(ground_occ)

    psi0 = qt.basis(dim, ground_index)
    rho0 = qt.ket2dm(psi0)

    times = np.linspace(0.0, t_final, num_times)

    result = qt.mesolve(
        H,
        rho0,
        times,
        c_ops=c_ops,
        e_ops=[],
    )

    return basis, energies, times, result, nbar


# --------------------------------------------------
# 5. USER INPUTS
# --------------------------------------------------

st.sidebar.header("Simulation parameters")

N = st.sidebar.number_input(
    "Number of bosons (N)",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
)

M = st.sidebar.number_input(
    "Number of energy levels (M)",
    min_value=2,
    max_value=8,
    value=3,
    step=1,
)

omega = st.sidebar.number_input(
    "Energy spacing (ω)",
    min_value=0.01,
    max_value=100.0,
    value=1.0,
    step=0.1,
)

temperature = st.sidebar.number_input(
    "Bath temperature (T)",
    min_value=0.0,
    max_value=100.0,
    value=3.0,
    step=0.1,
)

gamma = st.sidebar.number_input(
    "Bath coupling strength (γ)",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1,
)

t_final = st.sidebar.number_input(
    "Final simulation time",
    min_value=0.1,
    max_value=1000.0,
    value=10.0,
    step=1.0,
)

num_times = st.sidebar.number_input(
    "Number of time points",
    min_value=11,
    max_value=501,
    value=101,
    step=10,
)

dim = math.comb(N + M - 1, N)

st.sidebar.caption(f"Hilbert-space dimension: {dim}")

# Protect a phone or small server from oversized calculations.
if dim > 200:
    st.warning(
        "This system has too many basis states for this demo. "
        "Reduce N or M until the dimension is at most 200."
    )
    st.stop()

run = st.sidebar.button(
    "▶ Run Simulation",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# 6. RUN SIMULATION AND SHOW RESULTS
# --------------------------------------------------

if run:
    try:
        with st.spinner("Solving the Lindblad master equation..."):
            basis, energies, times, result, nbar = solve_system(
                int(N),
                int(M),
                float(omega),
                float(temperature),
                float(gamma),
                float(t_final),
                int(num_times),
            )

        states = result.states
        final_rho = states[-1]

        populations = np.array([
            np.real(np.diag(rho.full()))
            for rho in states
        ])

        st.success("Simulation completed.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Basis states", len(basis))
        col2.metric("Thermal occupation", f"{nbar:.5f}")
        col3.metric("Final trace", f"{np.real(final_rho.tr()):.8f}")

        # Population curves
        st.subheader("1. Fock-state populations")

        fig, ax = plt.subplots(figsize=(9, 5))

        # Plot up to eight states with the largest final populations.
        chosen = np.argsort(populations[-1])[-min(8, len(basis)):]

        for j in chosen:
            label = str(basis[j])
            ax.plot(times, populations[:, j], label=label)

        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Occupation state", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Each occupation tuple lists the boson count in levels "
            "0, 1, 2, ... respectively."
        )

        # Mean occupation of each single-particle level
        st.subheader("2. Mean occupation of each energy level")

        mean_occ = np.zeros((len(times), int(M)))

        for j, occ in enumerate(basis):
            for level in range(int(M)):
                mean_occ[:, level] += populations[:, j] * occ[level]

        fig, ax = plt.subplots(figsize=(9, 5))

        for level in range(int(M)):
            ax.plot(
                times,
                mean_occ[:, level],
                label=f"Level {level}",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Mean boson number")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Final density matrix
        st.subheader("3. Final density matrix")

        labels = [str(s) for s in basis]
        rho_array = final_rho.full()

        rho_df = pd.DataFrame(
            rho_array,
            index=labels,
            columns=labels,
        )

        st.write("Rows and columns correspond to occupation states.")
        st.dataframe(rho_df, use_container_width=True)

        st.download_button(
            "Download density matrix (CSV)",
            data=rho_df.to_csv().encode("utf-8"),
            file_name="final_density_matrix.csv",
            mime="text/csv",
        )

        # Final population table
        st.subheader("4. Final-state populations")

        pop_df = pd.DataFrame({
            "Occupation state": labels,
            "Energy": energies,
            "Final population": populations[-1],
        })

        st.dataframe(pop_df, use_container_width=True)

        st.download_button(
            "Download populations (CSV)",
            data=pop_df.to_csv(index=False).encode("utf-8"),
            file_name="final_populations.csv",
            mime="text/csv",
        )

        # Basic physical checks
        st.subheader("5. Physical checks")

        trace = complex(final_rho.tr())
        hermitian = final_rho.isherm
        hermiticity_error = np.linalg.norm(
            rho_array - rho_array.conj().T
        )

        st.write(f"Trace: {trace.real:.10f}")
        st.write(f"Hermitian: {hermitian}")
        st.write(f"Hermiticity error: {hermiticity_error:.3e}")
        st.write(
            "Total final population: "
            f"{populations[-1].sum():.10f}"
        )

        if abs(trace.imag) > 1e-8 or abs(trace.real - 1.0) > 1e-6:
            st.warning("The trace check did not pass the chosen tolerance.")

        if not hermitian or hermiticity_error > 1e-6:
            st.warning("The Hermiticity check did not pass.")

        if np.min(populations[-1]) < -1e-7:
            st.warning("A negative population was detected.")

    except Exception as exc:
        st.error("The simulation failed.")
        st.exception(exc)

else:
    st.subheader("Ready to simulate")
    st.write(
        "Choose your parameters in the sidebar, then tap "
        "**Run Simulation** to calculate the time evolution."
    )
    st.write(
        "Start with 2 bosons, 3 levels, T = 3, ω = 1, and γ = 1."
    )
