"""Quantum readiness experiment: QAOA on one vehicle's route.

Take one route from the city solution with 4 stops. Fix the depot as the start
and end, and encode "which stop is visited at position p" in 4 x 4 = 16 binary
variables (a standard TSP QUBO). Run QAOA at depth p = 1, 2, 3 on an exact
statevector simulator, then score it the way hardware would be scored: draw
shots and keep the best feasible tour. Compare with brute force over all
4! = 24 tours, which takes microseconds.

This is a readiness experiment, not a claim of advantage. The classical
routing solvers in this example handle 80 stops and 13 vehicles in seconds;
QAOA here handles one 4-stop route.

Conventions, checked in code:
  - cost unitary U_C(g) = exp(-i g H_C), H_C = diag(E(z)) with E the QUBO energy
  - mixer U_B(b) = exp(-i b sum_k X_k)
  - bit k of the basis index z is qubit k (Qiskit little-endian)
  - Qiskit parameters bound by name -> value dict, never by list order
"""

from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

SHOTS = 1000


def qubo_energies(D, A):
    """E(z) for all 2^(m*m) bitstrings; D is (m+1)x(m+1) with node 0 the depot."""
    m = D.shape[0] - 1
    nq = m * m
    z = np.arange(2**nq, dtype=np.int64)
    bits = ((z[:, None] >> np.arange(nq)) & 1).astype(np.float64)  # bit k -> qubit k
    X = bits.reshape(-1, m, m)  # X[:, i, p]: stop i+1 at position p
    E = np.zeros(len(z))
    E += X[:, :, 0] @ D[0, 1:]  # depot -> first
    E += X[:, :, m - 1] @ D[1:, 0]  # last -> depot
    Dss = D[1:, 1:].copy()
    np.fill_diagonal(Dss, 0.0)
    for p in range(m - 1):
        E += np.einsum("zi,ij,zj->z", X[:, :, p], Dss, X[:, :, p + 1])
    E += A * ((1 - X.sum(2)) ** 2).sum(1)  # each stop exactly once
    E += A * ((1 - X.sum(1)) ** 2).sum(1)  # each position exactly one stop
    feasible = ((X.sum(2) == 1).all(1)) & ((X.sum(1) == 1).all(1))
    return E, feasible, X


def tour_of(Xz):
    return [int(np.argmax(Xz[:, p])) + 1 for p in range(Xz.shape[1])]


def qaoa_state(E, nq, gammas, betas):
    psi = np.full(2**nq, 2 ** (-nq / 2), dtype=np.complex128)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * E)
        c, s = np.cos(b), -1j * np.sin(b)
        for k in range(nq):  # exp(-i b X) on qubit k: pairs differ in bit k
            v = psi.reshape(-1, 2, 2**k)
            a0 = v[:, 0, :].copy()
            a1 = v[:, 1, :]
            v[:, 0, :] = c * a0 + s * a1
            v[:, 1, :] = s * a0 + c * a1
    return psi


def qiskit_crosscheck(E, nq, gammas, betas, psi_np):
    """Same circuit in Qiskit, from the diagonal Hamiltonian, bound by name."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import DiagonalGate
    from qiskit.quantum_info import Statevector

    p = len(gammas)
    b = [Parameter(f"beta_{k}") for k in range(p)]
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for k in range(p):
        # DiagonalGate needs numbers, so build the cost layer from a bound value.
        qc.append(DiagonalGate(list(np.exp(-1j * gammas[k] * E))), range(nq))
        qc.rx(2 * b[k], range(nq))  # RX(2b) = exp(-i b X)
    bound = qc.assign_parameters({b[k]: float(betas[k]) for k in range(p)})
    psi_q = Statevector(bound).data
    return float(abs(np.vdot(psi_np, psi_q)))


def optimise(E, nq, p, init=None, restarts=8, seed=0):
    rng = np.random.default_rng(seed)
    Ec = E - E.mean()

    def f(th):
        psi = qaoa_state(Ec, nq, th[:p], th[p:])
        return float(np.real(np.vdot(psi, Ec * psi)))

    starts = [init] if init is not None else []
    starts += [np.concatenate([rng.uniform(0, 0.3, p), rng.uniform(0, np.pi / 2, p)])
               for _ in range(restarts)]
    best = None
    for s0 in starts:
        r = minimize(f, s0, method="COBYLA", options={"maxiter": 400, "rhobeg": 0.1})
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:p], best.x[p:], best.fun


def interp(v):
    """INTERP initialisation for depth p+1 from depth p (Zhou et al. 2020)."""
    p = len(v)
    return np.array([((i) / p) * (v[i - 1] if i > 0 else 0) + ((p - i) / p) * (v[i] if i < p else 0)
                     for i in range(p + 1)])


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    city = json.load(open("results/city.json"))
    routes = city["results"][city["winner"]]["routes"]
    route = next(r for r in routes if len(r) == 4) if any(len(r) == 4 for r in routes) else sorted(routes, key=len)[0][:4]
    dist = np.asarray(city["dist"])  # directed road distances, metres
    nodes = [0, *route]
    D = dist[np.ix_(nodes, nodes)].astype(float)
    scale = D.max()
    Dn = D / scale
    m = len(route)
    nq = m * m
    A = 2.0 * Dn.max() * m  # penalty dominates any tour-length difference

    E, feasible, X = qubo_energies(Dn, A)
    # Brute force over tours: the classical anchor.
    t0 = time.perf_counter()
    tours = {perm: sum(D[a, b] for a, b in zip((0, *perm), (*perm, 0)))
             for perm in itertools.permutations(range(1, m + 1))}
    t_brute = time.perf_counter() - t0
    opt_perm = min(tours, key=tours.get)
    opt_len = tours[opt_perm]
    z_opt = [z for z in np.flatnonzero(feasible) if tuple(tour_of(X[z])) == opt_perm]
    assert len(z_opt) == 1
    assert abs(E[z_opt[0]] * scale - opt_len) < 1e-6 * opt_len  # penalty is zero on feasible z
    # QUBO optimum must be the brute-force optimum.
    assert int(np.argmin(E)) == z_opt[0], "QUBO ground state is not the optimal tour"

    rng = np.random.default_rng(11)
    rows, th = [], None
    for p in (1, 2, 3):
        t1 = time.perf_counter()
        init = None if th is None else np.concatenate([interp(th[0]), interp(th[1])])
        g, b, fval = optimise(E, nq, p, init=init, restarts=8 if p == 1 else 3)
        th = (g, b)
        psi = qaoa_state(E - E.mean(), nq, g, b)
        prob = np.abs(psi) ** 2
        overlap = qiskit_crosscheck(E - E.mean(), nq, g, b, psi)
        shots = rng.choice(len(prob), size=SHOTS, p=prob / prob.sum())
        feas_shots = [z for z in shots if feasible[z]]
        best_z = min(feas_shots, key=lambda z: E[z]) if feas_shots else None
        best_len = float(tours[tuple(tour_of(X[best_z]))]) if best_z is not None else None
        rows.append({
            "p": p,
            "p_optimal": float(prob[z_opt[0]]),
            "p_feasible": float(prob[feasible].sum()),
            "shots": SHOTS,
            "shots_optimal": int((shots == z_opt[0]).sum()),
            "shots_feasible": len(feas_shots),
            "best_tour_m_in_shots": best_len,
            "found_optimum": bool(best_len is not None and abs(best_len - opt_len) < 1e-9),
            "qiskit_overlap": round(overlap, 10),
            "gammas": g.tolist(), "betas": b.tolist(),
            "optimise_s": round(time.perf_counter() - t1, 1),
        })
        print(rows[-1], flush=True)

    out = {
        "route_stops": route, "qubits": nq, "tours": len(tours),
        "optimal_tour": list(opt_perm), "optimal_m": float(opt_len),
        "worst_tour_m": float(max(tours.values())),
        "brute_force_s": t_brute,
        "p_optimal_uniform_bitstring": 1 / 2**nq,
        "p_optimal_random_tour": 1 / len(tours),
        "penalty_A": A, "distance_scale_m": float(scale),
        "rows": rows,
        "convention": "U_C=exp(-i g diag(E)), U_B=exp(-i b sum X); qubit k = bit k; Qiskit bound by name",
    }
    with open("results/qaoa.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
