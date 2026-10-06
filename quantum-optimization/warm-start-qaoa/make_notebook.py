"""Build warm_start_qaoa.ipynb (adapted from IBM's warm-start QAOA tutorial, Apache-2.0)."""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# Warm-start QAOA on qBraid

Follow-along notebook for the **IBM x qBraid webinar on quantum optimization** (October 7, 2026).

**Source and licence.** Adapted from IBM Quantum's tutorial
[*Warm-start QAOA with the Optimization Mapper Qiskit addon*](https://quantum.cloud.ibm.com/docs/en/tutorials/warm-start-qaoa)
([source](https://github.com/Qiskit/documentation/blob/main/docs/tutorials/warm-start-qaoa.ipynb)),
© IBM, licensed under the [Apache License 2.0](https://github.com/Qiskit/documentation/blob/main/LICENSE).
The method is from D. J. Egger, J. Mareček and S. Woerner, *Warm-starting quantum optimization*, Quantum 5, 479 (2021).

**What this version changes**
- Runs out of the box on qBraid: a credential check up front, and a `RUN_ON_HARDWARE` switch. The default runs on a local
  noisy simulator of an IBM Heron device (`FakeFez`), so nothing is billed.
- Adds an **exact classical optimum** (mixed-integer program solved with HiGHS) next to the tutorial's simulated-annealing
  reference, so every QAOA number is scored against the true best cut.
- Trains the p = 1 warm-start angles **exactly on classical compute** (light-cone reduction, verified against a full statevector),
  so the hardware run is **one sampling job** instead of a hardware-in-the-loop optimization.

**Cost.** IBM Quantum runs are billed to *your* IBM Quantum plan, not to qBraid credits. Simulator mode is free.
The tutorial's own hardware section runs ~150 optimizer iterations inside a `Session`; this notebook's default hardware path
submits a single job, which uses far less QPU time and also works on plans that do not offer session mode.
"""))

cells.append(md("## 0. Environment and IBM credentials\n\nSet the switches, then run all cells. Nothing here prints your token."))
cells.append(code(r"""# ---- switches -------------------------------------------------------------
RUN_ON_HARDWARE = False      # True: run the large example on a real IBM backend (billed to your IBM plan)
BACKEND_NAME = None          # e.g. "ibm_fez"; None = least busy operational device with >= 127 qubits
N_LARGE = 40 if RUN_ON_HARDWARE else 16   # nodes in the large example (16 keeps local noisy simulation fast)
SHOTS = 4096
# ----------------------------------------------------------------------------

import os, importlib.metadata as md_
for pkg in ["qiskit", "qiskit-ibm-runtime", "qiskit-aer", "qiskit-addon-opt-mapper", "highspy", "scipy", "networkx"]:
    try:
        print(f"{pkg:<26} {md_.version(pkg)}")
    except md_.PackageNotFoundError:
        print(f"{pkg:<26} MISSING -- install the qBraid environment 'qiskit-optimization'")

def ibm_credentials_available():
    # Names only: never print or log a token value.
    env_names = sorted(k for k in os.environ if k.startswith("QISKIT_IBM_") or k in ("IBM_QUANTUM_TOKEN", "IBM_CLOUD_API_KEY"))
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        saved = sorted(QiskitRuntimeService.saved_accounts().keys())
    except Exception:
        saved = []
    return env_names, saved

env_names, saved_accounts = ibm_credentials_available()
print("\nIBM credential environment variables found:", env_names or "none")
print("Saved Qiskit Runtime accounts:", saved_accounts or "none")
if RUN_ON_HARDWARE and not (env_names or saved_accounts):
    raise RuntimeError("RUN_ON_HARDWARE=True but no IBM credentials found. Add your IBM Quantum API key in the "
                       "qBraid Vault (see IBM_SETUP.md), restart the kernel, and run again.")
print("\nMode:", "IBM hardware" if RUN_ON_HARDWARE else "local noisy simulator (FakeFez), free")"""))

cells.append(md("## Setup\n\nImports and helpers, as in the tutorial."))
cells.append(code(r"""import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize, milp, LinearConstraint, Bounds

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    EstimatorOptions,
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
)

from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import to_ising

import warnings
# qiskit-ibm-runtime 0.50 deprecates SamplerV2 in favour of a client-side Sampler; the tutorial still uses SamplerV2.
warnings.filterwarnings("ignore", message=".*SamplerV2 class is deprecated.*")"""))

# ---------------- small example (tutorial, lightly trimmed) ----------------
cells.append(md(r"""# Part 1 — Small example on a statevector simulator

A four-node weighted max-cut problem, exactly as in the tutorial. Max-cut asks for a partition of the vertices that
maximizes the total weight of edges crossing it. As a QUBO minimization:
$$\min_{x \in \{0,1\}^n} -\sum_{(i,j) \in E} w_{ij}(x_i + x_j - 2x_i x_j)$$"""))
cells.append(code(r"""n_nodes = 4
edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]
G = nx.Graph()
G.add_nodes_from(range(n_nodes))
G.add_weighted_edges_from(edges)
pos = nx.spring_layout(G, seed=42)
edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
fig, ax = plt.subplots(figsize=(4, 3))
nx.draw(G, pos, with_labels=True, node_color="lightblue", ax=ax)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
ax.set_title("Max-Cut graph")
plt.tight_layout(); plt.show()"""))
cells.append(md("### Step 1: Map the problem to a QUBO and an Ising Hamiltonian, and solve the continuous relaxation"))
cells.append(code(r"""maxcut = Maxcut(G)
prob = maxcut.to_optimization_problem()
print(prob.prettyprint())

qubo = OptimizationProblemToQubo().convert(prob)
cost_operator, offset = to_ising(qubo)
n_qubits = cost_operator.num_qubits
print(f"Cost Hamiltonian H_C ({n_qubits} qubits):\n{cost_operator}\nOffset: {offset}")"""))
cells.append(code(r"""def qp_relaxation(qubo, n, starts=200, seed=42):
    # Multi-start L-BFGS-B on the continuous relaxation x in [0,1]^n (non-convex for max-cut).
    Q = qubo.objective.quadratic.to_array(symmetric=True)
    mu = qubo.objective.linear.to_array()
    const = qubo.objective.constant
    f = lambda x: x @ Q @ x + mu @ x + const
    rng = np.random.default_rng(seed)
    best_val, best_x = np.inf, None
    for _ in range(starts):
        r = minimize(f, rng.uniform(0, 1, n), method="L-BFGS-B", bounds=[(0, 1)] * n)
        if r.fun < best_val:
            best_val, best_x = r.fun, r.x
    return best_x, best_val

c_star, qp_val = qp_relaxation(qubo, n_qubits)
print(f"QP relaxation c* = {np.round(c_star, 4)}   objective = {qp_val:.4f}")"""))
cells.append(md(r"""### Step 2: Build standard QAOA and warm-start QAOA circuits

WS-QAOA prepares each qubit in $R_Y(\theta_i)|0\rangle$ with $\theta_i = 2\arcsin\sqrt{c^*_i}$ and uses the mixer
$R_Y(\theta_i) R_Z(-2\beta) R_Y(-\theta_i)$. $c^*$ is clipped to $[\varepsilon, 1-\varepsilon]$ with $\varepsilon = 0.25$."""))
cells.append(code(r"""p = 1
epsilon = 0.25
thetas = 2 * np.arcsin(np.sqrt(np.clip(c_star, epsilon, 1 - epsilon)))
print(f"Warm-start angles theta = {np.round(thetas, 4)} rad")


def apply_cost_unitary(qc, cost_op, gamma, only_terms=None):
    # exp(-i gamma H_C); optionally restricted to a subset of term indices (used by the light-cone trainer).
    for t, (pauli_term, coeff) in enumerate(zip(cost_op.paulis, cost_op.coeffs)):
        if only_terms is not None and t not in only_terms:
            continue
        idx = [j for j, q in enumerate(pauli_term.to_label()[::-1]) if q == "Z"]
        if len(idx) == 1:
            qc.rz(2 * gamma * coeff.real, idx[0])
        elif len(idx) == 2:
            qc.cx(idx[0], idx[1]); qc.rz(2 * gamma * coeff.real, idx[1]); qc.cx(idx[0], idx[1])


def build_ws_qaoa(cost_op, n_layers, n_qubits, thetas):
    gammas = ParameterVector("γ", n_layers)
    betas = ParameterVector("β", n_layers)
    qc = QuantumCircuit(n_qubits)
    for i, theta in enumerate(thetas):
        qc.ry(theta, i)
    for k in range(n_layers):
        apply_cost_unitary(qc, cost_op, gammas[k])
        for i, theta in enumerate(thetas):
            qc.ry(theta, i); qc.rz(-2 * betas[k], i); qc.ry(-theta, i)
    return qc, gammas, betas


std_qc = qaoa_ansatz(cost_operator, reps=p)
ws_qc, ws_gammas, ws_betas = build_ws_qaoa(cost_operator, p, n_qubits, thetas)
ws_qc.draw("mpl", fold=-1)"""))
cells.append(md("### Step 3: Optimize both on an exact statevector estimator"))
cells.append(code(r"""estimator = StatevectorEstimator()


def make_cost_fn(circuit, param_order, cost_op, estimator, history):
    def cost_fn(params):
        bound = circuit.assign_parameters(dict(zip(param_order, params)))   # dict binding: order-safe
        e = estimator.run([(bound, cost_op)]).result()[0].data.evs.real
        history.append(e)
        return e
    return cost_fn


np.random.seed(42)
std_param_order = list(std_qc.parameters)
std_history = []
std_result = minimize(make_cost_fn(std_qc, std_param_order, cost_operator, estimator, std_history),
                      np.random.uniform(0, np.pi, len(std_param_order)), method="COBYLA",
                      options={"maxiter": 300, "rhobeg": 0.5})

ws_param_order = list(ws_gammas) + list(ws_betas)
ws_history = []
ws_result = minimize(make_cost_fn(ws_qc, ws_param_order, cost_operator, estimator, ws_history),
                     np.concatenate([np.zeros(p), np.full(p, np.pi / 4)]), method="COBYLA",
                     options={"maxiter": 300, "rhobeg": 0.5})

all_energies = [Statevector.from_label(format(k, f"0{n_qubits}b")).expectation_value(cost_operator).real
                for k in range(2**n_qubits)]
optimal_energy = min(all_energies)
print(f"Exact optimal energy : {optimal_energy:.4f}")
print(f"Standard QAOA energy : {std_result.fun:.4f}   ratio {std_result.fun/optimal_energy:.4f}   calls {len(std_history)}")
print(f"WS-QAOA energy       : {ws_result.fun:.4f}   ratio {ws_result.fun/optimal_energy:.4f}   calls {len(ws_history)}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(std_history, label="Standard QAOA"); ax.plot(ws_history, label="WS-QAOA")
ax.axhline(optimal_energy, color="k", ls="--", label=f"Exact optimum ({optimal_energy:.2f})")
ax.set_xlabel("Optimizer call"); ax.set_ylabel(r"$\langle H_C \rangle$"); ax.legend(); plt.tight_layout(); plt.show()"""))
cells.append(md("### Step 4: Sample and decode"))
cells.append(code(r"""def evaluate_cut(x, G):
    # x: sequence of 0/1 indexed by node
    return sum(w for u, v, w in G.edges.data("weight", default=1) if x[u] != x[v])


def counts_to_cut_distribution(counts, G):
    # Qiskit bitstrings have qubit 0 rightmost; reverse to index by node.
    total = sum(counts.values()); dist = {}
    for bs, c in counts.items():
        cut = evaluate_cut([int(b) for b in bs[::-1]], G)
        dist[cut] = dist.get(cut, 0) + c / total
    return dist


sampler = StatevectorSampler()
def sample(circuit, order, params, shots=1024):
    b = circuit.assign_parameters(dict(zip(order, params))); b.measure_all()
    return sampler.run([b], shots=shots).result()[0].data.meas.get_counts()

optimal_cut = -(optimal_energy + offset)
for name, qc_, order, res in [("Standard QAOA", std_qc, std_param_order, std_result),
                              ("WS-QAOA", ws_qc, ws_param_order, ws_result)]:
    dist = counts_to_cut_distribution(sample(qc_, order, res.x), G)
    print(f"{name:<14} P(optimal cut = {optimal_cut:g}) = {dist.get(optimal_cut, 0):.3f}")"""))

# ---------------- large example ----------------
cells.append(md(r"""# Part 2 — Larger problem: exact classical answer first, then one quantum job

A random 3-regular graph (the tutorial's large example uses 40 nodes on hardware). In simulator mode this notebook uses
`N_LARGE = 16` so the noisy simulation of the routed circuit finishes in a couple of minutes on a laptop-class CPU.

**Classical anchors.** The tutorial compares against simulated annealing. We also solve the max-cut *exactly* as a
mixed-integer program with HiGHS (milliseconds at this size), so the approximation ratio below is against the true optimum."""))
cells.append(code(r"""G_large = nx.random_regular_graph(d=3, n=N_LARGE, seed=0)
prob_large = Maxcut(G_large).to_optimization_problem()
qubo_large = OptimizationProblemToQubo().convert(prob_large)
cost_op_large, offset_large = to_ising(qubo_large)
n_large = cost_op_large.num_qubits
print(f"Graph: {N_LARGE} nodes, {G_large.number_of_edges()} edges; cost operator: {n_large} qubits, {len(cost_op_large)} terms")


def exact_maxcut_highs(G):
    # max sum_e y_e  s.t.  y_e <= x_u + x_v,  y_e <= 2 - x_u - x_v,  x, y binary.  Solved by HiGHS via scipy.milp.
    nodes, E = list(G.nodes()), list(G.edges())
    n, m = len(nodes), len(E)
    c = np.concatenate([np.zeros(n), -np.ones(m)])
    A, lb, ub = [], [], []
    for k, (u, v) in enumerate(E):
        r1 = np.zeros(n + m); r1[n + k] = 1; r1[u] = -1; r1[v] = -1; A.append(r1); lb.append(-np.inf); ub.append(0)
        r2 = np.zeros(n + m); r2[n + k] = 1; r2[u] = 1; r2[v] = 1;  A.append(r2); lb.append(-np.inf); ub.append(2)
    res = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.ones(n + m), bounds=Bounds(0, 1))
    x = np.round(res.x[:n]).astype(int)
    return evaluate_cut(x, G), x


def simulated_annealing_maxcut(G, seed=0, T0=2.0, T_min=1e-4, alpha=0.995, n_steps=100_000):
    rng = np.random.default_rng(seed); n = G.number_of_nodes()
    x = rng.integers(0, 2, n); best_x = x.copy(); best = evaluate_cut(x, G); T = T0
    for _ in range(n_steps):
        i = rng.integers(0, n)
        delta = sum((-1 if x[i] != x[nb] else 1) for nb in G.neighbors(i))
        if delta > 0 or rng.random() < np.exp(delta / T):
            x[i] ^= 1
            cut = evaluate_cut(x, G)
            if cut > best:
                best, best_x = cut, x.copy()
        T = max(T * alpha, T_min)
    return best, best_x

t0 = time.perf_counter(); exact_cut, exact_x = exact_maxcut_highs(G_large); t_exact = time.perf_counter() - t0
t0 = time.perf_counter(); sa_cut, sa_x = simulated_annealing_maxcut(G_large); t_sa = time.perf_counter() - t0
print(f"Exact optimum (HiGHS MILP): cut = {exact_cut}   ({t_exact:.2f} s)")
print(f"Simulated annealing       : cut = {sa_cut}   ({t_sa:.2f} s)")

c_star_large, _ = qp_relaxation(qubo_large, n_large)
thetas_large = 2 * np.arcsin(np.sqrt(np.clip(c_star_large, epsilon, 1 - epsilon)))
ws_qc_large, g_large, b_large = build_ws_qaoa(cost_op_large, 1, n_large, thetas_large)
print(f"Warm start: c* in [{c_star_large.min():.3f}, {c_star_large.max():.3f}]")"""))

cells.append(md(r"""## Train the p = 1 angles exactly on classical compute

For p = 1, the expectation of each term $Z_iZ_j$ depends only on qubits $i$, $j$ and their graph neighbours (its *light cone*):
the cost terms that do not touch $i$ or $j$ commute through and cancel, and the warm-start state is a product state. On a
3-regular graph that is at most 6 qubits per term, so $\langle H_C\rangle$ is computed **exactly** for any number of nodes.
We check this against a full statevector on the 4-node example, then optimize $(\gamma, \beta)$ with COBYLA.

The tutorial instead optimizes on the QPU (about 150 hardware jobs). Training classically means the QPU only runs the
final sampling job. For p ≥ 2 the light cones grow quickly; that is where qBraid's on-demand GPUs (CUDA-Q or Aer GPU
statevector) come in."""))
cells.append(code(r"""def term_supports(cost_op):
    sup = []
    for pt in cost_op.paulis:
        sup.append(tuple(j for j, q in enumerate(pt.to_label()[::-1]) if q == "Z"))
    return sup


def lightcone_energy(cost_op, thetas, gamma, beta):
    # Exact p=1 <H_C> for a product warm-start state, using per-term light cones.
    sup = term_supports(cost_op)
    coeffs = cost_op.coeffs.real
    total = 0.0
    for t, S in enumerate(sup):
        if not S:
            total += coeffs[t]; continue
        touching = [k for k, T in enumerate(sup) if T and set(T) & set(S)]
        L = sorted(set(S).union(*[set(sup[k]) for k in touching]))
        loc = {q: i for i, q in enumerate(L)}
        qc = QuantumCircuit(len(L))
        for q in L:
            qc.ry(thetas[q], loc[q])
        for k in touching:
            idx = [loc[q] for q in sup[k]]
            if len(idx) == 1:
                qc.rz(2 * gamma * coeffs[k], idx[0])
            else:
                qc.cx(idx[0], idx[1]); qc.rz(2 * gamma * coeffs[k], idx[1]); qc.cx(idx[0], idx[1])
        for q in S:
            qc.ry(thetas[q], loc[q]); qc.rz(-2 * beta, loc[q]); qc.ry(-thetas[q], loc[q])
        label = ["I"] * len(L)
        for q in S:
            label[len(L) - 1 - loc[q]] = "Z"
        total += coeffs[t] * Statevector(qc).expectation_value(SparsePauliOp("".join(label))).real
    return total


# Verify against a full statevector on the 4-node example at random angles.
rng_chk = np.random.default_rng(7)
for _ in range(3):
    gam, bet = rng_chk.uniform(-np.pi, np.pi, 2)
    full = Statevector(ws_qc.assign_parameters({ws_gammas[0]: gam, ws_betas[0]: bet})).expectation_value(cost_operator).real
    lc = lightcone_energy(cost_operator, thetas, gam, bet)
    assert abs(full - lc) < 1e-9, (full, lc)
print("Light-cone energy matches the full statevector to < 1e-9 on the 4-node example.")

t0 = time.perf_counter()
train_hist = []
def lc_cost(x):
    e = lightcone_energy(cost_op_large, thetas_large, x[0], x[1]); train_hist.append(e); return e
train = minimize(lc_cost, [0.0, np.pi / 4], method="COBYLA", options={"maxiter": 120, "rhobeg": 0.3})
gamma_opt, beta_opt = train.x
t_train = time.perf_counter() - t0
print(f"Trained p=1 angles: gamma={gamma_opt:.4f}, beta={beta_opt:.4f}   <H_C> = {train.fun:.4f}   "
      f"({len(train_hist)} evaluations, {t_train:.1f} s on CPU)")

if n_large <= 20:   # cross-check the trained energy with a full statevector where that is cheap
    full_large = Statevector(ws_qc_large.assign_parameters({g_large[0]: gamma_opt, b_large[0]: beta_opt})
                             ).expectation_value(cost_op_large).real
    print(f"Full-statevector check at the trained angles: {full_large:.6f}")"""))

cells.append(md(r"""## Run the trained circuit: one job

`RUN_ON_HARDWARE = False` samples on a local noisy simulator built from IBM's `FakeFez` calibration snapshot
(Qiskit Runtime local testing mode). `True` sends a single Sampler job to an IBM backend through your IBM account."""))
cells.append(code(r"""if RUN_ON_HARDWARE:
    service = QiskitRuntimeService()
    backend = (service.backend(BACKEND_NAME) if BACKEND_NAME
               else service.least_busy(operational=True, simulator=False, min_num_qubits=127))
else:
    from qiskit_ibm_runtime.fake_provider import FakeFez
    backend = FakeFez()
print("Backend:", backend.name)

circ = ws_qc_large.assign_parameters({g_large[0]: gamma_opt, b_large[0]: beta_opt})
circ.measure_all()
pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
isa = pm.run(circ)
print(f"Transpiled: {isa.num_qubits} device qubits, 2-qubit depth {isa.depth(lambda x: x.operation.num_qubits == 2)}, "
      f"ops {dict(isa.count_ops())}")

t0 = time.perf_counter()
sampler_run = Sampler(mode=backend)                     # job mode: one job, works on every IBM plan
sampler_run.options.environment.job_tags = ["QBRAID_WSQAOA"]
job = sampler_run.run([isa], shots=SHOTS)
print("Job id:", job.job_id())
counts = job.result()[0].data.meas.get_counts()
t_run = time.perf_counter() - t0
print(f"Sampling took {t_run:.1f} s")"""))
cells.append(code(r"""dist = counts_to_cut_distribution(counts, G_large)
best_sampled = max(dist)
mode_bs = max(counts, key=counts.get)
mode_cut = evaluate_cut([int(b) for b in mode_bs[::-1]], G_large)
mean_cut = sum(c * p_ for c, p_ in dist.items())

print("=== Result vs classical ===")
print(f"{'Method':<34}{'Cut':>8}{'Ratio vs exact':>18}")
print(f"{'Exact optimum (HiGHS MILP)':<34}{exact_cut:>8}{1.0:>18.4f}")
print(f"{'Simulated annealing':<34}{sa_cut:>8}{sa_cut/exact_cut:>18.4f}")
print(f"{'WS-QAOA best sampled bitstring':<34}{best_sampled:>8}{best_sampled/exact_cut:>18.4f}")
print(f"{'WS-QAOA most frequent bitstring':<34}{mode_cut:>8}{mode_cut/exact_cut:>18.4f}")
print(f"{'WS-QAOA mean sampled cut':<34}{mean_cut:>8.2f}{mean_cut/exact_cut:>18.4f}")
print(f"P(optimal cut) = {dist.get(exact_cut, 0):.4f} over {SHOTS} shots")

cuts = sorted(dist)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar([str(c) for c in cuts], [dist[c] for c in cuts], color="salmon")
ax.axvline(cuts.index(exact_cut) if exact_cut in cuts else len(cuts) - 0.5, color="k", ls="--", alpha=0.5,
           label=f"exact optimum = {exact_cut}")
ax.set_xlabel("Cut value"); ax.set_ylabel("Probability"); ax.legend()
ax.set_title(f"WS-QAOA samples on {backend.name} ({N_LARGE} nodes, p=1)")
plt.tight_layout(); plt.show()"""))

cells.append(md(r"""## How to read this honestly

**Check the warm start before crediting the circuit.** The max-cut QUBO has no $x_i^2$ terms, so its box relaxation is
multilinear and always has an optimum at a corner of $[0,1]^n$. The multi-start relaxation therefore returns a *bitstring*
(here and on the tutorial's own 40-node instance, $c^*$ is exactly binary): it is a classical local-search answer.

With a binary $c^*$ and $\varepsilon = 0.25$, the best p = 1 angles are $\gamma = 0$, $\beta = \pi/2$. At those angles each
qubit is rotated exactly onto the warm-start bit ($R_Y(3\theta)|0\rangle$ with $\theta \in \{\pi/3, 2\pi/3\}$), so the
noiseless circuit returns the classical warm-start bitstring with certainty and the cost layer contributes nothing. On
hardware, noise can only lower the probability of sampling it. We checked this with an exact grid search over
$(\gamma, \beta)$ on the 16-node example, on the tutorial's 40-node instance, and on deliberately weak warm starts
(cut 18 of an optimum 21): the optimum is always $\gamma = 0$.

So in this notebook the hardware run measures **how faithfully a QPU reproduces a known bitstring under noise**, a useful
device benchmark, but not an optimization gain. The cell below checks it on your instance.

To give the quantum layer something to do: use p ≥ 2, a non-binary warm start (for example, the Goemans-Williamson SDP
relaxation discussed in the warm-start paper), or problems with constraints. Train those with qBraid's on-demand GPUs,
and always report the exact optimum and the rounded warm start alongside.
"""))
cells.append(code(r"""x_round = (c_star_large > 0.5).astype(int)
round_cut = evaluate_cut(x_round, G_large)
print(f"c* binary: {np.allclose(c_star_large, np.round(c_star_large), atol=1e-6)}")
print(f"Rounded warm start (no quantum): cut = {round_cut}   exact optimum = {exact_cut}")
print(f"Trained angles: gamma = {gamma_opt:.4f}, beta = {beta_opt:.4f}")
warm_bits_energy = -(round_cut) - offset_large      # Ising energy of the warm-start bitstring
print(f"<H_C> at trained angles = {train.fun:.4f};  Ising energy of the warm-start bitstring = {warm_bits_energy:.4f}")
print(f"P(sampling the optimum) on {backend.name}: {dist.get(exact_cut, 0):.4f}  (1.0 would mean noise-free reproduction)")"""))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("warm_start_qaoa.ipynb")
nbf.write(nb, out)
print("wrote", out)
