# Warm-start QAOA on qBraid

Follow-along material for the IBM x qBraid webinar on quantum optimization (October 7, 2026). It runs IBM's
[warm-start QAOA tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/warm-start-qaoa) on qBraid, adds an exact
classical baseline, and trains the circuit classically so a hardware run costs a single job.

Adapted from the tutorial in [Qiskit/documentation](https://github.com/Qiskit/documentation/blob/main/docs/tutorials/warm-start-qaoa.ipynb)
(© IBM, Apache-2.0). Method: D. J. Egger, J. Mareček and S. Woerner, *Warm-starting quantum optimization*, Quantum 5, 479 (2021).

## Quickstart (3 steps)

1. **Install the environment** (kernel included):
   ```
   qbraid envs install qiskit_zngl3z
   ```
   It ships Qiskit 2.5.2, Qiskit Runtime 0.50.0, Aer 0.17.2, qiskit-addon-opt-mapper 0.1.0 and HiGHS 1.15.1.
2. **Add your IBM Quantum API key** in the qBraid **Vault** extension (see [`../IBM_SETUP.md`](../IBM_SETUP.md)).
   Only needed for real hardware; simulator mode needs no IBM account.
3. **Open `warm_start_qaoa.ipynb`**, pick the kernel *Python 3 [Qiskit Optimization: warm-start QAOA]*, and run all cells.

## Simulator or hardware

The first code cell has the switches:

| Setting | Default | Meaning |
|---|---|---|
| `RUN_ON_HARDWARE` | `False` | `False`: local noisy simulator built from IBM's `FakeFez` calibration snapshot (free). `True`: one Sampler job on an IBM device |
| `BACKEND_NAME` | `None` | A device such as `"ibm_fez"`; `None` picks the least busy device with 127+ qubits |
| `N_LARGE` | 16 (simulator) / 40 (hardware) | Nodes in the 3-regular max-cut graph; 40 matches the tutorial |
| `SHOTS` | 4096 | Shots for the final sampling job |

Hardware runs go straight to IBM with your own key and are **billed to your IBM Quantum plan**, not to qBraid credits.
The tutorial's own hardware section optimizes inside a `Session` with ~150 iterations. This notebook trains the angles
classically and submits **one job in job mode**, which uses far less QPU time and works on plans without session mode.

## Expected runtimes (qBraid subscription pod, 4 vCPU, simulator mode)

| Step | Time |
|---|---|
| Whole notebook, top to bottom | about 2 minutes |
| Exact optimum (HiGHS MILP), 16 nodes | 0.02 s |
| Simulated annealing (tutorial's reference), 16 nodes | 0.3 s |
| Classical training of p = 1 angles (light cone, exact) | about 1 s |
| Noisy sampling of the routed 16-qubit circuit on `FakeFez` | about 2 minutes |

## What to compare against

Report every QAOA number next to these, on the same graph:

1. **The exact optimum** from the HiGHS MILP (the notebook prints it). The tutorial's simulated-annealing reference can
   fall short of it: on the tutorial's own 40-node graph (seed 0), SA finds 53 while the optimum is 54, so a "ratio vs SA"
   above 1 can still be below the optimum.
2. **The rounded warm start**, with no quantum circuit at all.

**Read this before presenting results.** The max-cut QUBO has no squared terms, so its box relaxation always has an
optimum at a corner: the multi-start relaxation returns a *bitstring* (exactly binary on the 16-node example and on the
tutorial's 40-node instance, where rounding it gives the optimum, 54). With a binary warm start and ε = 0.25, the best
p = 1 angles are γ = 0, β = π/2, which rotate every qubit exactly onto the warm-start bit. An exact grid search over
(γ, β) confirmed this on both instances and on deliberately weak warm starts. So at p = 1 the circuit reproduces its
classical input, and a hardware run measures how faithfully the device returns that bitstring under noise (P(optimum)
= 0.75 on the `FakeFez` noise model at 16 nodes), not an optimization gain. The notebook's last cell checks this on
your instance. Use p ≥ 2, a non-binary warm start (such as the Goemans-Williamson SDP relaxation), or constrained
problems to give the quantum layer room to help, and train those on qBraid GPUs.

## Files

- `warm_start_qaoa.ipynb`: the notebook, executed in simulator mode.
- `make_notebook.py`: generates the notebook (edit this, then re-run it).
