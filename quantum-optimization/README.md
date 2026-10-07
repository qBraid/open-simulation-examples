# Quantum optimization on qBraid

Everything needed to follow the IBM x qBraid webinar on quantum optimization
(October 7, 2026: Daniel Egger, IBM Research; Ibrahim Shehzad, IBM Quantum) and to
keep going afterwards: real IBM Quantum hardware, exact classical baselines, and the
QOBLIB benchmark library, in one Lab.

## Get running in 10 minutes

1. **Connect IBM Quantum** (5 min): [`IBM_SETUP.md`](IBM_SETUP.md). Free IBM Quantum
   account, API key and instance CRN saved in the qBraid Vault. Needed only for
   real hardware; everything here also runs on simulators.
2. **Install the environment**, kernel included:
   ```
   qbraid envs install qiskit_zngl3z
   ```
   Qiskit 2.5.2, Qiskit Runtime 0.50.0, Aer, qiskit-addon-opt-mapper, HiGHS.
3. **Pick a starting point:**

| Folder | What it is | Runs on |
|---|---|---|
| [`warm-start-qaoa/`](warm-start-qaoa/) | IBM's warm-start QAOA tutorial adapted for qBraid: exact classical baseline, angles trained classically, one hardware job | Simulator by default (about 2 min); one IBM job when you switch to hardware |
| [`qoblib/`](qoblib/) | The Quantum Optimization Benchmarking Library: fetch instances, run a classical baseline, plug in a quantum sampler, check with the official checkers, build a submission | CPU; any sampler you plug in |

## The rules that make results worth sharing

- **Classical anchor first.** Every quantum number goes next to the exact or
  best-known classical answer on the same instance. Both folders print it.
- **Train classically, sample on hardware.** QAOA angles can be optimized on a
  simulator or a qBraid GPU, so the QPU spends one job instead of hundreds.
- **Know what the experiment measures.** On max-cut, the warm start is already a
  bitstring and p = 1 warm-start QAOA returns it, so hardware measures fidelity to
  a known answer, not optimization gain (details in
  [`warm-start-qaoa/README.md`](warm-start-qaoa/README.md)). Constrained problems,
  such as the QOBLIB classes, are where the open questions are.
- **Cost.** IBM jobs are billed to your IBM plan (the free Open Plan has 10 QPU
  minutes per 28 days). Other QPUs on qBraid are billed in qBraid credits: run
  `qbraid devices get <qrn>` for the price before submitting.

## Going further on qBraid

- **GPUs on demand** for training and large simulations:
  `qbraid compute list`, then `qbraid compute up gpu-l4`; terminate when done.
- **Classical optimization at full strength** (PyVRP, OR-Tools, HiGHS, NVIDIA cuOpt,
  solver racing with certified gaps): [`../fleet-routing/`](../fleet-routing/) and the
  `optimization` environment, `qbraid envs install optimi_84bd83`.
- **Agents that know all of this:** qBraid's agent skills `solution-router`,
  `quantum-readiness` and `optimization-stack`
  (`qbraid skills search optimization`).

## Sources

- Warm-start QAOA: Egger, Mareček and Woerner, *Quantum* 5, 479 (2021); tutorial
  © IBM, Apache-2.0, from [Qiskit/documentation](https://github.com/Qiskit/documentation).
- QOBLIB: Koch et al., *Nature Computational Science* (2026),
  [doi:10.1038/s43588-026-00991-1](https://www.nature.com/articles/s43588-026-00991-1);
  code Apache-2.0, instances CC BY 4.0, <https://github.com/ZIB-AOPT/QOBLIB>.
- QAMOO: Kotil et al., *Nature Computational Science* (2025),
  [doi:10.1038/s43588-025-00873-y](https://doi.org/10.1038/s43588-025-00873-y);
  code at <https://github.com/stefan-woerner/qamoo>.
