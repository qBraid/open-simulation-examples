# QOBLIB on qBraid

Run the [Quantum Optimization Benchmarking Library](https://github.com/ZIB-AOPT/QOBLIB)
(QOBLIB) on qBraid: fetch instances, run a classical baseline, plug in a quantum or
hybrid sampler, check the result with the library's own checkers, and produce a
submission that passes the same validator QOBLIB's CI runs on every pull request.

QOBLIB is ten optimization problem classes that become hard for classical solvers at
roughly 100 to 10,000 variables, with reference solutions, per-class checkers and a
public leaderboard at <https://zib-aopt.github.io/QOBLIB/>. Paper: Koch et al.,
*The Quantum Optimization Benchmarking Library*, Nature Computational Science (2026),
[doi:10.1038/s43588-026-00991-1](https://www.nature.com/articles/s43588-026-00991-1);
preprint [arXiv:2504.03832](https://arxiv.org/abs/2504.03832). Code is Apache-2.0, instances
and solutions are CC BY 4.0. This kit fetches them at run time; none are copied here.

This kit wires up three of the ten classes end to end:

| Class | Problem | Baseline here | QUBO for samplers |
|---|---|---|---|
| `01-marketsplit` | Find x in {0,1}^n with Ax = b (pure feasibility) | HiGHS MIP | yes, energy 0 iff feasible |
| `07-independentset` | Maximum independent set | HiGHS MIP (gives a proven bound) | yes, energy = -\|I\| |
| `09-routing` | CVRP, 20 customers, 4 vehicles | PyVRP (heuristic, no bound) | no (use a routing formulation) |

The other seven classes (LABS, Birkhoff, Steiner tree packing, sports scheduling,
portfolio, network design, topology) follow the same layout; `qoblib.py` raises a clear
error for them rather than guessing a format.

## Quickstart (about 2 minutes)

```bash
# 1. Environment. On qBraid, check the catalog for the optimization environment
#    (HiGHS 1.15.1, PyVRP 0.14.0, vrplib 2.2.0, OR-Tools, Pyomo):
qbraid envs available            # look for "optimization" (slug optimi_84bd83 once published)
qbraid envs install optimi_84bd83
# Anywhere else: python -m venv venv && . venv/bin/activate && pip install -r requirements.txt

# 2. Fetch only the classes you need (sparse clone: about 200 MB for these three, 27 s)
python qoblib.py fetch

# 3. Look at instances and their best-known values
python qoblib.py list --class 07-independentset

# 4. Run a baseline; the solution is checked by QOBLIB's own checker if Rust is installed
python qoblib.py solve 07-independentset chesapeake --time 60
python qoblib.py solve 09-routing XSH-n20-k4-01 --time 30
```

Set `QOBLIB_HOME` to reuse an existing clone; by default the library goes to `./QOBLIB`.
Keep it off a small home disk if you fetch large classes (Steiner and portfolio are
several hundred MB each).

## Verified on qBraid (2026-10-06)

qBraid subscription pod, 4 CPUs (cgroup), one solve at a time, single-threaded solvers.
"Official checker" is the class's own Rust checker built from the fetched repository.

| Instance | Best known | Ours | Bound | Time | Official checker |
|---|---|---|---|---|---|
| `ms_03_050_002` | feasible | feasible | proven | 0.4 s | VALID |
| `ms_03_100_001` | feasible | feasible | proven | 0.3 s | VALID |
| `ms_04_050_001` | feasible | feasible | proven | 74 s | VALID |
| `MANN-a9` | 3 (optimal) | 3 | 3, proven optimal | < 0.1 s | VALID |
| `chesapeake` | 17 (optimal) | 17 | 17, proven optimal | < 0.1 s | VALID |
| `aves-sparrow-social` | 13 (optimal) | 13 | 13, proven optimal | < 0.1 s | VALID |
| `brock200-2` | 12 (optimal) | 12 | 18 (not proven in 120 s) | 120 s | VALID |
| `XSH-n20-k4-01` | 646 (optimal) | 646 | none (heuristic) | 30 s | VALID |
| `XSH-n20-k4-02` | 650 (optimal) | 650 | none (heuristic) | 30 s | VALID |
| `XSH-n20-k4-21` | 842 (best known) | 842 | none (heuristic) | 30 s | VALID |

Also verified:
- The published QUBOs score every curated reference solution checked (3 market split, 4 independent set)
  at exactly the expected energy (0 and -|I|).
- Submissions built by `qoblib.py submit` pass `misc/ci/check_submission.py` with the official checkers
  running, and an infeasible solution claimed as feasible is rejected (exit code 21, INFEASIBLE).

The easy instances are easy; that is the point of the ladder. Market split already takes
74 s at 4 x 50, and brock200-2 shows the gap between finding an optimum and proving it.
Larger instances in every class are where classical solvers stall and where a quantum or
hybrid method would have to show something.

## Plug in your own solver

A solver is a function `solve(instance, time_limit) -> Result`:

```bash
python qoblib.py solve 07-independentset MANN-a9 --solver my_module:my_solve
```

For a quantum or hybrid sampler you only need bitstrings. QOBLIB publishes its own QUBO
for each instance (`models/binary_unconstrained/qs_files`); `load_qubo` reads it and
`qubo_energy` scores a bit vector with the same convention:

```
E(x) = offset + sum_i q_ii x_i + 2 * sum_{i<j} q_ij x_i x_j
```

`best_of_samples(instance, bitstrings)` turns raw sampler output into the best solution
QOBLIB will accept (independent-set samples are repaired greedily; report that in your
submission's Workflow field). `example_sampler.py` shows the whole loop with a uniform
random stand-in sampler:

```bash
python example_sampler.py 07-independentset MANN-a9 --shots 2000
```

Replace its `sample()` with your sampler: a Qiskit `SamplerV2` run on IBM hardware, a QAOA
simulation, or a qBraid device job. The random stand-in reaches 2 of 3 on MANN-a9 and 14 of
17 on chesapeake, and finds nothing feasible on market split. That is the floor a real
sampler must beat, alongside the classical baseline.

**What is quantum-ready today.** The smallest QOBLIB QUBOs have 20 (market split) to about
40-50 (independent set) variables, so they fit on today's QPUs. As of 2026-10-06, of the
records in QOBLIB's best-known tables that credit a community submission, three (small Birkhoff
instances B7_7_8, B7_7_9 and B8_8_4) come from a submission labelled Quantum Hardware: a hybrid
loop of QAOA sampling plus CPLEX re-weighting
([20260804_E-FCFW_Pennington-Mohseni](https://github.com/ZIB-AOPT/QOBLIB/tree/main/03-birkhoff/submissions),
[arXiv:2509.10657](https://arxiv.org/abs/2509.10657)). All other credited records are classical.
A record means the first submission to reach a value, not a speed or quality win over the best
classical method on the same budget. Report quantum runs against both the classical baseline
and the random floor, with QPU time separated from classical time. IBM hardware
needs your IBM Quantum token in the qBraid Vault; see the platform's vault-credentials skill
and `../IBM_SETUP.md`.

## Validate and submit

```bash
python qoblib.py check 07-independentset MANN-a9 results/MANN-a9.sol

python qoblib.py submit 07-independentset MANN-a9 results/MANN-a9.sol \
    --name "Jane Doe" --affiliation "Example University" \
    --reference https://github.com/you/your-method --method HiGHS \
    --workflow "HiGHS MIP, 1 thread, default settings" --bound 3 --runtime 0.05
```

`submit` writes the submission where a pull request puts it,
`<class>/submissions/<YYYYMMDD>_<Method>_<Surname>/<instance>/` with
`<instance>_summary.csv` (the library's 30-column template, read from the repo so the
columns never drift) and `<instance>_solution.<ext>`, then runs QOBLIB's own validator.
Placement matters: the validator only runs the solution checker for submissions inside the
repository tree.

Then fork <https://github.com/ZIB-AOPT/QOBLIB>, commit that folder, and open a pull request.
The repository's GitHub Action re-runs the checker, two committee members review, and on
merge the best-known tables update automatically. The site also has an in-browser builder:
<https://zib-aopt.github.io/QOBLIB/submit.html>.

Rules worth knowing before you submit (from QOBLIB's CONTRIBUTING.md):
- Set `Optimality Bound` to `N/A` for heuristics. Setting it equal to your objective claims a
  proven optimum, and the checker then verifies optimality.
- An infeasible run can be reported honestly with `# Feasible Runs = 0`; `submit` does this
  automatically and warns you.
- Stochastic methods: at least 5 runs (10+ recommended). Report runtimes as averages,
  separate CPU, GPU and QPU time, and exclude queue time.
- An optional `<instance>_objective_time_series.json` enables time-to-solution analysis.

Full checking needs Rust (`curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal`).
Without it, `qoblib.py` still checks feasibility in Python and the validator checks the
submission's structure; the pull request's Action runs the full check either way.

## Multi-objective: QAMOO

Kotil, Pelofske, Riedmüller, Egger, Eidenbenz, Koch and Woerner, *Quantum Approximate
Multi-Objective Optimization*, Nature Computational Science (2025),
[doi:10.1038/s43588-025-00873-y](https://doi.org/10.1038/s43588-025-00873-y)
([arXiv:2503.22797](https://arxiv.org/abs/2503.22797)). It uses low-depth QAOA with
transferred parameters to approximate the Pareto front of multi-objective weighted max-cut,
run on IBM hardware and in MPS simulation. Code and data: <https://github.com/stefan-woerner/qamoo>
(Apache-2.0; Python 3.10, `pygmo` from conda-forge, JuliQAOA for parameter training; some
classical baselines need CPLEX or Gurobi). It is not part of this kit yet.
