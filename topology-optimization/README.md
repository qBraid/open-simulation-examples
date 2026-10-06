# Topology optimization: benchmarked, then built

Open-source SIMP topology optimization, verified against the published reference
codes and then used to design a printable titanium bracket for three load cases.
The viewer is `viewer.html`: a single self-contained page. It needs network access only
for the three.js CDN.

![bracket](results/viewer_bracket_light.png)

## What "top 10 %" means here, and the verdict

For design methods, the field measures itself against the published reference
implementations. The bar for this example:

| Bar | Source | Metric | Verdict |
|---|---|---|---|
| Reproduce **top88** (2D) | Andreassen, Clausen, Schevenels, Lazarov, Sigmund, *Struct Multidisc Optim* 43 (2011). MBB beam at the paper's three meshes, sensitivity and density filters; plus Sigmund's (2001) cantilever | Optimal compliance within 1 %, same layout | **Reached** (see table) |
| Reproduce **top3d** (3D) | Liu & Tovar, *Struct Multidisc Optim* 50 (2014), default cantilever 60 × 20 × 4 | Optimal compliance within 1 %, same layout | **Reached on compliance** (0.0032 % at matched iterations). Same topology; 4.6 % of the elements sit in a different place after iteration 56, see below |
| FE kernel correct | Timoshenko beam theory plus an independent P2 tetrahedral solution | Two discretisations agree on the 3D continuum answer | **Reached**: within 0.14 % |
| Industrial-flavour 3D design | Titanium mounting bracket, 3 load cases, printable STL | Crisp design (< 1 % grey), stress below yield with margin, watertight STL | **Reached** at 2 mm resolution: 88 g, 0.03 % grey, linear-elastic safety factor 4.0, watertight STL. See [bracket](#the-titanium-bracket) |

The papers print designs, not compliance values. So the numerical reference is the
authors' own code, downloaded at run time and run **unmodified** in Octave 10.3. The
only edits are removing the plotting and saving the final design
(`reference/make_headless.py`).

## Results

### 2D: top88 (Python port vs the published code)

| case | compliance, Python | compliance, published code | Δ | max abs layout difference | iterations | identical until it. |
|---|---|---|---|---|---|---|
| top88_mbb_60x20_ft1 | 216.8137 | 216.8137 | -1.7e-05 % | 5.4e-11 | 106 / 106 | 106 |
| top88_mbb_150x50_ft1 | 219.5199 | 219.5199 | -9.7e-06 % | 1.1e-10 | 95 / 95 | 95 |
| top88_mbb_300x100_ft1 | 246.0584 | 246.0584 | -1.4e-05 % | 8.6e-11 | 61 / 61 | 61 |
| top88_mbb_60x20_ft2 | 233.7146 | 233.7146 | -1.2e-05 % | 1.0e-10 | 144 / 144 | 144 |
| top88_mbb_150x50_ft2 | 235.7337 | 235.7337 | -2.0e-05 % | 1.9e-10 | 362 / 362 | 362 |
| top88_mbb_300x100_ft2 | 260.8374 | stopped by cap at it. 467 | -1.8e-05 % at it. 467 | – | 1236 / ≥467 | 467 |
| top88_cant_32x20_ft1 | 57.2805 | 57.2805 | +3.6e-05 % | 5.0e-11 | 72 / 72 | 72 |
| top88_cant_32x20_ft2 | 60.4441 | 60.4441 | +6.9e-05 % | 4.9e-11 | 94 / 94 | 94 |

The iterates are identical to round-off for long stretches; `identical_until_iter` in
`results/benchmarks.json` gives the first iteration where they differ by more than 1e-6.
Late in a run, the optimality-criteria update can oscillate. There, round-off differences
between two sparse solvers (CHOLMOD and Octave's solver) are amplified, and each run can
stop at a slightly different iteration. The final compliances still agree to well inside
the 1 % bar.

### 3D: top3d

| case | compliance, Python | compliance, published code | Δ | max abs layout difference | iterations | identical until it. |
|---|---|---|---|---|---|---|
| top3d_60x20x4 | 2418.1163 | 2416.6404 | +6.1e-02 % | 4.9e-01 | 135 / 200 | 56 |
| top3d_60x20x4_200it | 2416.7189 | 2416.6404 | +3.2e-03 % | 4.7e-01 | 200 / 200 | 56 |

`top3d_60x20x4` stops on the code's own rule (`change < 0.01`). The reference never meets that rule, because the OC update oscillates around 0.016, so it stops at its 200-iteration cap. `top3d_60x20x4_200it` forces the port to the same 200 iterations for a like-for-like comparison.

### FE validation: clamped cantilever, L/h = 10

| quantity | value |
|---|---|
| Euler–Bernoulli | 4000.0 |
| Timoshenko (Cowper κ = 0.8497) | 4030.6 |
| 3D continuum (P2 tetrahedra, extrapolated, order 1.93) | 4003.0 (-0.69 % vs Timoshenko) |
| H8 Richardson (order 1.76) | 4008.6 (+0.14 % vs continuum) |

| H8 bricks through depth | unknowns | tip deflection | error vs 3D continuum |
|---|---|---|---|
| 1 | 132 | 2591.2 | -35.27 % |
| 2 | 567 | 3503.2 | -12.49 % |
| 4 | 3,075 | 3859.4 | -3.59 % |
| 8 | 19,683 | 3964.6 | -0.96 % |

The 3D continuum answer sits 0.69 % below Timoshenko, because a fully clamped 3D root
restrains the cross-section, which beam theory ignores. The trilinear brick (the element
used by top3d and by the bracket) is stiff in bending. That is why the bracket keeps at
least 6 to 10 bricks across its members.

### The titanium bracket

Design space 96 × 48 × 36 mm, 2.0 mm bricks (20,736 elements, 69,825 unknowns), Ti-6Al-4V, 12 % of the volume, three load cases on the pin minimised together, Heaviside projection to β = 16.

| quantity | value |
|---|---|
| mass | 88.2 g (solid block 734.9 g) |
| design iterations / wall time | 160 / 21.4 min |
| grey (0.1 < ρ < 0.9) elements | 0.03 % |
| von Mises p99 in the body | 219.3 MPa (max 237.9 MPa) |
| linear-elastic safety factor vs 880 MPa yield (p99) | 4.01 |
| compliance LC1 vertical 8.0 kN | 3824.1 N·mm |
| compliance LC2 axial 6.0 kN | 376.9 N·mm |
| compliance LC3 45 deg 8.5 kN | 2467.4 N·mm |
| max displacement LC1 vertical 8.0 kN | 0.5633 mm |
| max displacement LC2 axial 6.0 kN | 0.0715 mm |
| max displacement LC3 45 deg 8.5 kN | 0.4338 mm |
| printable STL | `results/bracket_2mm.stl`: 27,696 triangles, watertight: True, mesh mass 88.2 g |

How it was made, and what is still open:

- **The run stopped at its 160-iteration cap, not on a convergence rule.** After the final projection step (β = 16) the OC update oscillates at its move limit, while the summed compliance stays within ±0.03 % (1517.2 to 1517.8). The design is settled; the stop rule is not met.
- **Resolution: 2 mm, not the planned 1.5 mm.** The 1.5 mm run (64 × 32 × 24, 161k unknowns) waited 35 minutes for a slot on the shared instance and was cancelled. On a dedicated `cpu-8v-32g` it is an estimated 1 to 2 hours. Finer bricks give thinner members and a smoother part; the layout should not change.
- **STL.** The density field is upsampled 2× (trilinear), contoured with marching cubes, and lightly Taubin-smoothed. The contour level (0.44) is chosen so the mesh carries exactly the optimized material volume (88.2 g), rather than using the nominal 0.5, which loses about 6 % on thin members.

## Reproduce

```bash
# environment (conda-forge), ~2.5 GB; put it on local disk
micromamba create -p ./env -c conda-forge python=3.12 numpy scipy scikit-sparse scikit-image scikit-fem trimesh matplotlib octave
export OCTAVE_HOME=$PWD/env            # conda-forge Octave 10.3 needs this

reference/fetch_reference.sh            # downloads top88.m / top3d.m and makes headless copies
(cd reference && octave-cli --no-gui run_reference.m cantilever)   # also: top3d, mbb, lk
python run_benchmarks.py all            # Python ports (single thread is fine)
python compare.py                       # results/benchmarks.json
python validate_fe.py                   # results/validation.json
BRACKET_H_MM=2.0 BRACKET_TAG=_2mm python bracket.py   # or 1.5 mm (default) on a bigger box
python export_viewer_data.py && python build_viewer.py # viewer.html
```

**Verification stamp.** date: 2026-10-01; env: conda-forge python 3.12, numpy 2.5, scipy 1.18, scikit-sparse 0.5 (CHOLMOD), scikit-fem 12.0, Octave 10.3 for the reference codes; machine: qBraid subscription pod (1 thread, nice) for Python runs and the 2 mm bracket; shared qBraid gpu-l4 instance (1 thread) for the Octave reference runs; wall: bracket 21 min, top88 suite 1 h (300x100 density filter alone 52 min), top3d 1.2-3.9 min, references about 50 min; cost: $0 on the pod; about 1 CPU-hour on the shared instance

## Honest limits

- **Physics.** Linear elastic, small strain, compliance objective. No contact at the
  pin or bolts, plasticity, fatigue, buckling or thermal loads.
- **Manufacturing.** No overhang, minimum-member-size or build-direction constraints. The
  STL is a starting geometry for an additive-manufacturing engineer, not a qualified part.
- **Stress.** Element-centroid von Mises stress on trilinear bricks. The p99 excludes the
  elements next to the clamped pads and the loaded hole, where the boundary conditions
  create singularities.
- **Against Abaqus/ATOM, Ansys and OptiStruct.** The optimization core is the same family
  (SIMP, filters, projection, multiple load cases, passive regions). What the commercial
  tools add is manufacturing and stress constraints, nonlinear and contact analysis,
  certified solvers and support. That gap is real; this example doesn't claim to close it.
