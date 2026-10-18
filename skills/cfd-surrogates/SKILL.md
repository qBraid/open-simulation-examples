---
name: cfd-surrogates
description: Predict car aerodynamics (surface pressure, wall shear stress, drag) in minutes with NVIDIA's pretrained PhysicsNeMo surrogates (GeoTransolver, DoMINO, Transolver, X-MeshGraphNet, FigConvNet) on a qBraid GPU. Use when a user wants fast drag or pressure estimates for a car shape, to benchmark AI surrogates against high-fidelity CFD (DrivAerML), to run their own or an open car mesh through a pretrained model, or an open alternative to a first-pass Fluent/STAR-CCM+ run. Covers the verified GPU install and its traps, mesh preparation, the controls that make an out-of-distribution number honest, drag conventions, cost, and the three.js viewer.
metadata:
  version: "0.1.0"
  status: "provisional"
  verified: "2026-10-18"
---

# CFD surrogates on qBraid

A surrogate is a neural network trained on many high-fidelity CFD runs. Given a
surface mesh it predicts pressure and wall shear stress on every cell, and so
the drag, in 1-5 minutes on one A100 instead of hours to days of CFD. NVIDIA
publishes several, trained on DrivAerML (500 morphs of the DrivAer car, the
open reference body from TU Munich, Audi and BMW), under the NVIDIA Open Model
Agreement, which allows commercial use.

What an agent has to supply: a working CUDA stack, an input mesh in exactly the
convention the model was trained on, an honest reference to compare against,
and a control that measures what the mesh preparation itself costs.

Start from the user's goal (see solution-router). If they need a number they
will sign off on, a surrogate is a screening tool, not the answer: pair it with
a validated CFD run (cfd-stack).

## Get the scripts

The worked example is `car-aero-surrogates/` in
https://github.com/qBraid/open-simulation-examples. Its README has the full
reproduce block, every result and the compute bill.

```bash
git clone --depth 1 https://github.com/qBraid/open-simulation-examples
```

## 1. Machine and timers

- One `gpu-a100-sxm` (80 GB) runs every model. Check the driver first:
  `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`. Boxes
  under the same profile carried driver 570 (CUDA 12.8) and 595.
- **Set both shutdown timers before a long run.** `qbraid compute instances set
  <label> --max-session N --auto-stop 0` is enforced by qBraid's servers. The
  Lab extension inside the box also stops it at 120 minutes of container uptime
  and after 15 idle minutes (idle = no browser input, CPU under 30% and GPU
  under 20%), read from `~/.qbraid/qbraidrc` `[lab-settings]`. A fresh box has
  no such file, so the defaults apply. Turn them off over ssh:
  ```bash
  ssh $H 'python3 -c "
  import configparser, pathlib
  p = pathlib.Path.home()/\".qbraid\"/\"qbraidrc\"; p.parent.mkdir(exist_ok=True)
  c = configparser.ConfigParser(); c.read(p); c.has_section(\"lab-settings\") or c.add_section(\"lab-settings\")
  [c.set(\"lab-settings\", k, \"0\") for k in (\"auto_cutoff_minutes\", \"idle_timeout_minutes\", \"default_session_length\")]
  c.write(open(p, \"w\"))"'
  ```
  Keep `--max-session` as the backstop, and terminate the box when done.

## 2. Environment (verified 2026-10-18)

`car-aero-surrogates/setup_box.sh` builds it in about 10 minutes and downloads
the 48 DrivAerML validation cars (35 GB) in parallel. Traps it handles, none of
which announce themselves:

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False, or warp finds no device | `uv` resolves CUDA 13 builds of torch and warp; driver 570 can't run them | `torch==2.11.0+cu128` (newest cu128 build; PhysicsNeMo needs >= 2.10) and the `warp_lang-1.18.0+cu12` wheel from NVIDIA's GitHub releases (PyPI ships cu13 only; PhysicsNeMo needs >= 1.14, 1.12 lacks `LOG_WARNING`) |
| `CUDNN_STATUS_NOT_INITIALIZED` | the cu13 cuDNN wheel overwrites the cu12 one in the same directory | uninstall `nvidia-cudnn-cu13`, reinstall `nvidia-cudnn-cu12==9.19.0.56` |
| DoMINO: CUDA OOM asking for ~740,000 GiB | no cuML, so k-nearest-neighbours falls back to brute-force torch, O(N x M) memory | patch the dispatcher to scipy's KD-tree on the CPU (in `setup_box.sh`) |
| X-MeshGraphNet / FigConvNet import errors | need `torch_scatter`, `torch_cluster` | PyG wheels for `torch-2.11.0+cu128` |
| DoMINO: "No STL file found" | DoMINO reads `drivaer_<i>.stl` as well as the surface `.vtp` | download both |
| No prediction meshes written | only `reports.visual_case_ids` (default `[run_1, run_11]`) are saved, even with `run.save_inference_mesh=true` | pass both |
| Hydra: key not in struct | `case_ids` is not in the schema | `+benchmark.datasets.0.case_ids=[run_4,...]` |

Never `pkill -f` a pattern your own ssh command line also matches, and never
print `ps aux` on a box (Jupyter's token is in its arguments). Anchor patterns:
`pgrep -f "^/home/jovyan/aero/pn/bin/python main.py"`.

## 3. Decision rules

| Situation | Use |
|---|---|
| A car close to the DrivAer family (sedan/fastback/estate proportions), best accuracy | GeoTransolver |
| A shape unlike the training set (SUV, sports car, real production car) | DoMINO, and say how far off the controls were. On six production cars: DoMINO 17% mean absolute error against published Cd, GeoTransolver 40% (the most accurate in distribution, the worst out of it); both read low, 12-17% of it from the mesh repair |
| Many design variants, interactive ranking | any surrogate, one GPU, 1-5 min per car; rank, then confirm the shortlist with CFD |
| A number someone signs off on, or a shape far outside training | CFD (cfd-stack), or fine-tune the surrogate on CFD of the user's own family of shapes |

Measured on one A100, 8M-cell DrivAerML surfaces: DoMINO about 1 min per car
(45.7 min for 48), GeoTransolver about 4-6 min per car. Two processes share
one GPU fine but split its time. On 1-1.7M-cell repaired meshes, 0.7-1.4 min.
At 3.74 credits per A100-minute: about 4 (DoMINO) to 20 (GeoTransolver)
credits per car. When splitting cars across boxes, download each box's own
list, and poll for idle boxes: a finished box bills until terminated.

## 4. Validation is mandatory

Run the DrivAerML validation split (48 cars held out from training) before any
user car, and report it next to the user's result. The benchmark is NVIDIA's
`physicsnemo-cfd/workflows/benchmarking/main.py`; its metrics (relative drag
error, area-weighted pressure L2) are computed on the CFD surface mesh.
Verified on qBraid, 48 cars: GeoTransolver mean drag error 0.84% (worst
2.18%, pressure L2 0.038); DoMINO 1.50% (worst 5.68%, L2 0.112). Match these
before trusting anything else.

## 5. A user's own car

Read `car-aero-surrogates/wrap_mesh.py`. The steps, each of which broke a run:

1. **Frame.** DrivAerML: x downstream, front at x = -0.924 m, y centred,
   z up, ground at z = -0.3176 m, real size in metres. Scale to the real
   overall length. Automatic orientation from mesh extents gets up/down and
   front/back wrong often (4 of 6 cars): check wheelbase, overhangs and mirror
   position against published specs with a side and top view.
2. **One clean outer skin.** Fan and CAD-export meshes have zero-area faces (a
   single one makes GeoTransolver return NaN on the whole car: its global
   attention spreads it), interior shells and open seams. Shrink-wrap: voxels
   within r of the surface, flood-fill the outside, erode back by r on a
   distance field, marching cubes, Taubin smoothing, keep the largest piece.
   r = 24 mm closes seams and grille slots; an open underside needed 50 mm.
   Check surface area and volume against a wrapped DrivAer (about 28-34 m^2, 5-7.6 m^3).
3. **Winding, the trap that costs the most.** DrivAerML's boundary `.vtp` is
   wound **inward** (OpenFOAM patch normals point out of the fluid, into the
   car); its STL is wound outward. The models learned that. An outward `.vtp`
   reads as an inside-out car: drag came out negative, or 68-77% off on
   controls, until the `.vtp` was flipped.
4. **Controls.** Put 2-3 DrivAerML cars through exactly the same preparation
   (`--keep-frame --truth boundary_<i>.vtp`) and score the predicted Cd
   against their original CFD Cd. That is the error the preparation alone
   adds; quote it next to every production-car number.

## 6. Drag conventions

DrivAerML pressure is kinematic (p / rho) and its wall shear stress has
OpenFOAM's sign (wall on fluid). Drag = -sum((p n_x + tau_x) A) with n
outward, q = U^2 / 2 = 756.18 at U = 38.889 m/s; this reproduces every
car's published Cd from its own fields. Subtracting tau instead lowers Cd by
about 20%. For DrivAerML cars, Cd uses the dataset's reference area; for other
cars, use the frontal area of the mesh (rasterise the y-z projection), and
check it against a published frontal area where one exists.

The reference for a production car is the manufacturer's published Cd, which
comes from a different tunnel, possibly another trim, mirrors or not. Say so.

## 7. Show it

`car-aero-surrogates/build_viewer.py` decimates each surface to about 120k
triangles, quantises Cp to 8 bits (255 = NaN, drawn grey) and positions to
16 bits, and inlines it all into one three.js page: car, model, Cp or error
against CFD, camera presets, and the accuracy tables. A dozen cars come to
about 15 MB, over the canvas's inline limit of about 5 MB, so serve the folder
and render with `qbraid-canvas --port <port>` (agent-canvas skill). With z up (the DrivAerML
frame), set `camera.up` before constructing `OrbitControls`: it fixes its orbit
axis at construction, and setting it afterwards makes every drag jerk and flip.

## Quantum: the short answer

None. These are classical surrogates; nothing here waits on quantum hardware.
For differential-equation solvers and the honest horizon, use the
quantum-readiness skill.

## Verification stamp

- 2026-10-18: PhysicsNeMo-CFD commit `0612ec4`, nvidia-physicsnemo 2.2.2,
  torch 2.11.0+cu128, warp-lang 1.18.0+cu12, cuDNN 9.19, pyvista 0.49;
  checkpoints `nvidia/geotransolver_drivaerml@626c1158`,
  `nvidia/domino_drivaerml@35b1bf1e`.
- Machines: two `gpu-a100-sxm` instances (A100 80 GB; drivers 570.172 and
  595.91).
- Re-verify on every PhysicsNeMo or checkpoint bump.
