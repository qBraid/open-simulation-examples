# AI car aerodynamics

NVIDIA's pretrained aerodynamics surrogates on qBraid GPUs. A surrogate is a
neural network trained on thousands of hours of high-fidelity CFD; given a car's
surface mesh it predicts the pressure and wall shear stress everywhere on it,
and so the drag, in seconds to minutes on one GPU instead of hours to days on a
cluster. This example scores two of them, **GeoTransolver** and **DoMINO**, on
the DrivAerML benchmark (the open dataset built on the DrivAer car developed
by TU Munich with Audi and BMW), then asks the question a carmaker would ask
next: does it work on *my* car? For that it runs six production cars, built
from open-licence fan meshes, against the drag coefficients the manufacturers
publish. Everything renders as a three.js scene in the Agent Canvas.

**Stands in for:** a first-pass CFD run in Ansys Fluent or STAR-CCM+ during
early design.
**Stack:** NVIDIA PhysicsNeMo 2.2 and PhysicsNeMo-CFD (Apache-2.0), model
checkpoints from Hugging Face (NVIDIA Open Model Agreement, commercial use
allowed), PyTorch 2.11, trimesh, scikit-image, pyvista.
**Quantum today:** none. These are classical deep-learning surrogates; nothing
in the workflow is waiting on a quantum computer.

## The bar

- **In distribution (DrivAerML validation split, 48 cars held out from
  training).** The high-fidelity answer is a hybrid RANS-LES run per car
  (Ashton et al. 2024). The bar: mean drag error under 2% and no car worse
  than 5%, the level at which a surrogate can stand in for CFD when ranking
  design variants (the split spans Cd 0.252 to 0.309).
- **Out of distribution (production cars).** There is no CFD for these. The
  reference is the manufacturer's published Cd, which comes from different
  wind tunnels, with moving ground or not, mirrors or not. Within 10% would be
  good for a model trained on morphs of a single car; within 20% is usable for
  ranking designs; worse than that means the model does not transfer.
- **Controls.** Production meshes need repair before a model can read them
  (see Method). Three DrivAerML cars go through the same repair, scored
  against their original CFD drag, so the cost of the repair is measured
  separately from the cost of a new shape.

## Results (verified 2026-10-18)

### DrivAerML validation split, 48 cars: GeoTransolver **reached**, DoMINO **close**

On every car of the held-out split (NVIDIA's benchmark metrics, against the
high-fidelity CFD on the same surface mesh). GeoTransolver clears the bar on
all 48; DoMINO's mean is well inside it, but one car (5.68%) is over 5%:

| Model | Cars | Drag error, mean | median | worst | Pressure L2 (area-weighted) | Time per car, one A100 |
|---|---|---|---|---|---|---|
| **GeoTransolver** | 48 | **0.84%** | 0.62% | 2.18% | **0.038** | about 4-6 min |
| DoMINO | 48 | 1.50% | 1.19% | 5.68% | 0.112 | about 1 min (45.7 min for all 48) |

The three cars in the viewer, drag integrated from the predicted surface
fields (`postproc.py`; reference area from DrivAerML):

| Car | CFD Cd | GeoTransolver | DoMINO |
|---|---|---|---|
| DrivAerML 202 (typical) | 0.286 | 0.288 | 0.279 |
| DrivAerML 419 (lowest drag in the split) | 0.252 | 0.252 | 0.254 |
| DrivAerML 439 (highest drag) | 0.309 | 0.305 | 0.306 |

### Controls: what the mesh repair costs

The same three cars after the shrink-wrap used for production cars, scored
against their original CFD Cd, exactly as a production car is scored:

| Model | Mean Cd bias | Worst |
|---|---|---|
| GeoTransolver | -16.5% | 23.8% |
| DoMINO | -12.3% | 16.4% |

The CFD fields themselves, carried over onto the wrapped surface, already
integrate about 10% low: the wrap smooths away underbody detail that carries
drag. So **any production-car number below reads 12-17% low from the repair
alone**, before any question of shape.

### Production cars: **not reached** (DoMINO close in magnitude, GeoTransolver far off)

| Car | Published Cd | DoMINO | | GeoTransolver | |
|---|---|---|---|---|---|
| Tesla Model X | 0.24 | 0.185 | -23% | 0.137 | -43% |
| Porsche Taycan | 0.22 | 0.247 | +12% | 0.156 | -29% |
| Hyundai Ioniq 5 | 0.288 | 0.253 | -12% | 0.198 | -31% |
| BMW i8 | 0.26 | 0.221 | -15% | 0.147 | -43% |
| BMW M3 (E92) | 0.31 | 0.225 | -27% | 0.157 | -49% |
| BMW M3 (E30) | 0.33 | 0.291 | -12% | 0.174 | -47% |
| **Mean absolute error** | | | **17%** | | **40%** |

- **DoMINO stays within 12-27%** on all six, and most of that is the
  12% low bias the controls measured. Ranking is weaker (Spearman 0.54 against
  the published order): the boxy 1980s E30 comes out highest, correctly, but
  the Taycan, lowest published, comes out fourth.
- **GeoTransolver, the most accurate model in distribution, is the worst out
  of it:** 29-49% low on every production car. Accuracy on cars like the
  training set does not tell you how a model transfers; only a test like this
  one does.
- No NaNs on any repaired mesh. On the raw fan meshes, GeoTransolver returned
  NaN on 23-100% of the surface of four cars (see Traps).

What this means for a carmaker: these checkpoints rank variants of a
DrivAer-class car in minutes with sub-2% drag error. For a different vehicle class,
expect to fine-tune on your own CFD before trusting absolute numbers, and
keep a control like the one here to know which regime you are in.

## Method

1. **Benchmark** (`run_bench.sh`, `mkconf.py`): NVIDIA's
   `workflows/benchmarking/main.py` from PhysicsNeMo-CFD at commit `0612ec4`,
   one Hydra config per model generated from `config_matrix_surface_hf.yaml`.
   Checkpoints are the ones pinned by that commit:
   `nvidia/geotransolver_drivaerml@626c1158` and `nvidia/domino_drivaerml@35b1bf1e`.
   Metrics (relative drag error, area-weighted pressure L2) are the
   benchmark's own, computed on the CFD surface mesh.
2. **Production cars** (`wrap_mesh.py`, `wrap_all.sh`): each fan mesh is
   scaled to the real car's length and placed in the DrivAerML frame (x
   downstream, front at x = -0.924 m, ground at z = -0.3176 m). It is then
   shrink-wrapped into one watertight outer skin:
   - sample the surface densely; mark the voxels (8 mm) within r of it;
     flood-fill the outside air; erode the closed solid back by r with a
     distance field; marching cubes; Taubin smoothing; keep the largest piece.
   - r = 24 mm closes panel seams and grille slots; the i8 needed r = 50 mm to
     close an open underside.
   - The STL is wound outward and the boundary `.vtp` **inward**, which is the
     DrivAerML convention the models were trained on (see Traps).
3. **Controls**: DrivAerML runs 202 (typical), 419 (lowest drag) and 439
   (highest drag) through the same wrap with `--keep-frame`, the CFD fields
   carried over by nearest cell, as cases 9202, 9419 and 9439.
4. **Drag** (`postproc.py`): D = -sum((p n_x + tau_x) A) over the predicted
   surface, n outward. DrivAerML pressure is kinematic (p / rho) and its wall
   shear stress has OpenFOAM's sign, so q = U^2 / 2 = 756.18 m^2/s^2 at
   U = 38.889 m/s. That reproduces every DrivAerML car's published Cd from its
   own CFD fields. Cd = D / (q A), with A = the DrivAerML reference area for
   DrivAerML cars and controls, and the frontal area of the repaired mesh
   (rasterised at 5 mm) for production cars.
5. **Viewer** (`build_viewer.py`, `viewer_template.html`): each surface is
   decimated to about 120k triangles (quadric), Cp quantised to 8 bits (255 = no value), geometry to
   16 bits, gzipped and inlined into `viewer.html`.

## The cars

| Car | Mesh licence | Published Cd | Source |
|---|---|---|---|
| [Tesla Model X](https://www.printables.com/model/916009) | CC-BY (Spark Models, Printables 916009) | 0.24 | Tesla: Model X design page (tesla.com/modelx/design) |
| [Porsche Taycan](https://www.printables.com/model/1301846) | CC0 (Printables 1301846) | 0.22 | Porsche Taycan technical data (newsroom.porsche.com), A = 2.33 m2 |
| [Hyundai Ioniq 5](https://www.printables.com/model/962998) | CC-BY-SA (low-poly by andikapratamaw, Printables 962998) | 0.288 | Hyundai Motor Group (hyundaimotorgroup.com) |
| [BMW i8](https://www.printables.com/model/1253551) | CC0 (Printables 1253551), toy-grade mesh | 0.26 | BMW i8 2014 efficiency page (bmw.mn) |
| [BMW M3 (E92)](https://www.printables.com/model/779483) | CC-BY (Printables 779483) | 0.31 | auto-data.net (not confirmed by BMW) |
| [BMW M3 (E30)](https://commons.wikimedia.org/wiki/File:BMW_M3_(E30).stl) | CC BY 4.0 (Martin Trafas, Wikimedia Commons) | 0.33 | Wikipedia / Hagerty (no BMW primary source found) |

The meshes are not redistributed here; each name links to its download page. Each was scaled to the
real car's overall length; the per-car rotations are in `wrap_all.sh`.

## The viewer

`viewer.html` is one self-contained file (about 15 MB, so the canvas serves it
from a local server rather than inline):

```bash
mkdir -p /tmp/carview && ln -sf "$PWD/car-aero-surrogates/viewer.html" /tmp/carview/index.html
(cd /tmp/carview && python3 -m http.server 8099 --bind 127.0.0.1 &)
qbraid-canvas --port 8099 --title "AI car aerodynamics"
```

- **Car:** the DrivAerML test cars, the wrapped controls and the six production cars.
- **Surface:** high-fidelity CFD (where it exists), GeoTransolver, DoMINO.
- **Colour:** pressure coefficient, or |Cp error| against CFD.
- **Tag and table:** each model's Cd against the CFD or published value, and
  the 48-car accuracy table. Grey patches mean the model returned no value.
- **Mouse:** drag to orbit, right-drag or shift-drag to pan, wheel to zoom.
  The orbit is clamped to stay outside the car and just below the ground plane
  (enough to see the underbody).

## Reproduce

```bash
# on a qBraid A100 (gpu-a100-sxm), from this folder
qbraid compute up gpu-a100-sxm                     # prints the ssh alias
H=<alias>; L=<instance label>
qbraid compute instances set $L --auto-stop 0 --max-session 360
# the Lab's own timers (120 min cutoff, 15 min idle) are separate: turn them off on the box
ssh $H 'python3 -c "
import configparser, pathlib
p = pathlib.Path.home()/\".qbraid\"/\"qbraidrc\"; p.parent.mkdir(exist_ok=True)
c = configparser.ConfigParser(); c.read(p); c.has_section(\"lab-settings\") or c.add_section(\"lab-settings\")
[c.set(\"lab-settings\", k, \"0\") for k in (\"auto_cutoff_minutes\", \"idle_timeout_minutes\", \"default_session_length\")]
c.write(open(p, \"w\"))"'
ssh $H mkdir -p /home/jovyan/aero
scp setup_box.sh mkconf.py run_bench.sh wrap_mesh.py wrap_all.sh postproc.py work/val_ids.txt $H:/home/jovyan/aero/
ssh $H 'cd /home/jovyan/aero && bash setup_box.sh'          # ~10 min: env + 48 cars (35 GB)

# production cars: download each mesh (links in The cars) as <name>.stl into src/, scale and place them,
# then wrap (wrap_all.sh carries the per-car rotation fixes for the orientation heuristic's misses)
python3 prep_fan.py src fanml && ssh $H mkdir -p /home/jovyan/aero/fanin && for i in 9001 9002 9003 9004 9006 9007; do scp fanml/run_$i/drivaer_$i.stl $H:/home/jovyan/aero/fanin/; done
ssh $H 'cd /home/jovyan/aero && bash wrap_all.sh'

# benchmark: one process per model; separate runs can share the GPU (80 GB is plenty), but they split its time
ssh $H 'cd /home/jovyan/aero && pn/bin/python mkconf.py && C="[$(sed s/^/run_/ val_ids.txt | paste -sd,)]" &&
  MODELS=geotransolver_surface EXTRA="+benchmark.datasets.0.case_ids=$C run.save_inference_mesh=true reports.visual_case_ids=[run_202,run_419,run_439]" bash run_bench.sh'
# same for domino_surface, and for the production cars + controls with TAG=_fan and run.output_dir=.../out/<model>_fan

# drag and viewer surfaces, then pull and build
ssh $H 'cd /home/jovyan/aero && mkdir -p vnpz && ls out/*/inference_*.vtp | xargs -P 8 -I{} pn/bin/python postproc.py {} drivaerml /dev/null vnpz'
scp -r $H:/home/jovyan/aero/vnpz/. work/viewer_npz/      # and out/*/metrics_cache into work/out/
python3 build_viewer.py
qbraid compute instances terminate $L --yes
```

### Traps, all handled in the scripts

- **Two shutdown timers, and the CLI only sets one.** `qbraid compute instances
  set --max-session/--auto-stop` is enforced by qBraid's servers. The Lab
  extension inside the box has its own: a 120-minute cutoff from container
  start and a 15-minute idle timeout (idle = no browser input *and* CPU under
  30% *and* GPU under 20%). They live in `~/.qbraid/qbraidrc` `[lab-settings]`,
  which a fresh box doesn't have, so the defaults apply whatever the CLI says.
  The stricter one wins. Write `0` to all three keys (Reproduce, above).
  Reported as qBraid/qBraid-CLI#392.
- **The driver decides the CUDA stack, not the GPU.** One A100 box carried
  driver 570 (CUDA 12.8), another 595. `uv` resolves torch and warp builds for
  CUDA 13 that the 570 driver can't run. `setup_box.sh` pins
  `torch==2.11.0+cu128` (the newest cu128 build; PhysicsNeMo needs >= 2.10) and
  the `warp-lang 1.18.0+cu12` wheel from NVIDIA's GitHub releases (PyPI only
  ships cu13; PhysicsNeMo needs >= 1.14).
- **cuDNN `NOT_INITIALIZED`.** A cu13 cuDNN wheel lands in the same directory as
  the cu12 one and overwrites it. Uninstall `nvidia-cudnn-cu13`, reinstall
  `nvidia-cudnn-cu12==9.19.0.56`.
- **DoMINO asks for ~740,000 GiB of GPU memory.** Without cuML, PhysicsNeMo's
  k-nearest-neighbour search falls back to brute force in torch, which is
  O(N x M) in memory. `setup_box.sh` patches the dispatcher to use scipy's
  KD-tree on the CPU and move the result back to the GPU.
- **XMGN and FigConvNet need `torch_scatter` and `torch_cluster`**, from the
  PyG wheel index for torch 2.11 + cu128.
- **No inference meshes are saved for most cases.** The benchmark writes them
  only for `reports.visual_case_ids` (default `[run_1, run_11]`), even with
  `run.save_inference_mesh=true`. Pass both.
- **Hydra keys not in the schema need a `+`:** `+benchmark.datasets.0.case_ids=[...]`.
- **DrivAerML winds its surface files inward.** The boundary `.vtp` normals
  point into the car (OpenFOAM patch convention, out of the fluid); the STL's
  point outward. The models were trained on that. An outward-wound `.vtp` reads
  as an inside-out car: DoMINO's drag on the wrapped controls was off by
  68-77% until the winding was flipped, and 4-7% after.
- **Fan meshes are not CFD meshes.** Out of six: four were upside down or back
  to front after an automatic orientation (fixed by checking wheelbase,
  overhangs and mirror position against published specs); several had
  zero-area faces, and a single one makes GeoTransolver return NaN over the
  whole car (its global attention spreads it); two had interior shells
  doubling the surface area. The shrink-wrap removes all three problems.
- **DrivAerML wall shear stress has OpenFOAM's sign** (force of the wall on the
  fluid). Subtracting it, as for a fluid-on-wall convention, lowers every Cd by
  about 20% and makes the implied dynamic pressure wander between cars. Add it.
- **three.js orbit with z up:** set `camera.up` *before* constructing
  `OrbitControls`. The controls fix their orbit axis at construction, so
  setting it afterwards (the first version of this viewer) makes every drag
  jerk, spin and flip.
- **Never `pkill -f` a pattern your own shell matches,** and never print
  `ps aux` on a box: Jupyter's token is in its command line. Use anchored
  patterns like `pgrep -f "^/home/jovyan/aero/pn/bin/python main.py"`.

## Limits

- **Two models, not five.** PhysicsNeMo-CFD also ships Transolver,
  X-MeshGraphNet and FigConvNet DrivAerML checkpoints; the environment
  supports them (`setup_box.sh` installs their extra wheels), but they were
  not run for time. `MODELS=transolver_surface ... bash run_bench.sh` adds one.
- **Production cars are a stress test, not a validation.** There is no CFD
  for them. The published Cd values come from different tunnels, ground
  simulation, trims and mirror configurations, and two (M3 E92, M3 E30) are
  not from a BMW primary source.
- **The meshes are approximate.** Fan models have smoothed underbodies, closed
  grilles and simplified wheels, and the shrink-wrap smooths further. The
  controls measure what the wrap costs on DrivAer itself; on a production car
  the missing underbody detail could matter more or less.
- **Frontal area comes from the mesh.** For the Taycan it matches Porsche's
  2.33 m^2 to 0.3%; the Model X mesh gives 3.30 m^2, which looks high, so its
  Cd may read low for that reason alone.
- **Single operating point.** Every model was trained at one speed
  (38.889 m/s) with no yaw. Nothing here says how they do at other speeds or
  in a crosswind.
- **Surface only.** These are the surface checkpoints: pressure and wall shear
  stress on the body. See "Next" for the flow field.

## Next

| Step | What it needs | Estimate |
|---|---|---|
| **Flow field around each car** (velocity, pressure, turbulence in the air; particles and vortices as in `wind-tunnel/`) | NVIDIA's volume checkpoints (`geotransolver_volume`, `transolver_volume`, `domino_volume`, in `config_matrix_volume_hf.yaml`). Checked in the code, not run: DoMINO's volume wrapper evaluates at the points of whatever `.vtu` it is given (`physicsnemo/cfd/evaluation/models/wrappers/domino/inference.py`, "coordinate-conditioned: feeding mesh.points"), plus the car's STL, so a grid of points around a production car is valid input. Validate first on one or two DrivAerML cars, whose CFD volume is about 49 GB each | 1-2 h on one A100, about 250-450 credits |
| The other three surface models on all 48 cars | `MODELS=transolver_surface xmgn_surface fignet_surface` | about 2-4 h of A100 |
| Fine-tune on a customer's own family of shapes | their CFD runs (tens to hundreds of cases); PhysicsNeMo's training recipes | depends on data |
| CFD truth for one production car | OpenFOAM on the wrapped Taycan (cfd-stack skill), to separate "the model is off" from "the published Cd is from a different setup" | 1-2 days on `cpu-64v-256g` |

## Compute used

| Run | Machine | Time | Credits |
|---|---|---|---|
| 2026-10-17: first pass (5 DrivAerML cars, raw fan meshes; the fan results were invalid, see Traps) | 1 x `gpu-a100-sxm`, driver 570 | 110 min | 407.60 |
| 2026-10-18: this README (48 cars x 2 models, 6 production cars, 3 controls) | 2 x `gpu-a100-sxm` (A100 80 GB PCIe, driver 595; A100-SXM4 80 GB, driver 570) | 198 + 152 min | 746.65 + 559.18 |
| **Total** | | | **1,713.43 credits (about $17 USD)** |

At 3.74 credits per minute per A100. About 360 credits of the second run were
idle time: a second box waiting 30 minutes on cars that hadn't been downloaded
to it, and both boxes sitting finished for 24-42 minutes before being
terminated. A clean rerun of everything here is about 3 GPU-hours, roughly 700
credits. The mesh preparation, drag integration and viewer build ran on the
instances' CPUs and on the subscription pod (no solver work there).
