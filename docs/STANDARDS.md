# Standards for every use case

## 1. The benchmark bar ("top-10% for this field")
Each example states up front, in its README, what top-10% means in its field and why:
- The **public benchmark or reference set**: published leaderboard, workshop, standard test set, or analytic or experimental reference.
- The **metric**: gap to best-known, RMSE against reanalysis, success rate at RMSD < 2 Å, error against experiment, and so on.
- The **comparison**: where published commercial and open results sit, so the claim is relative and sourced.

Report what was achieved, with an honest verdict: **reached**, **close** (say what's missing and what it would cost), or **not reached** (say why). Never round a miss up to a hit.

## 2. Visual quality
Every viewer is a single self-contained `viewer.html`. Data is inlined, three.js 0.170 comes via importmap from cdn.jsdelivr.net, and the page builds with `build_viewer.py`. The bar is "a demo you would show on a conference stage":
- **Physically meaningful encoding.** Perceptually uniform colour maps (viridis or cividis for magnitude, diverging for signed values) with a labelled colour bar and units. No rainbow maps.
- **Lighting and materials.** Use `MeshStandardMaterial` or `MeshPhysicalMaterial` with an environment map or hemisphere and key lights. Add soft shadows or SSAO where they help depth. Use tone mapping (ACES) and correct colour space.
- **Motion that explains.** Use animated quantities: particles advected along the real field, time-series playback with a scrubber, and morphing between design iterations. Include a guided "tour" button that flies the camera through 3–5 annotated viewpoints.
- **Comparison built in.** Toggle or slide between baseline and best, or between open-source and reference, and between mesh levels. The benchmark result is visible in the scene, not only in a table.
- **Performance.** 60 fps on a laptop GPU. Use instancing and BufferGeometry, and decimate to keep the page at 8 MB or less. Pause rendering when the tab is hidden.
- **Both themes.** Light and dark via CSS tokens and `prefers-color-scheme`. Responsive down to phone width.
- **Verification.** Take headless-Chromium screenshots of key states in both themes and commit them as `results/viewer_*.png`. On the pod, Chromium needs conda-forge system libs. Build a private copy per stream (for example `<scratchpad>/<stream>-chromelibs`, never a shared path; `/tmp` is wiped on pod restart), using the recipe in `wind-tunnel/README.md`. Run one screenshot session at a time.

## 3. Verification stamp
Every result records the date, environment spec, machine (pod or pool L4 box), wall time and cost. No stamp, no claim.

## 4. Pushing
Commit and push your branch after every milestone, at least every ~45 minutes of work. That includes the draft skill in `skills/<name>/SKILL.md`. Never leave work only on a pool box or in `/tmp`.
