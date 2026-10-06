#!/bin/bash
# Headless screenshots of key viewer states (both themes). On the qBraid pod the
# Playwright headless shell needs conda-forge GUI libs: LD_LIBRARY_PATH=/tmp/envs/chromelibs/lib
set -euo pipefail
cd "$(dirname "$0")"
SH=${CHROME:-$(ls ~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell | head -1)}
export LD_LIBRARY_PATH=${CHROME_LIBS:-/tmp/envs/chromelibs/lib}
shot() { # name, hash
  timeout 120 "$SH" --no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader --hide-scrollbars \
    --window-size=1440,980 --virtual-time-budget=${VT:-15000} --screenshot="results/viewer_$1.png" \
    "file://$PWD/viewer.html#$2" 2>/dev/null | tail -1
}
shot bracket_light "tab=bracket&theme=light"
shot bracket_dark "tab=bracket&theme=dark"
shot bracket_stress_light "tab=bracket&mode=stress&theme=light"
shot bracket_evolution_dark "tab=bracket&frame=350&theme=dark"
shot bracket_tour_dark "tab=bracket&tour=3&theme=dark"
shot top3d_light "tab=top3d&theme=light"
shot top88_dark "tab=top88&theme=dark"
shot validation_light "tab=validation&theme=light"
ls -la results/viewer_*.png
