#!/bin/bash
# Download the published reference codes (not redistributed in this repo) and make
# headless copies (plotting removed, final design saved). The algorithms are untouched.
set -euo pipefail
cd "$(dirname "$0")"
curl -sfL -o top88.m "https://www.topopt.mek.dtu.dk/-/media/subsites/topopt/apps/dokumenter-og-filer-til-apps/top88.ashx"
curl -sfL -o top3d.m "https://raw.githubusercontent.com/kliu99/Top3d/master/top3d.m"
grep -q "AN 88 LINE" top88.m && grep -q "169 LINE 3D" top3d.m
python3 make_headless.py
