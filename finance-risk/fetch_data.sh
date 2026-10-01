#!/bin/bash
# Fetch the public data used by this example into ./data (not committed: redistribution terms
# belong to the publishers). Ken French Data Library (Dartmouth) and CBOE index history.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data; cd data
for f in 30_Industry_Portfolios_CSV F-F_Research_Data_Factors_CSV; do
  curl -sfL -o $f.zip "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/$f.zip"; unzip -o -q $f.zip; rm $f.zip
done
for s in VIX VIX9D VIX3M VIX6M VIX1Y SKEW; do
  curl -sfL -A "Mozilla/5.0" -o ${s}_History.csv "https://cdn.cboe.com/api/global/us_indices/daily_prices/${s}_History.csv"
done
ls -la
