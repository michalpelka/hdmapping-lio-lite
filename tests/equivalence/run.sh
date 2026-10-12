#!/usr/bin/env bash
# Differential test: upstream HDMapping step-1 code (extracted verbatim, TBB/spdlog shimmed) vs mini_lio.
#
#   tests/equivalence/run.sh <HDMapping checkout> <observation_equations checkout>
#
# Upstream functions are cut out by line range, so the checkout must be the pinned commit below
# (observation_equations: 6850f190e380b763c0f6a4f93520e61e7c853aae, the HDMapping submodule pin).
set -euo pipefail

HD=${1:?usage: run.sh <HDMapping dir> <observation_equations dir>}
OE=${2:?usage: run.sh <HDMapping dir> <observation_equations dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

PIN=6f7622b83eb335c882255646a51de495a33d4d75
have=$(git -C "$HD" rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$have" != "$PIN" ]; then
  echo "ERROR: HDMapping is at $have, line ranges were taken at $PIN" >&2
  exit 1
fi

STEP1=$HD/apps/lidar_odometry_step_1
OUT=${TMPDIR:-/tmp}/mini_lio_equiv
mkdir -p "$OUT"

sed -n 79,238p   "$STEP1/lidar_odometry_utils.cpp"            > "$OUT/upstream_rgd.inc"   # update_rgd_ray
{ sed -n 14,42p  "$STEP1/lidar_odometry_utils_optimizers.cpp"
  sed -n 702,1505p "$STEP1/lidar_odometry_utils_optimizers.cpp"; } > "$OUT/upstream_opt.inc" # optimize_lidar_odometry + helpers
sed -n 2025,2166p "$STEP1/lidar_odometry_utils_optimizers.cpp" > "$OUT/upstream_hist.inc"  # anisotropic histograms

head -1 "$OUT/upstream_rgd.inc" | grep -q '^void update_rgd_ray('
grep -q '^void optimize_lidar_odometry(' "$OUT/upstream_opt.inc"
head -1 "$OUT/upstream_hist.inc" | grep -q 'table_buckets_nv_indoor'

g++ -std=c++17 -O2 -DNDEBUG -w "$HERE/equiv.cpp" \
  -I"$OUT" -I"$ROOT/include" -I"$ROOT/ext/unordered_dense" -I"$ROOT/ext/obs_eq" \
  -I"$HD/core/include" -I"$OE/codes" $(pkg-config --cflags eigen3 2>/dev/null || echo -I/usr/include/eigen3) \
  -o "$OUT/equiv"
"$OUT/equiv"
