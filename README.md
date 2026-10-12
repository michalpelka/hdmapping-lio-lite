# mini_lio

Minimal, header-only LiDAR-inertial odometry for hand-held scanners, ported from **HDMapping step 1**
(`MapsHD/HDMapping`, `apps/lidar_odometry_step_1`). Streaming API, no files, no GUI.

Dependencies of the library: Eigen (system), plus ~220 KB vendored in `ext/` (VQF, Fusion/Madgwick, unordered_dense,
one generated Jacobian header). No TBB, spdlog, nlohmann, toml++, LASzip, imgui, raylib, proj. OpenMP is optional.
The session runner `mini_lio_run` (below) additionally needs system LASzip (`apt install liblaszip-dev`).

## Pipeline (same as upstream)

1. **AHRS** (Madgwick/Fusion or VQF) turns IMU samples into an orientation per IMU sample.
2. **Chunk** = `poses_per_chunk` (20 = 0.1 s @ 200 Hz) IMU-rate poses plus the points between them.
3. **Predict**: previous end pose + AHRS relative rotations + constant velocity from the two previous chunks.
4. **Optimize** all poses of the chunk jointly (Tait-Bryan, LM): point-to-NDT terms against a hierarchical
   indoor/outdoor grid map (0.1 / 0.3 m), a tight motion-model prior between consecutive poses, first pose fixed,
   anisotropic weighting, view-point/normal check.
5. **Update the map** with the chunk's points; rebuild it after `sliding_window_distance` metres,
   or by time with `window_seconds` (map then covers roughly `window/2 .. window` seconds).

## Use

```cpp
#include <mini_lio/lio.hpp>

mini_lio::Params p;
p.use_vqf = true;          // recommended for hand-held motion, see below
p.nr_iter = 10;            // see "Tuning"
p.window_seconds = 3.0;    // rolling map of the last ~1.5-3 s
mini_lio::Lio lio(p);
lio.setCallback([](const mini_lio::ChunkResult& r) { /* r.poses, r.points_global */ });

// any order between the two streams, any batch size:
lio.addImu({t, gyro_rad_s, acc_g});            // acc in g, +1 g "up" at rest
lio.addPoints(points);                         // LidarPoint{p_in_imu_frame, t}, time-sorted
lio.process();                                 // runs every chunk that has all its data
```

Inputs: IMU and LiDAR on the **same clock**; points already in the **IMU frame** (apply the lidar->imu
extrinsics yourself); `0.3 < |xy| < 70 m` is filtered like upstream. The first `initial_points` (10 000) points seed
the map and the first chunk starts after them. World frame = gravity-aligned AHRS frame, origin at the first pose.
`lio.windowPoints()` returns the rolling map points, `lio.trajectory()` all IMU-rate poses.

Build and demo: `cmake -S . -B build -G Ninja && cmake --build build && ./build/sim_demo --seconds 10 --iters 10 --vqf --window 3`
(synthetic room, spinning 32-ring LiDAR, noisy 200 Hz IMU, ground truth, ATE report).

## Session runner: `mini_lio_run` (LAZ + IMU csv in, HDMapping-style folder out)

```
cmake -S . -B build -G Ninja && cmake --build build          # builds mini_lio_run if LASzip is found
./build/mini_lio_run <session dir> [-o out_dir] --vqf --iters 10 [--window 3] [--max-seconds 20]
```

Input: `*.laz/*.las` and `*.csv` are sorted and paired by position. **No calibration**: points are used as recorded, and
a warning is printed only if a LAZ contains more than one lidar id (`user_data`). IMU rows of other `imuId`s are skipped
(`--imu-id`, default 0), legacy headerless IMU files are read. LAZ `gps_time` and csv `timestamp` are nanoseconds. Like
upstream, the points of the first laz are dropped (`--use-first-file` keeps them); its IMU rows are used.

Output folder (default `<session>/lio_result_N`), upstream names and columns:

| file | content |
|---|---|
| `scan_lio_K.laz` | raw (not decimated) points, intensity and `gps_time` (ns) kept, in the frame of scan K's first pose; points closer than `--min-out-range` (0.5 m) dropped |
| `trajectory_lio_K.csv` | IMU-rate poses relative to scan K's first pose (`timestamp_nanoseconds pose00..pose23 timestampUnix_nanoseconds om_rad fi_rad ka_rad`) |
| `session_poses.mrp`, `session_ini_poses.mri` | first pose of every scan (scan -> world); world = gravity-aligned AHRS frame, origin at the first pose |
| `session.mjs`, `point_sizes_per_chunk.json` | session description for HDMapping step 2 |

A scan is `--chunks-per-scan` chunks (51 = 1020 poses ~ 5 s, as upstream); a shorter tail is kept. Differences: `om/fi/ka`
hold the Tait-Bryan angles of the optimized world pose (upstream: raw AHRS orientation); `timestampUnix` is looked up from the
IMU csv; with `--no-laz` `session.mjs` still lists the (absent) scan files. Not tested against HDMapping's own reader, only
against the formats read in its source (`save_result`, `load_trajectory_csv`, `save_poses`).

Test data: `sim_demo --export-session DIR` writes the simulated run as a Mandeye-style session (a junk second IMU id,
a dropped first file) plus `ground_truth.tum`.

## Verification

`tests/equivalence/run.sh <HDMapping@6f7622b> <observation_equations@6850f19>` cuts upstream's functions out of the
checkout **verbatim** (TBB/spdlog shimmed, real `Core/structures.h`/`transformations.h`, real generated Jacobians) and
runs them against mini_lio on random inputs:

| component | result |
|---|---|
| RGD map update (`update_rgd_ray`, with and without ray-cast, two viewpoints) | **bit-identical** cells: mean, cov, cov^-1, normal, count, hits |
| anisotropic normal histograms (upstream lambda) | **bit-identical** |
| `optimize_lidar_odometry` vs `optimize_chunk`, 20 poses, 12 LM iterations | poses agree to **~1e-12 m**, step norm to 6+ digits (relative-pose Jacobian is numeric here, upstream's is generated) |

Session path (`sim_demo --export-session` -> `mini_lio_run` -> independent numpy/laspy check of the written files,
12 s, VQF, 10 iterations, window 3 s): trajectory read back from `session_poses.mrp` + `trajectory_lio_K.csv`
**7.0 mm / 0.20 deg** best-fit ATE (same as the in-memory run), and `scan_lio_K.laz` points mapped back to world lie
**0.5 cm median / 1.6 cm p90** from the true room surfaces; time and intensity fields survive. Synthetic data again, not a real Mandeye session.

Not covered by the differential test: the streaming driver (`lio.hpp`: chunking, prediction, map update/window
logic) and the AHRS wrappers. They are a hand port of the same code and only checked end to end by the simulation.

Synthetic end-to-end result (`sim_demo`, 8 s, VQF, 2 vCPU, standard best-fit rigid alignment): position ATE
**~7 mm RMS**, rotation **~0.2 deg RMS**, identical for 5..100 iterations. Single- and multi-thread runs produce
byte-identical trajectories (fixed 16-chunk reduction). This is a clean synthetic room, not real Mandeye data.

## Tuning

| `nr_iter` | accuracy (sim) | speed on 2 vCPU |
|---|---|---|
| 5 | same | 4.9x real time |
| 15 | same | 2.1x |
| 100 (upstream-like) | same | 0.35x |

Upstream grows `lm_factor` x10 every 10 iterations until the step vanishes, so it effectively always runs ~100+
iterations per chunk. Start at 8-15 and check on your own data; a real scene is less well conditioned than the sim.

AHRS: in the sim the alignment-free tilt error was ~2.4 deg with Fusion (gain 0.5, follows the accelerometer during
hand motion) vs ~0.45 deg with VQF. Position ATE was the same, but VQF is the safer default for hand-held use.

## Differences from upstream

- Streaming instead of load-everything; no `.cache` files, TOML, GUI (LAZ/CSV only in `mini_lio_run`).
- Dropped: planarity/norm ablation flags, outer-RGD threshold flag, IMU preintegration, `robust_and_accurate`
  (sf2/rigid_sf), consistency/step 2, reference-point alignment, per-axis motion sigmas (xyz and angles share one value).
- Relative-pose Jacobian by central differences (the generated header is 838 KB); verified above.
- A chunk with <= `min_points_per_chunk` points is propagated by the motion model (upstream drops it, leaving a gap).
- Optional time-based window (`window_seconds`).
- NV histogram tables are persistent with touched-bin reset instead of two fresh 8 MB tables per chunk (same values).

## LASzip note

Debian/Ubuntu ship two libraries: `liblaszip.so` (the real one) and `liblaszip_api.so`, a dlopen shim whose
`laszip_create` fails unless `laszip_load_dll()` was called. CMake therefore looks for `laszip` before `laszip_api`.
If you build LASzip yourself, where `laszip_api` is the real library, it falls back to that.

## Upstream quirks kept on purpose (so results stay comparable)

- Cell **mean is never updated** (first point of the cell); covariance accumulates around it.
- Cell index is a truncating `int16` cast: cell 0 is twice as wide, range +-3.2 km at 0.1 m.
- A chunk uses points in `[t(first pose), t(last pose))`, so the last IMU period of each chunk is not used.
- Point timestamps are re-spread uniformly across the chunk (`correct_point_timestamps`), assuming uniform sampling.
- The first chunk is registered against the map seeded from the first 10 000 points.
- The orientation prior is tight (0.01 deg between consecutive poses), so LiDAR barely corrects AHRS rotation.
