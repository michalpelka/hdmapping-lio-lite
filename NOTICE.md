# Third-party code and attribution

mini_lio is a port of the LiDAR-inertial odometry of **HDMapping step 1**
(<https://github.com/MapsHD/HDMapping>, `apps/lidar_odometry_step_1`, MIT, Copyright (c) 2024 MapsHD).
The algorithms, update rules, default values and Tait-Bryan conventions follow that code; the following
files are derived from it: `include/mini_lio/rgd_map.hpp` (`update_rgd_ray`), `include/mini_lio/optimizer.hpp`
(`optimize_lidar_odometry`, `compute_hessian`), `include/mini_lio/types.hpp` (`Core/transformations.h`),
`include/mini_lio/lio.hpp` (`compute_step_1`, `compute_step_2_impl`, `process_worker_step_update_rgd_after`).
Ported from commit `6f7622b83eb335c882255646a51de495a33d4d75` (2026-10-09).

Vendored in `ext/` (all MIT, license text next to each):

| dir | project | pinned commit |
|---|---|---|
| `ext/obs_eq` | JanuszBedkowski/observation_equations (one generated header) | 6850f190e380b763c0f6a4f93520e61e7c853aae |
| `ext/vqf` | dlaidig/vqf (C++ core) | b66ff218128ab0993b4ac19b538a4a57863c3382 |
| `ext/fusion` | xioTechnologies/Fusion (C core) | 53c67a79e0d08b82ffa975af28631dbb2a68529c |
| `ext/unordered_dense` | martinus/unordered_dense | 3234af2c03549bc85656bfd3a86993bf1cd8aef1 |

Eigen (>= 3.3) is expected from the system.

`tools/` (mini_lio_run) reads/writes the file formats of HDMapping step 1 (`load_data`, `load_imu`, `load_point_cloud`,
`save_result`, `LazWriter`); the code is a rewrite, not a copy. It links the system LASzip (`liblaszip-dev`,
not vendored; check its licence for redistribution of binaries). The tarball contains no LASzip code.
