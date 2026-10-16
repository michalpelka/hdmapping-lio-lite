// mini_lio: minimal LiDAR-inertial odometry, ported from HDMapping step 1
// (MapsHD/HDMapping, apps/lidar_odometry_step_1, MIT). See NOTICE.md.
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <vector>

namespace mini_lio
{

constexpr double kDegToRad = M_PI / 180.0;

// Input sample. Same clock as the LiDAR points.
struct ImuSample
{
    double t = 0.0; // [s]
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero(); // [rad/s]
    Eigen::Vector3d acc = Eigen::Vector3d::Zero(); // [g], +1 g along "up" at rest
};

// Input point, already in the IMU frame (apply lidar->imu extrinsics before).
// Points must arrive in non-decreasing time order.
struct LidarPoint
{
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    double t = 0.0; // [s]
    uint16_t intensity = 0; // passed through to ChunkResult::output_points only
};

// A raw (not decimated) chunk point for export: IMU-frame position, original timestamp, intensity and the
// index of the chunk pose that transforms it.
struct OutputPoint
{
    Eigen::Vector3d p;
    double t;
    uint16_t intensity;
    int pose;
};

struct PoseStamped
{
    double t = 0.0;
    Eigen::Affine3d T = Eigen::Affine3d::Identity(); // sensor -> world
};

struct TaitBryanPose
{
    double px = 0, py = 0, pz = 0, om = 0, fi = 0, ka = 0;
};

// Same conventions as HDMapping Core/transformations.h (R = Rx(om) Ry(fi) Rz(ka), "wc" pose).
inline TaitBryanPose tb_from_affine(const Eigen::Affine3d& m)
{
    TaitBryanPose pose;
    pose.px = m(0, 3);
    pose.py = m(1, 3);
    pose.pz = m(2, 3);

    if (m(0, 2) < 1)
    {
        if (m(0, 2) > -1)
        {
            pose.fi = std::asin(m(0, 2));
            pose.om = std::atan2(-m(1, 2), m(2, 2));
            pose.ka = std::atan2(-m(0, 1), m(0, 0));
            return pose;
        }
        pose.fi = -M_PI / 2.0;
        pose.om = -std::atan2(m(1, 0), m(1, 1));
        pose.ka = 0;
        return pose;
    }
    pose.fi = M_PI / 2.0;
    pose.om = std::atan2(m(1, 0), m(1, 1));
    pose.ka = 0.0;
    return pose;
}

inline Eigen::Affine3d affine_from_tb(const TaitBryanPose& pose)
{
    Eigen::Affine3d m = Eigen::Affine3d::Identity();
    const double sx = std::sin(pose.om), cx = std::cos(pose.om);
    const double sy = std::sin(pose.fi), cy = std::cos(pose.fi);
    const double sz = std::sin(pose.ka), cz = std::cos(pose.ka);

    m(0, 0) = cy * cz;
    m(1, 0) = cz * sx * sy + cx * sz;
    m(2, 0) = -cx * cz * sy + sx * sz;

    m(0, 1) = -cy * sz;
    m(1, 1) = cx * cz - sx * sy * sz;
    m(2, 1) = cz * sx + cx * sy * sz;

    m(0, 2) = sy;
    m(1, 2) = -cy * sx;
    m(2, 2) = cx * cy;

    m(0, 3) = pose.px;
    m(1, 3) = pose.py;
    m(2, 3) = pose.pz;
    return m;
}

struct Params
{
    bool collect_output_points = false; // fill ChunkResult::output_points (raw, non-decimated points of the chunk)
    // ---- AHRS (orientation prior from the IMU) ----
    bool use_vqf = false; // false: Fusion (Madgwick), true: VQF
    double imu_rate_hz = 200.0; // VQF needs a fixed sample time
    double fusion_gain = 0.5;
    // Fusion only: median gyro bias of the first 1000 samples (assumes the device starts still)
    bool fusion_remove_gyro_bias = false;
    double vqf_tau_acc = 0.5;
    bool vqf_motion_bias_est = true;
    bool vqf_rest_bias_est = true;
    double vqf_bias_sigma_init = 0.5;
    double vqf_bias_forgetting_time = 100.0;
    double vqf_bias_clip = 2.0;
    double vqf_bias_sigma_motion = 0.1;
    double vqf_bias_vertical_forgetting = 0.0001;
    double vqf_bias_sigma_rest = 0.03;
    double vqf_rest_min_t = 1.5;
    double vqf_rest_filter_tau = 0.5;
    double vqf_rest_th_gyr = 2.0;
    double vqf_rest_th_acc = 0.5;

    // ---- point preprocessing ----
    double min_range_xy = 0.3; // load-time filter: min_range_xy < |xy| < max_range_xy
    double max_range_xy = 70.0;
    double decimation = 0.01; // voxel size, first point per voxel is kept (0 = off)
    int initial_points = 10000; // points used to seed the map (first chunk starts after them)
    int min_points_per_chunk = 1000; // fewer: chunk is propagated by the motion model, not optimized
    bool correct_point_timestamps = true; // re-spread timestamps uniformly over the chunk (as upstream)

    // ---- chunking ----
    int poses_per_chunk = 20; // IMU-rate poses optimized jointly (20 @ 200 Hz = 0.1 s)

    // ---- map ----
    bool hierarchical = true; // indoor + outdoor grids
    double indoor_resolution = 0.1; // [m]
    double outdoor_resolution = 0.3; // [m]
    double max_distance = 70.0; // points further than this (3D range) are ignored
    double sliding_window_distance = 200.0; // [m] rebuild the map after this much travel
    double window_seconds = 0.0; // > 0: rebuild by time instead (map covers ~window/2 .. window)

    // ---- optimization ----
    int nr_iter = 100;
    double convergence_delta = 1e-12;
    double lm_damping = 10.0; // lm_factor *= lm_damping every 10 iterations
    double max_time_per_chunk_s = 10.0;
    bool use_view_point_and_normal = true;
    bool use_anisotropic_weighting = true;
    bool use_motion_from_previous_step = true; // constant velocity prediction
    bool multithread = true;
    // OpenMP threads for the per-point work (0: OpenMP's default, all cores). The parallel regions
    // are short and memory-bound: on a Raspberry Pi 5, 2 threads are fastest (~7.6 ms per JT128
    // chunk vs ~9 with 4), and they leave cores for the lidar driver.
    int threads = 0;

    // motion model (relative pose between consecutive IMU poses) and fixed first pose
    double mm_sigma_xyz = 0.0005; // [m]
    double mm_sigma_angle_deg = 0.01;
    double fix_origin_sigma = 0.000001; // all 6 DoF
};

} // namespace mini_lio
