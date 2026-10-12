// Differential test: upstream HDMapping code (extracted verbatim, TBB/spdlog shimmed) vs mini_lio.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ankerl/unordered_dense.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// ---------------- shims for what upstream pulls in ----------------
namespace tbb
{
template <class F>
void parallel_for(size_t a, size_t b, F f)
{
    for (size_t i = a; i < b; ++i)
        f(i);
}
template <class T>
struct combinable
{
    T v;
    T& local() { return v; }
    template <class F>
    void combine_each(F f)
    {
        f(v);
    }
};
} // namespace tbb
namespace spdlog
{
template <class... A>
void warn(const char*, A&&...)
{
}
template <class... A>
void info(const char*, A&&...)
{
}
} // namespace spdlog
#define HDMAP_ZONE_SCOPE(name)
#define HDMAP_ZONE_BEGIN(id, name)
#define HDMAP_ZONE_END(id)

#include <Core/structures.h> // real HDMapping header: Point3Di, TaitBryanPose
#include <Core/transformations.h> // real HDMapping header: pose_tait_bryan_from_affine_matrix, affine_matrix_from_pose_tait_bryan

#include <common/include/cauchy.h>
#include <python-scripts/constraints/relative_pose_tait_bryan_wc_jacobian.h>
#include <python-scripts/point-to-point-metrics/point_to_point_source_to_target_tait_bryan_wc_jacobian_simplified.h>

// copied from Core/ndt.h (the real header drags in session.h)
namespace NDT
{
struct GridParameters
{
    double bounding_box_min_X, bounding_box_min_Y, bounding_box_min_Z;
    double bounding_box_max_X, bounding_box_max_Y, bounding_box_max_Z;
    double bounding_box_extension;
    int number_of_buckets_X, number_of_buckets_Y, number_of_buckets_Z;
    uint64_t number_of_buckets;
    double resolution_X, resolution_Y, resolution_Z;
};
struct Bucket
{
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d cov_inverse = Eigen::Matrix3d::Zero();
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_vector = Eigen::Vector3d::Zero();
    uint64_t number_of_points = 0;
    uint64_t index_begin = 0;
    uint64_t index_end = 0;
    char number_of_hits = 0;
};
} // namespace NDT
using NDTBucketMapType = ankerl::unordered_dense::map<uint64_t, NDT::Bucket>;
struct LookupStats
{
    size_t indoor_lookups = 0;
    size_t outdoor_lookups = 0;
};

// copied from core/src/hash_utils.cpp
static uint64_t get_index_3d(const int16_t x, const int16_t y, const int16_t z)
{
    static constexpr auto shift_x = 16 * 2;
    static constexpr auto shift_y = 16 * 1;
    static constexpr auto shift_z = 16 * 0;
    static constexpr auto mask_x = 0x0000FFFF00000000ull;
    static constexpr auto mask_y = 0x00000000FFFF0000ull;
    static constexpr auto mask_z = 0x000000000000FFFFull;
    const auto x_part = ((static_cast<uint64_t>(x) << shift_x) & (mask_x));
    const auto y_part = ((static_cast<uint64_t>(y) << shift_y) & (mask_y));
    const auto z_part = ((static_cast<uint64_t>(z) << shift_z) & (mask_z));
    return x_part | y_part | z_part;
}
static uint64_t get_rgd_index_3d(const Eigen::Vector3d& p, const Eigen::Vector3d& b)
{
    const int16_t x = static_cast<int16_t>(p.x() / b.x());
    const int16_t y = static_cast<int16_t>(p.y() / b.y());
    const int16_t z = static_cast<int16_t>(p.z() / b.z());
    return get_index_3d(x, y, z);
}

// ---------------- upstream, verbatim ----------------
#include "upstream_rgd.inc" // update_rgd_ray
#include "upstream_opt.inc" // optimize_lidar_odometry + helpers

// upstream's anisotropic-weighting histogram builder (lambda inside process_worker_step_lidar_odometry_core), verbatim
#include <thread>
struct FakeWD
{
    std::vector<Eigen::Affine3d> intermediate_trajectory;
};
struct FakeParams
{
    NDT::GridParameters in_out_params_indoor{}, in_out_params_outdoor{};
    NDTBucketMapType buckets_indoor, buckets_outdoor;
    bool ablation_study_use_hierarchical_rgd = true;
    bool useMultithread = false;
    double max_distance_lidar = 70.0;
};
static void upstream_hist(
    FakeParams& params, FakeWD& worker_data, std::vector<Point3Di>& intermediate_points, std::vector<double>& out_in,
    std::vector<double>& out_out)
{
#include "upstream_hist.inc"
    build_normal_vector_histograms();
    out_in = table_buckets_nv_indoor;
    out_out = table_buckets_nv_outdoor;
}

// ---------------- mini_lio ----------------
#include <mini_lio/optimizer.hpp>
#include <mini_lio/rgd_map.hpp>

namespace
{

std::mt19937 rng(1234);
double U(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
double N01() { return std::normal_distribution<double>(0, 1)(rng); }

// points on the 6 walls of a 10x8x3 room + 2 boxes, seen from `o` along random directions
bool hit_room(const Eigen::Vector3d& o, const Eigen::Vector3d& d, double& r)
{
    const Eigen::Vector3d mx(10, 8, 3);
    double best = 1e9;
    for (int a = 0; a < 3; ++a)
        if (std::fabs(d[a]) > 1e-12)
            best = std::min(best, d[a] > 0 ? (mx[a] - o[a]) / d[a] : (0 - o[a]) / d[a]);
    r = best;
    return best < 60;
}

struct Stat
{
    double max_abs = 0;
    void add(double v) { max_abs = std::max(max_abs, std::fabs(v)); }
};

} // namespace

int main()
{
    int failures = 0;
    auto check = [&](const char* name, double err, double tol)
    {
        const bool ok = err <= tol;
        std::printf("  %-52s max err %.3e  (tol %.0e)  %s\n", name, err, tol, ok ? "OK" : "FAIL");
        if (!ok)
            ++failures;
    };

    // ======== Test A: RGD map update ========
    std::printf("A. RGD map update (upstream update_rgd_ray vs mini_lio::RgdMap)\n");
    for (int ray = 0; ray < 2; ++ray)
    {
        std::vector<Point3Di> up_pts;
        std::vector<Eigen::Vector3d> my_pts;
        const Eigen::Vector3d vp(5.0, 4.0, 1.4);
        for (int i = 0; i < 40000; ++i)
        {
            Eigen::Vector3d d(N01(), N01(), N01() * 0.4);
            d.normalize();
            double r;
            hit_room(vp, d, r);
            const Eigen::Vector3d p = vp + d * (r + 0.01 * N01());
            Point3Di pi{};
            pi.point = p;
            up_pts.push_back(pi);
            my_pts.push_back(p);
        }
        NDT::GridParameters gp{};
        gp.resolution_X = gp.resolution_Y = gp.resolution_Z = 0.1;
        NDTBucketMapType up;
        // two successive updates from different viewpoints (exercises the n>3 viewpoint branch and hits)
        update_rgd_ray(gp, up, up_pts, vp, nullptr, ray == 1);
        update_rgd_ray(gp, up, up_pts, Eigen::Vector3d(4, 3, 1.2), nullptr, ray == 1);
        mini_lio::RgdMap mine(0.1);
        mine.update(my_pts, vp, ray == 1);
        mine.update(my_pts, Eigen::Vector3d(4, 3, 1.2), ray == 1);

        Stat s_mean, s_cov, s_inv, s_nv, s_n, s_hits;
        size_t missing = 0;
        for (const auto& [k, ub] : up)
        {
            const auto it = mine.cells().find(k);
            if (it == mine.cells().end())
            {
                ++missing;
                continue;
            }
            const auto& mb = it->second;
            s_mean.add((ub.mean - mb.mean).cwiseAbs().maxCoeff());
            s_cov.add((ub.cov - mb.cov).cwiseAbs().maxCoeff());
            s_inv.add((ub.cov_inverse - mb.cov_inverse).cwiseAbs().maxCoeff() / std::max(1.0, ub.cov_inverse.cwiseAbs().maxCoeff()));
            s_nv.add((ub.normal_vector - mb.normal).cwiseAbs().maxCoeff());
            s_n.add(static_cast<double>(ub.number_of_points) - static_cast<double>(mb.n));
            s_hits.add(static_cast<double>(ub.number_of_hits) - static_cast<double>(mb.hits));
        }
        std::printf("  ray_cast=%d: %zu upstream cells, %zu mini cells, %zu missing\n", ray, up.size(), mine.cells().size(), missing);
        check("cell set (missing + extra)", static_cast<double>(missing) + std::fabs(double(up.size()) - double(mine.cells().size())), 0);
        check("mean", s_mean.max_abs, 0);
        check("cov", s_cov.max_abs, 1e-15);
        check("cov_inverse (relative)", s_inv.max_abs, 1e-12);
        check("normal", s_nv.max_abs, 1e-15);
        check("number_of_points", s_n.max_abs, 0);
        check("number_of_hits", s_hits.max_abs, 0);
    }

    // ======== Test B: one chunk, several LM iterations ========
    std::printf("B. optimize_lidar_odometry (upstream) vs mini_lio::optimize_chunk, 20 poses, perturbed start\n");
    const int Np = 20;
    // ground-truth chunk path: 0.1 s at ~1 m/s, small rotation
    std::vector<Eigen::Affine3d> gt(Np);
    for (int i = 0; i < Np; ++i)
    {
        mini_lio::TaitBryanPose tb;
        tb.px = 3.0 + 0.005 * i;
        tb.py = 2.5 + 0.002 * i;
        tb.pz = 1.4;
        tb.om = 0.02 * std::sin(0.1 * i);
        tb.fi = 0.01 * i * 0.05;
        tb.ka = 0.3 + 0.0015 * i;
        gt[i] = mini_lio::affine_from_tb(tb);
    }
    // map built from a previous, different scan (so the chunk has a real registration problem)
    NDT::GridParameters gin{}, gout{};
    gin.resolution_X = gin.resolution_Y = gin.resolution_Z = 0.1;
    gout.resolution_X = gout.resolution_Y = gout.resolution_Z = 0.3;
    NDTBucketMapType up_in, up_out;
    mini_lio::RgdMap my_in(0.1), my_out(0.3);
    {
        std::vector<Point3Di> pm;
        std::vector<Eigen::Vector3d> pmine;
        const Eigen::Vector3d vp(3.0, 2.5, 1.4);
        for (int i = 0; i < 60000; ++i)
        {
            Eigen::Vector3d d(N01(), N01(), N01() * 0.5);
            d.normalize();
            double r;
            hit_room(vp, d, r);
            const Eigen::Vector3d p = vp + d * (r + 0.01 * N01());
            Point3Di pi{};
            pi.point = p;
            pm.push_back(pi);
            pmine.push_back(p);
        }
        update_rgd_ray(gin, up_in, pm, vp, nullptr, true);
        update_rgd_ray(gout, up_out, pm, vp, nullptr, true);
        my_in.update(pmine, vp, true);
        my_out.update(pmine, vp, true);
    }
    // chunk points (local frame), each tied to a pose
    std::vector<Point3Di> up_pts;
    std::vector<mini_lio::ChunkPoint> my_pts;
    for (int i = 0; i < 12000; ++i)
    {
        const int pi_ = i % Np;
        Eigen::Vector3d d(N01(), N01(), N01() * 0.5);
        d.normalize();
        const Eigen::Vector3d o = gt[pi_].translation();
        const Eigen::Vector3d dw = gt[pi_].linear() * d;
        double r;
        hit_room(o, dw, r);
        const Eigen::Vector3d local = d * (r + 0.01 * N01());
        Point3Di pt{};
        pt.point = local;
        pt.index_pose = pi_;
        up_pts.push_back(pt);
        my_pts.push_back({ local, pi_ });
    }

    // perturbed initial trajectory = also the motion-model prediction (as in the driver)
    std::vector<Eigen::Affine3d> start(Np);
    for (int i = 0; i < Np; ++i)
    {
        Eigen::Affine3d e = Eigen::Affine3d::Identity();
        e.linear() = Eigen::AngleAxisd(0.004, Eigen::Vector3d(0.3, 0.5, 0.8).normalized()).toRotationMatrix();
        e.translation() = Eigen::Vector3d(0.03, -0.02, 0.015);
        start[i] = e * gt[i];
    }
    start[0] = gt[0]; // chunk starts at a known pose, as in the driver

    mini_lio::Params P; // defaults mirror upstream defaults
    P.multithread = false;

    auto up_traj = start, up_mm = start;
    auto my_traj = start, my_mm = start;
    mini_lio::NvTables tables;
    std::vector<double> t_in(101 * 101 * 101, 0.0), t_out(101 * 101 * 101, 0.0);

    double lm = 1.0;
    for (int it = 0; it < 12; ++it)
    {
        // anisotropic tables from the CURRENT trajectory (mini's builder), shared by both
        mini_lio::build_nv_tables(my_pts, my_traj, my_in, my_out, P, tables);
        std::fill(t_in.begin(), t_in.end(), 0.0);
        std::fill(t_out.begin(), t_out.end(), 0.0);
        for (size_t i = 0; i < tables.indoor.size(); ++i)
            t_in[i] = tables.indoor[i];
        for (size_t i = 0; i < tables.outdoor.size(); ++i)
            t_out[i] = tables.outdoor[i];

        double up_delta = 100000.0;
        LookupStats ls;
        const bool thr_flag = false;
        const double conv_res = 0.0, conv_thr = 1e-6;
        TaitBryanPose zero_corr{};
        optimize_lidar_odometry(
            up_pts, up_traj, up_mm, gin, up_in, gout, up_out, false, P.max_distance, up_delta, lm, zero_corr, P.mm_sigma_xyz, P.mm_sigma_xyz,
            P.mm_sigma_xyz, P.mm_sigma_angle_deg, P.mm_sigma_angle_deg, P.mm_sigma_angle_deg, P.fix_origin_sigma, P.fix_origin_sigma,
            P.fix_origin_sigma, P.fix_origin_sigma, P.fix_origin_sigma, P.fix_origin_sigma, false, false, true, true, true, ls, thr_flag, conv_res,
            conv_thr, t_in, t_out);

        const double my_delta = mini_lio::optimize_chunk(my_pts, my_traj, my_mm, my_in, my_out, P, lm, tables);

        double dpos = 0, drot = 0;
        for (int i = 0; i < Np; ++i)
        {
            dpos = std::max(dpos, (up_traj[i].translation() - my_traj[i].translation()).norm());
            drot = std::max(drot, (up_traj[i].linear() - my_traj[i].linear()).cwiseAbs().maxCoeff());
        }
        std::printf("  iter %2d: upstream delta %.6e, mini delta %.6e | max pose diff: pos %.3e m, rot %.3e\n", it + 1, up_delta, my_delta, dpos, drot);
        if (it == 0 || it == 11)
        {
            check("trajectory position", dpos, 1e-6);
            check("trajectory rotation matrix", drot, 1e-6);
            check("delta (relative)", std::fabs(up_delta - my_delta) / std::max(1e-12, up_delta), 1e-3);
        }
        if (it % 10 == 0 && it > 0)
            lm *= P.lm_damping;
    }

    // residual vs ground truth after the iterations (sanity: both converge to the same place)
    double e_up = 0, e_my = 0, e_start = 0;
    for (int i = 0; i < Np; ++i)
    {
        e_up = std::max(e_up, (up_traj[i].translation() - gt[i].translation()).norm());
        e_my = std::max(e_my, (my_traj[i].translation() - gt[i].translation()).norm());
        e_start = std::max(e_start, (start[i].translation() - gt[i].translation()).norm());
    }
    std::printf("  position error vs ground truth: start %.4f m -> upstream %.4f m, mini %.4f m\n", e_start, e_up, e_my);

    // ======== Test C: anisotropic-weighting histograms ========
    std::printf("C. normal-vector histograms (upstream lambda vs mini_lio::build_nv_tables)\n");
    {
        FakeParams fp;
        fp.in_out_params_indoor = gin;
        fp.in_out_params_outdoor = gout;
        fp.buckets_indoor = up_in;
        fp.buckets_outdoor = up_out;
        FakeWD fwd;
        fwd.intermediate_trajectory = start;
        std::vector<double> h_in, h_out;
        upstream_hist(fp, fwd, up_pts, h_in, h_out);
        mini_lio::NvTables nt;
        mini_lio::build_nv_tables(my_pts, start, my_in, my_out, P, nt);
        double d_in = 0, d_out = 0, total_in = 0, total_out = 0;
        for (size_t i = 0; i < h_in.size(); ++i)
        {
            d_in = std::max(d_in, std::fabs(h_in[i] - static_cast<double>(nt.indoor[i])));
            d_out = std::max(d_out, std::fabs(h_out[i] - static_cast<double>(nt.outdoor[i])));
            total_in += h_in[i];
            total_out += h_out[i];
        }
        std::printf("  histogram mass: indoor %.0f, outdoor %.0f points\n", total_in, total_out);
        check("indoor histogram", d_in, 0);
        check("outdoor histogram", d_out, 0);
    }

    std::printf("\n%s (%d failed checks)\n", failures ? "FAILED" : "ALL EQUIVALENCE CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
