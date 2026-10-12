// One chunk of IMU-rate poses optimized jointly against the NDT map
// (HDMapping optimize_lidar_odometry: point-to-distribution terms + motion model + fixed first pose, LM).
#pragma once

#include "rgd_map.hpp"
#include "types.hpp"

#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>

// generated point-to-NDT Jacobians (MIT, JanuszBedkowski/observation_equations); needs <cmath> + Eigen first
#include <point_to_point_source_to_target_tait_bryan_wc_jacobian_simplified.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mini_lio
{

using Mat6 = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;
using Mat6c = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

struct ChunkPoint
{
    Eigen::Vector3d p; // local (IMU) frame
    int pose; // index of the chunk pose that transforms it
};

// Histogram of map-normal directions (anisotropic weighting: dominant directions are down-weighted).
struct NvTables
{
    static constexpr size_t kSize = 101 * 101 * 101;
    std::vector<uint32_t> indoor = std::vector<uint32_t>(kSize, 0);
    std::vector<uint32_t> outdoor = std::vector<uint32_t>(kSize, 0);
    std::vector<int> touched_in, touched_out;
};

namespace detail
{

struct TbTrig
{
    TaitBryanPose pose;
    double sin_om, cos_om, sin_fi, cos_fi, sin_ka, cos_ka;
};

inline int nv_bin(double x, double y, double z)
{
    return static_cast<int>((x + 1.0) * 0.5 * 100.0 + (y + 1.0) * 0.5 * 100.0 * 100.0 + (z + 1.0) * 0.5 * 100.0 * 100.0 * 100.0);
}

inline double cauchy1(double d)
{
    return 1.0 / (M_PI * (1.0 + d * d));
}

inline void add_ndt(
    const Bucket& b, const Eigen::Vector3d& src, const TbTrig& t, const Eigen::Vector3d& viewport, bool use_vp,
    const std::vector<uint32_t>* table, Mat6& A, Vec6& g)
{
    const Eigen::Matrix3d& infm = b.cov_inverse;
    if ((infm.array() > 100000.0).any() || (infm.array() < -100000.0).any())
        return;
    if (use_vp && b.normal.dot(viewport - b.mean) < 0)
        return;

    Mat6 AtPA;
    point_to_point_source_to_target_tait_bryan_wc_AtPA_simplified_precomputed_trig(
        AtPA, t.pose.px, t.pose.py, t.pose.pz, t.sin_om, t.cos_om, t.sin_fi, t.cos_fi, t.sin_ka, t.cos_ka, src.x(), src.y(), src.z(),
        infm(0, 0), infm(0, 1), infm(0, 2), infm(1, 0), infm(1, 1), infm(1, 2), infm(2, 0), infm(2, 1), infm(2, 2));
    Vec6 AtPB;
    point_to_point_source_to_target_tait_bryan_wc_AtPB_simplified_precomputed_trig(
        AtPB, t.pose.px, t.pose.py, t.pose.pz, t.sin_om, t.cos_om, t.sin_fi, t.cos_fi, t.sin_ka, t.cos_ka, src.x(), src.y(), src.z(),
        infm(0, 0), infm(0, 1), infm(0, 2), infm(1, 0), infm(1, 1), infm(1, 2), infm(2, 0), infm(2, 1), infm(2, 2), b.mean.x(),
        b.mean.y(), b.mean.z());

    if (table)
    {
        const double x = b.normal.x(), y = b.normal.y(), z = b.normal.z();
        if (std::fabs(x) < 1.0 && std::fabs(y) < 1.0 && std::fabs(z) < 1.0)
        {
            const double w1 = static_cast<double>((*table)[nv_bin(x, y, z)]);
            if (w1 > 0.0)
            {
                AtPA /= w1;
                AtPB /= w1;
            }
        }
    }
    A += AtPA;
    g += AtPB;
}

// relative pose (translation + Tait-Bryan) of b seen from a
inline Vec6 rel6(const TaitBryanPose& a, const TaitBryanPose& b)
{
    const Eigen::Affine3d A = affine_from_tb(a), B = affine_from_tb(b);
    const TaitBryanPose t = tb_from_affine(A.inverse(Eigen::Isometry) * B);
    Vec6 v;
    v << t.px, t.py, t.pz, t.om, t.fi, t.ka;
    return v;
}

inline void bump(TaitBryanPose& p, int k, double h)
{
    switch (k)
    {
    case 0: p.px += h; break;
    case 1: p.py += h; break;
    case 2: p.pz += h; break;
    case 3: p.om += h; break;
    case 4: p.fi += h; break;
    default: p.ka += h; break;
    }
}

template <class F>
inline void parallel_chunks(size_t nc, bool mt, F&& f)
{
#ifdef _OPENMP
    if (mt)
    {
#pragma omp parallel for schedule(static)
        for (int c = 0; c < static_cast<int>(nc); ++c)
            f(static_cast<size_t>(c));
        return;
    }
#endif
    (void)mt;
    for (size_t c = 0; c < nc; ++c)
        f(c);
}

constexpr size_t kChunks = 16; // fixed => the reduction order (and the result) does not depend on thread count

} // namespace detail

inline void build_nv_tables(
    const std::vector<ChunkPoint>& pts, const std::vector<Eigen::Affine3d>& traj, const RgdMap& indoor, const RgdMap& outdoor,
    const Params& P, NvTables& T)
{
    for (int b : T.touched_in)
        T.indoor[b] = 0;
    for (int b : T.touched_out)
        T.outdoor[b] = 0;
    T.touched_in.clear();
    T.touched_out.clear();

    const size_t np = pts.size();
    const size_t nc = std::max<size_t>(1, std::min(detail::kChunks, np));
    const size_t csz = (np + nc - 1) / nc;
    const double max2 = P.max_distance * P.max_distance;
    std::vector<std::vector<int>> bin_in(nc), bin_out(nc);

    detail::parallel_chunks(
        nc, P.multithread,
        [&](size_t c)
        {
            const size_t b0 = c * csz, b1 = std::min(b0 + csz, np);
            for (size_t i = b0; i < b1; ++i)
            {
                const double r2 = pts[i].p.squaredNorm();
                if (r2 < 0.01 || r2 > max2)
                    continue;
                const Eigen::Vector3d pg = traj[pts[i].pose] * pts[i].p;
                if (const Bucket* b = indoor.find(pg))
                {
                    const double x = b->normal.x(), y = b->normal.y(), z = b->normal.z();
                    if (std::fabs(x) < 1.0 && std::fabs(y) < 1.0 && std::fabs(z) < 1.0)
                        bin_in[c].push_back(detail::nv_bin(x, y, z));
                }
                if (P.hierarchical && r2 >= 25.0)
                {
                    if (const Bucket* b = outdoor.find(pg))
                    {
                        const double x = b->normal.x(), y = b->normal.y(), z = b->normal.z();
                        if (std::fabs(x) < 1.0 && std::fabs(y) < 1.0 && std::fabs(z) < 1.0)
                            bin_out[c].push_back(detail::nv_bin(x, y, z));
                    }
                }
            }
        });

    for (size_t c = 0; c < nc; ++c)
    {
        for (int b : bin_in[c])
            if (T.indoor[b]++ == 0)
                T.touched_in.push_back(b);
        for (int b : bin_out[c])
            if (T.outdoor[b]++ == 0)
                T.touched_out.push_back(b);
    }
}

// One Levenberg-Marquardt step on all poses of the chunk. Returns delta = sum |dx| (1e6 on failure).
inline double optimize_chunk(
    const std::vector<ChunkPoint>& pts, std::vector<Eigen::Affine3d>& traj, const std::vector<Eigen::Affine3d>& traj_mm,
    const RgdMap& indoor, const RgdMap& outdoor, const Params& P, double lm, const NvTables& tables)
{
    using namespace detail;
    const size_t N = traj.size();

    std::vector<TbTrig> tb(N);
    for (size_t i = 0; i < N; ++i)
    {
        tb[i].pose = tb_from_affine(traj[i]);
        tb[i].sin_om = std::sin(tb[i].pose.om);
        tb[i].cos_om = std::cos(tb[i].pose.om);
        tb[i].sin_fi = std::sin(tb[i].pose.fi);
        tb[i].cos_fi = std::cos(tb[i].pose.fi);
        tb[i].sin_ka = std::sin(tb[i].pose.ka);
        tb[i].cos_ka = std::cos(tb[i].pose.ka);
    }

    // ---- point terms: Hessian blocks per pose, accumulated in fixed chunks ----
    const size_t np = pts.size();
    const size_t nc = std::max<size_t>(1, std::min(kChunks, np));
    const size_t csz = (np + nc - 1) / nc;
    const double max2 = P.max_distance * P.max_distance;
    std::vector<std::vector<Mat6>> cA(nc, std::vector<Mat6>(N, Mat6::Zero()));
    std::vector<std::vector<Vec6>> cG(nc, std::vector<Vec6>(N, Vec6::Zero()));

    parallel_chunks(
        nc, P.multithread,
        [&](size_t c)
        {
            const size_t b0 = c * csz, b1 = std::min(b0 + csz, np);
            for (size_t i = b0; i < b1; ++i)
            {
                const ChunkPoint& cp = pts[i];
                const double r2 = cp.p.squaredNorm();
                if (r2 < 0.01 || r2 > max2)
                    continue;
                const Eigen::Vector3d pg = traj[cp.pose] * cp.p;
                const Eigen::Vector3d vp = traj[cp.pose].translation();
                Mat6& A = cA[c][cp.pose];
                Vec6& g = cG[c][cp.pose];

                const Bucket* bi = indoor.find(pg);
                if (bi && bi->hits < 20)
                    add_ndt(*bi, cp.p, tb[cp.pose], vp, P.use_view_point_and_normal, P.use_anisotropic_weighting ? &tables.indoor : nullptr, A, g);

                if (P.hierarchical && r2 >= 25.0)
                {
                    const Bucket* bo = outdoor.find(pg);
                    if (bo && bo->hits < 20)
                        add_ndt(*bo, cp.p, tb[cp.pose], vp, P.use_view_point_and_normal, P.use_anisotropic_weighting ? &tables.outdoor : nullptr, A, g);
                }
            }
        });

    std::vector<Mat6c> D(N, Mat6c::Zero()); // diagonal blocks
    std::vector<Mat6c> E(N, Mat6c::Zero()); // E[i] = block (i-1, i)
    Eigen::VectorXd gtot = Eigen::VectorXd::Zero(6 * N);
    for (size_t c = 0; c < nc; ++c)
        for (size_t p = 0; p < N; ++p)
        {
            D[p] += cA[c][p];
            gtot.segment<6>(6 * p) += cG[c][p];
        }

    // ---- motion model: relative pose of consecutive poses stays at the predicted one ----
    const double wxyz = 1.0 / (P.mm_sigma_xyz * P.mm_sigma_xyz);
    const double sa = P.mm_sigma_angle_deg * kDegToRad;
    const double wang = 1.0 / (sa * sa);
    std::vector<TaitBryanPose> pose(N), des(N);
    for (size_t i = 0; i < N; ++i)
    {
        pose[i] = tb[i].pose;
        des[i] = tb_from_affine(traj_mm[i]);
    }
    constexpr double h = 1e-6;
    for (size_t i = 1; i < N; ++i)
    {
        const Vec6 meas = rel6(des[i - 1], des[i]);
        const Vec6 r = rel6(pose[i - 1], pose[i]) - meas;
        Eigen::Matrix<double, 6, 12> J;
        for (int k = 0; k < 12; ++k)
        {
            TaitBryanPose ap = pose[i - 1], bp = pose[i], am = pose[i - 1], bm = pose[i];
            if (k < 6)
            {
                bump(ap, k, h);
                bump(am, k, -h);
            }
            else
            {
                bump(bp, k - 6, h);
                bump(bm, k - 6, -h);
            }
            J.col(k) = (rel6(ap, bp) - rel6(am, bm)) / (2.0 * h);
        }
        Vec6 w;
        w << wxyz, wxyz, wxyz, wang * cauchy1(meas(3)), wang * cauchy1(meas(4)), wang * cauchy1(meas(5));
        const Eigen::Matrix<double, 12, 12> H = J.transpose() * w.asDiagonal() * J;
        const Eigen::Matrix<double, 12, 1> g = J.transpose() * (w.asDiagonal() * r);
        D[i - 1] += H.block<6, 6>(0, 0);
        D[i] += H.block<6, 6>(6, 6);
        E[i] = H.block<6, 6>(0, 6);
        gtot.segment<6>(6 * (i - 1)) += g.head<6>();
        gtot.segment<6>(6 * i) += g.tail<6>();
    }

    // ---- first pose is fixed; LM damping ----
    D[0].diagonal().array() += 1.0 / (P.fix_origin_sigma * P.fix_origin_sigma);
    for (size_t i = 0; i < N; ++i)
        D[i].diagonal().array() += lm;

    // ---- solve (block tridiagonal, sparse LDLT) ----
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(36 * (N + 2 * N));
    for (size_t i = 0; i < N; ++i)
    {
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                trip.emplace_back(6 * i + r, 6 * i + c, D[i](r, c));
        if (i > 0)
            for (int r = 0; r < 6; ++r)
                for (int c = 0; c < 6; ++c)
                {
                    trip.emplace_back(6 * (i - 1) + r, 6 * i + c, E[i](r, c));
                    trip.emplace_back(6 * i + c, 6 * (i - 1) + r, E[i](r, c));
                }
    }
    Eigen::SparseMatrix<double> A(6 * N, 6 * N);
    A.setFromTriplets(trip.begin(), trip.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
    if (solver.info() != Eigen::Success)
        return 1000000.0;
    const Eigen::VectorXd hx = solver.solve(-gtot);
    if (!hx.allFinite())
        return 1000000.0;

    // ---- update (a pose never moves more than 1 m away from its prediction) ----
    for (size_t i = 0; i < N; ++i)
    {
        TaitBryanPose p = tb_from_affine(traj[i]);
        const TaitBryanPose prev = tb_from_affine(traj_mm[i]);
        p.px += hx(6 * i + 0);
        p.py += hx(6 * i + 1);
        p.pz += hx(6 * i + 2);
        p.om += hx(6 * i + 3);
        p.fi += hx(6 * i + 4);
        p.ka += hx(6 * i + 5);
        if ((Eigen::Vector3d(prev.px, prev.py, prev.pz) - Eigen::Vector3d(p.px, p.py, p.pz)).norm() < 1.0)
            traj[i] = affine_from_tb(p);
    }
    return hx.cwiseAbs().sum();
}

} // namespace mini_lio
