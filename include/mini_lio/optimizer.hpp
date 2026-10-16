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

// Per point of the chunk, the map cells it falls in at the current poses (null: none / not used).
// Kept up to date by lookup_cells() each iteration and used by optimize_chunk(): the map does not
// change while a chunk is optimized.
struct PointCells
{
    const Bucket* indoor = nullptr;
    const Bucket* outdoor = nullptr;
    uint64_t key_in = ~0ull, key_out = ~0ull; // cell keys of the last lookup (~0: none)
    int nv_in = -1, nv_out = -1; // their normal-histogram bins (-1: none)
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

// One point-to-distribution term: A += J^T P J, g += J^T P r for the source point `src` (local
// frame) against the cell's mean and inverse covariance P. The same as the generated
// point_to_point_source_to_target_tait_bryan_wc_AtPA/AtPB_simplified_precomputed_trig, using
// their structure: J = [I3 | a b c] (a, b, c = derivatives of R(om, fi, ka) * src), so the blocks
// are P, P [a b c] and [a b c]^T P [a b c] - about half the flops, 2x faster per term on a
// Raspberry Pi 5 - and the anisotropic weight scales P once instead of dividing the 42 outputs. `nv_bin` is the cell's histogram bin
// (PointCells::nv_in / nv_out, -1: none); `table` null: no anisotropic weighting.
inline void add_ndt(
    const Bucket& b, const Eigen::Vector3d& src, const TbTrig& t, const Eigen::Vector3d& viewport, bool use_vp,
    const std::vector<uint32_t>* table, int nv_bin, Mat6& A, Vec6& g)
{
    const Eigen::Matrix3d& infm = b.cov_inverse;
    if ((infm.array() > 100000.0).any() || (infm.array() < -100000.0).any())
        return;
    if (use_vp && b.normal.dot(viewport - b.mean) < 0)
        return;

    Eigen::Matrix3d P = infm;
    if (table && nv_bin >= 0)
    {
        const double w1 = static_cast<double>((*table)[nv_bin]);
        if (w1 > 0.0)
            P *= 1.0 / w1;
    }

    // the generated code's subexpressions (same names), see the header above
    const double x_s = src.x(), y_s = src.y(), z_s = src.z();
    const double x0 = t.cos_ka * t.sin_om, x1 = t.cos_om * t.sin_ka;
    const double x2 = t.sin_fi * x1 + x0;
    const double x3 = t.sin_ka * t.sin_om, x4 = t.cos_ka * t.cos_om;
    const double x5 = t.sin_fi * x4 - x3;
    const double x6 = t.cos_fi * z_s;
    const double x7 = t.cos_om * x6 + x2 * y_s - x5 * x_s; // (R src).z
    const double x8 = t.sin_fi * x0 + x1;
    const double x9 = -t.sin_fi * x3 + x4;
    const double x10 = -t.sin_om * x6 + x8 * x_s + x9 * y_s; // (R src).y
    const double x11 = t.sin_ka * y_s, x12 = t.cos_ka * x_s;
    const double x13 = t.sin_fi * x11 - t.sin_fi * x12 + x6;
    const double x15 = -t.cos_fi * x11 + t.cos_fi * x12 + t.sin_fi * z_s; // (R src).x
    const double x16 = t.sin_om * x15, x17 = t.cos_om * x15;
    const double x18 = -x8 * y_s + x9 * x_s;
    const double x19 = x2 * x_s + x5 * y_s;
    const double x20 = t.cos_fi * (t.cos_ka * y_s + t.sin_ka * x_s);

    Eigen::Matrix3d Jr; // columns a, b, c: d(R src)/d(om, fi, ka)
    Jr << 0.0, x13, -x20,
          -x7, x16, x18,
          x10, -x17, x19;
    const Eigen::Vector3d r(t.pose.px + x15 - b.mean.x(), t.pose.py + x10 - b.mean.y(), t.pose.pz + x7 - b.mean.z());

    const Eigen::Matrix3d PJ = P * Jr;
    const Eigen::Vector3d Pr = P * r;
    // built locally, then one contiguous add (block-wise adds into the strided 6x6 were slower)
    Mat6 H;
    H.topLeftCorner<3, 3>() = P;
    H.topRightCorner<3, 3>() = PJ;
    H.bottomLeftCorner<3, 3>() = PJ.transpose();   // = Jr^T P (P symmetric up to rounding)
    H.bottomRightCorner<3, 3>().noalias() = Jr.transpose() * PJ;
    Vec6 G;
    G.head<3>() = Pr;
    G.tail<3>().noalias() = Jr.transpose() * Pr;
    A += H;
    g += G;
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

// `threads` > 0: that many OpenMP threads, else OpenMP's default (all cores)
template <class F>
inline void parallel_chunks(size_t nc, bool mt, int threads, F&& f)
{
#ifdef _OPENMP
    if (mt)
    {
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : omp_get_max_threads())
        for (int c = 0; c < static_cast<int>(nc); ++c)
            f(static_cast<size_t>(c));
        return;
    }
#endif
    (void)mt;
    (void)threads;
    for (size_t c = 0; c < nc; ++c)
        f(c);
}

constexpr size_t kChunks = 16; // fixed => the reduction order (and the result) does not depend on thread count

} // namespace detail

// The cells of every point at the current poses (points outside (0.1 m, max_distance) get none),
// and the normal histograms of those cells (anisotropic weighting, `T` non-null).
// With `again` (same points, same map, poses moved by an iteration) only points whose cell key
// changed are looked up again, and the histograms are adjusted for just those points: after the
// first iteration that is a small fraction, and the hash lookups were the expensive part. The
// result is the same as a full rebuild.
inline void lookup_cells(
    const std::vector<ChunkPoint>& pts, const std::vector<Eigen::Affine3d>& traj, const RgdMap& indoor, const RgdMap& outdoor,
    const Params& P, std::vector<PointCells>& cells, NvTables* T, bool again = false)
{
    const size_t np = pts.size();
    if (cells.size() != np)
        again = false;
    if (!again)
        cells.assign(np, PointCells{});
    const size_t nc = std::max<size_t>(1, std::min(detail::kChunks, np));
    const size_t csz = (np + nc - 1) / nc;
    const double max2 = P.max_distance * P.max_distance;
    const auto bin_of = [](const Bucket* b)
    {
        if (!b)
            return -1;
        const double x = b->normal.x(), y = b->normal.y(), z = b->normal.z();
        return std::fabs(x) < 1.0 && std::fabs(y) < 1.0 && std::fabs(z) < 1.0 ? detail::nv_bin(x, y, z) : -1;
    };
    // histogram changes per block: (old bin, new bin), -1 = none
    std::vector<std::vector<std::pair<int, int>>> ch_in(again ? nc : 0), ch_out(again ? nc : 0);
    detail::parallel_chunks(
        nc, P.multithread, P.threads,
        [&](size_t c)
        {
            const size_t b0 = c * csz, b1 = std::min(b0 + csz, np);
            for (size_t i = b0; i < b1; ++i)
            {
                const double r2 = pts[i].p.squaredNorm();
                if (r2 < 0.01 || r2 > max2)
                    continue;
                const Eigen::Vector3d pg = traj[pts[i].pose] * pts[i].p;
                PointCells& pc = cells[i];
                const uint64_t ki = indoor.key(pg);
                if (ki != pc.key_in)
                {
                    pc.key_in = ki;
                    pc.indoor = indoor.findKey(ki);
                    const int nb = bin_of(pc.indoor);
                    if (again && nb != pc.nv_in)
                        ch_in[c].emplace_back(pc.nv_in, nb);
                    pc.nv_in = nb;
                }
                if (P.hierarchical && r2 >= 25.0)
                {
                    const uint64_t ko = outdoor.key(pg);
                    if (ko != pc.key_out)
                    {
                        pc.key_out = ko;
                        pc.outdoor = outdoor.findKey(ko);
                        const int nb = bin_of(pc.outdoor);
                        if (again && nb != pc.nv_out)
                            ch_out[c].emplace_back(pc.nv_out, nb);
                        pc.nv_out = nb;
                    }
                }
            }
        });
    if (!T)
        return;
    const auto inc = [](int bin, std::vector<uint32_t>& table, std::vector<int>& touched)
    {
        if (table[bin]++ == 0)
            touched.push_back(bin); // a bin may be listed twice: harmless, the list only resets
    };
    if (again)
    {
        for (size_t c = 0; c < ch_in.size(); ++c)
        {
            for (const auto& [o, n] : ch_in[c])
            {
                if (o >= 0)
                    --T->indoor[o];
                if (n >= 0)
                    inc(n, T->indoor, T->touched_in);
            }
            for (const auto& [o, n] : ch_out[c])
            {
                if (o >= 0)
                    --T->outdoor[o];
                if (n >= 0)
                    inc(n, T->outdoor, T->touched_out);
            }
        }
        return;
    }
    for (int b : T->touched_in)
        T->indoor[b] = 0;
    for (int b : T->touched_out)
        T->outdoor[b] = 0;
    T->touched_in.clear();
    T->touched_out.clear();
    for (const PointCells& pc : cells)
    {
        if (pc.nv_in >= 0)
            inc(pc.nv_in, T->indoor, T->touched_in);
        if (pc.nv_out >= 0)
            inc(pc.nv_out, T->outdoor, T->touched_out);
    }
}

// One Levenberg-Marquardt step on all poses of the chunk. Returns delta = sum |dx| (1e6 on failure).
inline double optimize_chunk(
    const std::vector<ChunkPoint>& pts, const std::vector<PointCells>& cells, std::vector<Eigen::Affine3d>& traj,
    const std::vector<Eigen::Affine3d>& traj_mm, const Params& P, double lm, const NvTables& tables)
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
    std::vector<std::vector<Mat6>> cA(nc, std::vector<Mat6>(N, Mat6::Zero()));
    std::vector<std::vector<Vec6>> cG(nc, std::vector<Vec6>(N, Vec6::Zero()));

    parallel_chunks(
        nc, P.multithread, P.threads,
        [&](size_t c)
        {
            const size_t b0 = c * csz, b1 = std::min(b0 + csz, np);
            for (size_t i = b0; i < b1; ++i)
            {
                const ChunkPoint& cp = pts[i];
                const Bucket* bi = cells[i].indoor;
                const Bucket* bo = cells[i].outdoor;
                if (!bi && !bo)
                    continue;
                const Eigen::Vector3d vp = traj[cp.pose].translation();
                Mat6& A = cA[c][cp.pose];
                Vec6& g = cG[c][cp.pose];

                if (bi && bi->hits < 20)
                    add_ndt(*bi, cp.p, tb[cp.pose], vp, P.use_view_point_and_normal, P.use_anisotropic_weighting ? &tables.indoor : nullptr,
                            cells[i].nv_in, A, g);
                if (bo && bo->hits < 20)
                    add_ndt(*bo, cp.p, tb[cp.pose], vp, P.use_view_point_and_normal, P.use_anisotropic_weighting ? &tables.outdoor : nullptr,
                            cells[i].nv_out, A, g);
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
