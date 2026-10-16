// Streaming driver: IMU + LiDAR in, IMU-rate poses and a rolling NDT map out.
// Follows HDMapping step 1 (compute_step_1 chunking + compute_step_2_impl odometry loop), without files or GUI.
#pragma once

#include "ahrs.hpp"
#include "optimizer.hpp"
#include "rgd_map.hpp"
#include "types.hpp"

#include <ankerl/unordered_dense.h>

#include <chrono>
#include <deque>
#include <functional>

namespace mini_lio
{

struct ChunkResult
{
    std::vector<PoseStamped> poses; // optimized IMU-rate poses of the chunk
    std::vector<Eigen::Vector3d> points_global; // the chunk's (decimated) points in the world frame
    std::vector<OutputPoint> output_points; // raw points of the chunk, only if Params::collect_output_points
    int iterations = 0;
    double delta = 0.0; // sum |dx| of the last iteration
    double ms = 0.0; // wall time of the chunk
    bool optimized = false; // false: too few points, poses are the motion-model prediction
};

class Lio
{
public:
    using Callback = std::function<void(const ChunkResult&)>;

    explicit Lio(const Params& p = Params())
        : P_(p)
        , ahrs_(p)
        , indoor_(p.indoor_resolution)
        , outdoor_(p.outdoor_resolution)
    {
    }

    void setCallback(Callback cb) { cb_ = std::move(cb); }

    void addImu(const ImuSample& s)
    {
        std::vector<std::pair<double, Eigen::Quaterniond>> out;
        if (!ahrs_.update(s, out))
            return;
        for (const auto& [t, q] : out)
        {
            ImuPose ip;
            ip.t = t;
            ip.T = Eigen::Affine3d::Identity();
            ip.T.linear() = q.toRotationMatrix();
            imu_.push_back(ip);
        }
    }

    void addPoints(const LidarPoint* pts, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
        {
            const double rxy = pts[i].p.head<2>().norm();
            if (!(rxy > P_.min_range_xy && rxy < P_.max_range_xy))
                continue;
            pts_.push_back(pts[i]);
            latest_t_ = std::max(latest_t_, pts[i].t);
            if (phase_ == Phase::WaitInit)
            {
                init_pts_.push_back(pts[i].p);
                if (static_cast<int>(init_pts_.size()) > P_.initial_points)
                {
                    t_begin_ = pts[i].t;
                    phase_ = Phase::Run;
                }
            }
        }
    }
    void addPoints(const std::vector<LidarPoint>& v) { addPoints(v.data(), v.size()); }

    // Runs every chunk that has all its data. Returns the number of chunks processed.
    size_t process()
    {
        if (phase_ != Phase::Run)
            return 0;
        if (chunk_idx_ == 0)
            while (!imu_.empty() && imu_.front().t < t_begin_)
                imu_.pop_front();
        size_t n = 0;
        const size_t thr = static_cast<size_t>(P_.poses_per_chunk);
        while (imu_.size() >= thr && latest_t_ >= imu_[thr - 1].t)
        {
            processChunk();
            ++n;
        }
        return n;
    }

    const std::vector<PoseStamped>& trajectory() const { return trajectory_; }
    void clearTrajectory() { trajectory_.clear(); }

    // all points currently in the rolling window (world frame)
    std::vector<Eigen::Vector3d> windowPoints() const
    {
        std::vector<Eigen::Vector3d> all;
        for (const auto& g : globals_)
            all.insert(all.end(), g.pts.begin(), g.pts.end());
        return all;
    }
    size_t mapBuckets() const { return indoor_.size() + outdoor_.size(); }
    size_t chunksProcessed() const { return chunk_idx_; }

private:
    using Clock = std::chrono::steady_clock;
    enum class Phase { WaitInit, Run };
    struct ImuPose
    {
        double t = 0.0;
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
    };
    struct GlobalBlock
    {
        double t = 0.0; // chunk end time
        std::vector<Eigen::Vector3d> pts;
    };

    void updateMaps(const std::vector<Eigen::Vector3d>& pts, const Eigen::Vector3d& viewport, bool ray_cast)
    {
#ifdef _OPENMP
        if (P_.multithread && P_.hierarchical)
        {
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                indoor_.update(pts, viewport, ray_cast);
#pragma omp section
                outdoor_.update(pts, viewport, ray_cast);
            }
            return;
        }
#endif
        indoor_.update(pts, viewport, ray_cast);
        if (P_.hierarchical)
            outdoor_.update(pts, viewport, ray_cast);
    }

    // HDMapping process_worker_step_update_rgd_after
    void updateMapAfter(const std::vector<Eigen::Vector3d>& pg_chunk, const std::vector<Eigen::Affine3d>& traj, double t_now)
    {
        const bool time_mode = P_.window_seconds > 0.0;
        const bool reset = time_mode ? (!globals_.empty() && t_now - globals_.front().t > P_.window_seconds)
                                     : (acc_distance_ > P_.sliding_window_distance);
        if (reset)
        {
            indoor_.clear();
            outdoor_.clear();
            if (time_mode)
            {
                const double cut = t_now - 0.5 * P_.window_seconds;
                while (globals_.size() > 1 && globals_.front().t < cut)
                    globals_.pop_front();
            }
            else
            {
                size_t drop = globals_.size() / 2;
                while (drop-- > 0)
                    globals_.pop_front();
            }
            acc_distance_ = 0.0;
            const std::vector<Eigen::Vector3d> all = windowPoints();
            updateMaps(all, traj.front().translation(), all.size() < 100000);
            return;
        }

        if (P_.hierarchical)
        {
            const Eigen::Affine3d cur = traj.back();
            if (!has_last_m_)
            {
                last_m_ = cur;
                has_last_m_ = true;
            }
            if ((cur.translation() - last_m_.translation()).norm() < indoor_.resolution() * 0.5)
                return;
            const TaitBryanPose rel = tb_from_affine(traj.front().inverse() * traj.back());
            if (std::fabs(rel.om) > 5.0 * kDegToRad || std::fabs(rel.fi) > 5.0 * kDegToRad || std::fabs(rel.ka) > 5.0 * kDegToRad)
                return;
            updateMaps(pg_chunk, traj.front().translation(), pg_chunk.size() < 100000);
            last_m_ = cur;
        }
        else
        {
            updateMaps(pg_chunk, traj.front().translation(), pg_chunk.size() < 100000);
        }
    }

    void processChunk()
    {
        const auto t_begin = Clock::now();
        const size_t thr = static_cast<size_t>(P_.poses_per_chunk);
        const double t0 = imu_[0].t, t1 = imu_[thr - 1].t;

        // ---- points of this chunk: [t0, t1), timestamps re-spread uniformly (as upstream), pose = last IMU pose before the point ----
        const auto cmp = [](const LidarPoint& a, double t) { return a.t < t; };
        const auto lo = std::lower_bound(pts_.begin(), pts_.end(), t0, cmp);
        const auto hi = std::lower_bound(lo, pts_.end(), t1, cmp);
        const size_t n_raw = static_cast<size_t>(hi - lo);
        const double ts_step = (P_.correct_point_timestamps && n_raw > 0) ? (t1 - t0) / static_cast<double>(n_raw) : 0.0;
        std::vector<double> pose_t(thr);
        for (size_t k = 0; k < thr; ++k)
            pose_t[k] = imu_[k].t;

        std::vector<ChunkPoint> pts;
        std::vector<OutputPoint> out_pts;
        pts.reserve(n_raw);
        size_t k = 0;
        size_t first_ge = 0; // first pose with pose_t >= t: the times only grow, so walk instead of searching
        for (auto it = lo; it != hi; ++it, ++k)
        {
            const double t = ts_step > 0.0 ? t0 + static_cast<double>(k) * ts_step : it->t;
            while (first_ge < thr && pose_t[first_ge] < t)
                ++first_ge;
            const long idx = static_cast<long>(first_ge) - 1;
            if (idx >= 0 && idx < static_cast<long>(thr) && std::fabs(t - pose_t[idx]) <= 0.01)
            {
                pts.push_back({ it->p, static_cast<int>(idx) });
                if (P_.collect_output_points)
                    out_pts.push_back({ it->p, it->t, it->intensity, static_cast<int>(idx) });
            }
        }
        const bool enough = static_cast<int>(pts.size()) > P_.min_points_per_chunk;

        if (P_.decimation > 0.0 && pts.size() > 1000)
        {
            const Eigen::Vector3d b(P_.decimation, P_.decimation, P_.decimation);
            ankerl::unordered_dense::set<uint64_t>& seen = seen_; // kept: no allocation per chunk
            seen.clear();
            seen.reserve(pts.size());
            std::vector<ChunkPoint> dec;
            dec.reserve(pts.size());
            for (const auto& cp : pts)
                if (seen.insert(rgd_key(cp.p, b)).second)
                    dec.push_back(cp);
            pts.swap(dec);
        }

        // ---- initial guess: previous end pose + AHRS relative rotations (+ constant velocity from the 2 previous chunks) ----
        std::vector<Eigen::Affine3d> traj(thr);
        traj[0] = chunk_idx_ == 0 ? imu_[0].T : prev_end_;
        for (size_t j = 1; j < thr; ++j)
            traj[j] = traj[j - 1] * (imu_[j - 1].T.inverse() * imu_[j].T);
        if (chunk_idx_ >= 2 && P_.use_motion_from_previous_step)
        {
            Eigen::Vector3d shift = (prev_end_.translation() - prevprev_end_.translation()) / static_cast<double>(thr);
            if (shift.norm() > 1.0)
                shift.setZero();
            for (size_t j = 0; j < thr; ++j)
                traj[j].translation() += shift * static_cast<double>(j);
        }

        // ---- seed the map from the first points, in the AHRS (gravity aligned) frame ----
        if (chunk_idx_ == 0)
        {
            std::vector<Eigen::Vector3d> pp(init_pts_.size());
            for (size_t i = 0; i < pp.size(); ++i)
                pp[i] = traj[0] * init_pts_[i];
            updateMaps(pp, traj[0].translation(), pp.size() < 100000);
            init_pts_.clear();
            init_pts_.shrink_to_fit();
        }

        // ---- optimize ----
        const std::vector<Eigen::Affine3d> traj_mm = traj;
        ChunkResult res;
        double delta = 100000.0;
        int it_end = 0;
        if (enough)
        {
            res.optimized = true;
            double lm = 1.0;
            const auto t_opt = Clock::now();
            for (int it = 0; it < P_.nr_iter; ++it)
            {
                it_end = it + 1;
                lookup_cells(pts, traj, indoor_, outdoor_, P_, cells_, P_.use_anisotropic_weighting ? &nv_ : nullptr, it > 0);
                delta = optimize_chunk(pts, cells_, traj, traj_mm, P_, lm, nv_);
                if (delta < P_.convergence_delta)
                    break;
                if (it % 10 == 0 && it > 0)
                    lm *= P_.lm_damping;
                if (std::chrono::duration<double>(Clock::now() - t_opt).count() > P_.max_time_per_chunk_s)
                    break;
            }
        }

        double step = (traj.front().inverse() * traj.back()).translation().norm();
        if (!std::isfinite(step))
        {
            traj = traj_mm;
            step = (traj.front().inverse() * traj.back()).translation().norm();
        }
        acc_distance_ += step;

        std::vector<Eigen::Vector3d> pg;
        pg.reserve(pts.size());
        for (const auto& cp : pts)
            pg.push_back(traj[cp.pose] * cp.p);
        globals_.push_back({ t1, pg });
        updateMapAfter(pg, traj, t1);

        // ---- output + bookkeeping ----
        res.poses.resize(thr);
        for (size_t j = 0; j < thr; ++j)
        {
            res.poses[j].t = pose_t[j];
            res.poses[j].T = traj[j];
            trajectory_.push_back(res.poses[j]);
        }
        res.iterations = it_end;
        res.delta = delta;
        res.points_global = std::move(pg);
        res.output_points = std::move(out_pts);
        res.ms = std::chrono::duration<double, std::milli>(Clock::now() - t_begin).count();

        prevprev_end_ = prev_end_;
        prev_end_ = traj.back();
        ++chunk_idx_;
        for (size_t j = 0; j < thr; ++j)
            imu_.pop_front();
        while (!pts_.empty() && pts_.front().t < t1)
            pts_.pop_front();

        if (cb_)
            cb_(res);
    }

    Params P_;
    Ahrs ahrs_;
    RgdMap indoor_, outdoor_;
    NvTables nv_;
    std::vector<PointCells> cells_;
    ankerl::unordered_dense::set<uint64_t> seen_;
    Callback cb_;

    std::deque<ImuPose> imu_;
    std::deque<LidarPoint> pts_;
    std::vector<Eigen::Vector3d> init_pts_;
    std::deque<GlobalBlock> globals_;
    std::vector<PoseStamped> trajectory_;

    Phase phase_ = Phase::WaitInit;
    double t_begin_ = 0.0;
    double latest_t_ = -1e300;
    size_t chunk_idx_ = 0;
    Eigen::Affine3d prev_end_ = Eigen::Affine3d::Identity();
    Eigen::Affine3d prevprev_end_ = Eigen::Affine3d::Identity();
    double acc_distance_ = 0.0;
    Eigen::Affine3d last_m_ = Eigen::Affine3d::Identity();
    bool has_last_m_ = false;
};

} // namespace mini_lio
