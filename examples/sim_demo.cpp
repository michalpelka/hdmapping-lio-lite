// Synthetic handheld run: a spinning 32-ring LiDAR + noisy 200 Hz IMU in a room with a few obstacles,
// streamed through mini_lio and compared with the ground truth trajectory.
//
//   sim_demo [--seconds 20] [--iters N] [--vqf] [--window S] [--no-aniso] [--no-gyro-bias] [--single-thread] [--out traj.tum]
//            [--export-session DIR]   (needs LASzip) also writes the simulated data as a Mandeye-style session:
//                                     imu0000/0001.csv, lidar0000/0001.laz, ground_truth.tum

#include <mini_lio/lio.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#ifdef MINI_LIO_HAVE_LASZIP
#include <laszip/laszip_api.h>
#include <filesystem>
#endif

using namespace mini_lio;

namespace
{

constexpr double kG = 9.80665;
const Eigen::Vector3d kRoomMax(10.0, 8.0, 3.0);

struct Box
{
    Eigen::Vector3d lo, hi;
};
const Box kBoxes[] = {
    { { 1.0, 1.0, 0.0 }, { 2.0, 3.0, 1.0 } },
    { { 8.0, 5.8, 0.0 }, { 9.5, 7.2, 2.0 } },
    { { 3.5, 6.5, 0.0 }, { 4.5, 7.5, 0.8 } },
    { { 8.0, 1.0, 0.0 }, { 8.5, 1.5, 3.0 } },
};

struct GtPose
{
    Eigen::Vector3d p;
    Eigen::Matrix3d R;
};

GtPose gt(double t)
{
    GtPose g;
    g.p = Eigen::Vector3d(
        5.0 + 2.5 * std::sin(0.30 * t), 4.0 + 2.0 * std::sin(0.45 * t + 0.5), 1.4 + 0.05 * std::sin(2 * M_PI * 0.9 * t));
    const double yaw = 0.3 + 0.6 * std::sin(0.25 * t);
    const double roll = 0.12 * std::sin(2 * M_PI * 0.8 * t);
    const double pitch = 0.10 * std::sin(2 * M_PI * 0.6 * t + 1.0);
    g.R = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
              .toRotationMatrix();
    return g;
}

double raycast(const Eigen::Vector3d& o, const Eigen::Vector3d& d)
{
    double best = 1e9;
    // room: leave through the nearest wall
    for (int a = 0; a < 3; ++a)
    {
        if (std::fabs(d[a]) < 1e-12)
            continue;
        const double t = d[a] > 0 ? (kRoomMax[a] - o[a]) / d[a] : (0.0 - o[a]) / d[a];
        best = std::min(best, t);
    }
    for (const Box& b : kBoxes)
    {
        double tn = 0.0, tf = 1e9;
        bool ok = true;
        for (int a = 0; a < 3 && ok; ++a)
        {
            if (std::fabs(d[a]) < 1e-12)
            {
                ok = o[a] >= b.lo[a] && o[a] <= b.hi[a];
                continue;
            }
            double t1 = (b.lo[a] - o[a]) / d[a], t2 = (b.hi[a] - o[a]) / d[a];
            if (t1 > t2)
                std::swap(t1, t2);
            tn = std::max(tn, t1);
            tf = std::min(tf, t2);
            ok = tn <= tf;
        }
        if (ok && tn > 0.0)
            best = std::min(best, tn);
    }
    return best;
}

} // namespace

int main(int argc, char** argv)
{
    double seconds = 20.0;
    std::string out_path, export_dir;
    Params P;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc)
            seconds = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--vqf"))
            P.use_vqf = true;
        else if (!std::strcmp(argv[i], "--window") && i + 1 < argc)
            P.window_seconds = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
            P.nr_iter = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-aniso"))
            P.use_anisotropic_weighting = false;
        else if (!std::strcmp(argv[i], "--single-thread"))
            P.multithread = false;
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc)
            out_path = argv[++i];
        else if (!std::strcmp(argv[i], "--export-session") && i + 1 < argc)
            export_dir = argv[++i];
    }

    constexpr double imu_rate = 200.0, pt_rate = 200000.0, step = 0.1;
    constexpr int rings = 32;
    std::mt19937 rng(42);
    std::normal_distribution<double> n01(0.0, 1.0);
    Eigen::Vector3d gyro_bias(0.002, -0.0015, 0.001);
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--no-gyro-bias"))
            gyro_bias.setZero();

    std::vector<ImuSample> imu_log;
    std::vector<LidarPoint> pt_log;
    std::vector<int> pt_ring;
    Lio lio(P);
    double process_s = 0.0;
    size_t iters = 0, n_chunks = 0, n_opt = 0;
    double ms_sum = 0.0, ms_max = 0.0;
    lio.setCallback(
        [&](const ChunkResult& r)
        {
            ++n_chunks;
            iters += r.iterations;
            n_opt += r.optimized;
            ms_sum += r.ms;
            ms_max = std::max(ms_max, r.ms);
        });

    size_t imu_i = 0, pt_i = 0;
    const size_t steps = static_cast<size_t>(seconds / step);
    for (size_t s = 0; s < steps; ++s)
    {
        const double t_end = (s + 1) * step;
        for (; imu_i < static_cast<size_t>(t_end * imu_rate + 0.5); ++imu_i)
        {
            const double t = imu_i / imu_rate;
            const double h = 1e-4, h2 = 1e-3;
            const GtPose g0 = gt(t), gm = gt(t - h), gp = gt(t + h);
            const Eigen::AngleAxisd aa(Eigen::Matrix3d(gm.R.transpose() * gp.R));
            const Eigen::Vector3d w = aa.axis() * aa.angle() / (2 * h);
            const Eigen::Vector3d a_w = (gt(t + h2).p - 2.0 * g0.p + gt(t - h2).p) / (h2 * h2);
            ImuSample im;
            im.t = t;
            im.gyro = w + gyro_bias + 0.0035 * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
            im.acc = g0.R.transpose() * (a_w - Eigen::Vector3d(0, 0, -kG)) / kG + 0.004 * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
            lio.addImu(im);
            if (!export_dir.empty())
                imu_log.push_back(im);
        }

        std::vector<LidarPoint> batch;
        for (; pt_i < static_cast<size_t>(t_end * pt_rate + 0.5); ++pt_i)
        {
            const double t = pt_i / pt_rate;
            const GtPose g = gt(t);
            const double az = 2 * M_PI * 10.0 * t;
            const double el = (-30.0 + 60.0 * (pt_i % rings) / (rings - 1)) * M_PI / 180.0;
            const Eigen::Vector3d d_b(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
            const double r = raycast(g.p, g.R * d_b) + 0.01 * n01(rng);
            if (r < 0.3 || r > 70.0)
                continue;
            LidarPoint lp;
            lp.p = r * d_b;
            lp.t = t;
            batch.push_back(lp);
            if (!export_dir.empty())
                pt_ring.push_back(static_cast<int>(pt_i % rings));
        }
        lio.addPoints(batch);
        if (!export_dir.empty())
            pt_log.insert(pt_log.end(), batch.begin(), batch.end());

        const auto c0 = std::chrono::steady_clock::now();
        lio.process();
        process_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    }

    // ---- evaluation against ground truth, two alignments: first pose, and best-fit rigid (Umeyama, standard ATE) ----
    const auto& tr = lio.trajectory();
    if (tr.size() < 10)
    {
        std::printf("FAILED: only %zu poses\n", tr.size());
        return 1;
    }
    const auto gt_affine = [](double t)
    {
        const GtPose g = gt(t);
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.linear() = g.R;
        T.translation() = g.p;
        return T;
    };
    struct Err
    {
        double pos_rms = 0, pos_max = 0, pos_final = 0, rot_rms = 0;
    };
    const auto evaluate = [&](const Eigen::Affine3d& T_align, bool trace_it)
    {
        Err e;
        double se = 0, rse = 0;
        int last_slot = -1;
        for (const auto& ps : tr)
        {
            const Eigen::Affine3d T_ref = T_align * gt_affine(ps.t);
            const double pe = (T_ref.translation() - ps.T.translation()).norm();
            const double re = Eigen::AngleAxisd(Eigen::Matrix3d(T_ref.linear().transpose() * ps.T.linear())).angle() * 180.0 / M_PI;
            se += pe * pe;
            rse += re * re;
            e.pos_max = std::max(e.pos_max, pe);
            e.pos_final = pe;
            if (trace_it && static_cast<int>(ps.t * 2.0) != last_slot)
            {
                last_slot = static_cast<int>(ps.t * 2.0);
                const Eigen::Vector3d up_lio = ps.T.linear().transpose() * Eigen::Vector3d::UnitZ();
                const Eigen::Vector3d up_gt = gt(ps.t).R.transpose() * Eigen::Vector3d::UnitZ();
                const double tilt = std::acos(std::min(1.0, up_lio.dot(up_gt))) * 180.0 / M_PI;
                std::printf("  t=%5.1f s  pos err %.4f m  rot err %.3f deg  (tilt vs gravity, alignment-free: %.3f deg)\n", ps.t, pe, re, tilt);
            }
        }
        e.pos_rms = std::sqrt(se / tr.size());
        e.rot_rms = std::sqrt(rse / tr.size());
        return e;
    };

    bool trace = false;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--trace"))
            trace = true;

    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, tr.size()), dst(3, tr.size());
    for (size_t i = 0; i < tr.size(); ++i)
    {
        src.col(i) = gt(tr[i].t).p;
        dst.col(i) = tr[i].T.translation();
    }
    Eigen::Affine3d T_fit;
    T_fit.matrix() = Eigen::umeyama(src, dst, false);
    const Eigen::Affine3d T_first = tr.front().T * gt_affine(tr.front().t).inverse();
    const Err e_fit = evaluate(T_fit, trace), e_first = evaluate(T_first, false);

    if (!out_path.empty())
    {
        std::ofstream out(out_path);
        for (const auto& ps : tr)
        {
            const Eigen::Quaterniond q(ps.T.linear());
            out << ps.t << " " << ps.T.translation().transpose() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
    }
    double path_len = 0.0;
    for (size_t i = 1; i < tr.size(); ++i)
        path_len += (gt(tr[i].t).p - gt(tr[i - 1].t).p).norm();

    std::printf("AHRS: %s | threads: %s | anisotropic: %d | window: %.1f s | iters: %d\n", P.use_vqf ? "VQF" : "Fusion", P.multithread ? "OpenMP" : "1", P.use_anisotropic_weighting, P.window_seconds, P.nr_iter);
    std::printf("poses: %zu in %zu chunks (%zu optimized), path %.1f m\n", tr.size(), n_chunks, n_opt, path_len);
    std::printf("ATE best-fit rigid : position RMS %.4f m (max %.4f, final %.4f), rotation RMS %.3f deg\n", e_fit.pos_rms, e_fit.pos_max, e_fit.pos_final, e_fit.rot_rms);
    std::printf("first-pose aligned : position RMS %.4f m (max %.4f, final %.4f), rotation RMS %.3f deg\n", e_first.pos_rms, e_first.pos_max, e_first.pos_final, e_first.rot_rms);
    std::printf("map: %zu buckets, %zu window points\n", lio.mapBuckets(), lio.windowPoints().size());
    std::printf(
        "timing: %.2f s processing for %.1f s of data (x%.2f real time), chunk avg %.1f ms, max %.1f ms, %.1f iterations/chunk\n", process_s, seconds,
        seconds / process_s, ms_sum / std::max<size_t>(1, n_chunks), ms_max, static_cast<double>(iters) / std::max<size_t>(1, n_chunks));
    if (!export_dir.empty())
    {
#ifdef MINI_LIO_HAVE_LASZIP
        namespace fs = std::filesystem;
        fs::create_directories(export_dir);
        // A junk second IMU id is interleaved to exercise the id filter.
        const double t_boot = 1000.0, t_split = 2.0;
        {
            std::ofstream g(fs::path(export_dir) / "ground_truth.tum");
            g.precision(12);
            for (const auto& ps : lio.trajectory())
            {
                const GtPose gp = gt(ps.t);
                const Eigen::Quaterniond q(gp.R);
                g << ps.t + t_boot << " " << gp.p.transpose() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
            }
        }
        for (int f = 0; f < 2; ++f)
        {
            const double lo = f == 0 ? -1.0 : t_split, hi = f == 0 ? t_split : 1e9;
            char nm[32];
            std::snprintf(nm, sizeof nm, "%04d", f);
            std::ofstream csv(fs::path(export_dir) / (std::string("imu") + nm + ".csv"));
            csv << "timestamp,timestampUnix,gyroX,gyroY,gyroZ,accX,accY,accZ,imuId\n";
            csv.precision(17);
            for (const auto& im : imu_log)
                if (im.t >= lo && im.t < hi)
                {
                    char buf[512];
                    std::snprintf(
                        buf, sizeof buf, "%.0f,%.0f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,0\n", (im.t + t_boot) * 1e9, (1.7e9 + im.t) * 1e9, im.gyro.x(), im.gyro.y(), im.gyro.z(),
                        im.acc.x(), im.acc.y(), im.acc.z());
                    csv << buf;
                    std::snprintf(buf, sizeof buf, "%.0f,%.0f,9,9,9,9,9,9,1\n", (im.t + t_boot) * 1e9, (1.7e9 + im.t) * 1e9);
                    csv << buf;
                }
            laszip_POINTER w = nullptr;
            laszip_create(&w);
            laszip_header* h = nullptr;
            laszip_get_header_pointer(w, &h);
            h->version_major = 1;
            h->version_minor = 2;
            h->point_data_format = 1;
            h->point_data_record_length = 28;
            h->x_scale_factor = h->y_scale_factor = h->z_scale_factor = 0.0001;
            if (laszip_open_writer(w, (fs::path(export_dir) / (std::string("lidar") + nm + ".laz")).string().c_str(), 1))
            {
                laszip_CHAR* err = nullptr;
                laszip_get_error(w, &err);
                std::fprintf(stderr, "laszip_open_writer: %s\n", err ? err : "?");
                return 1;
            }
            laszip_point* pt = nullptr;
            laszip_get_point_pointer(w, &pt);
            for (size_t i = 0; i < pt_log.size(); ++i)
            {
                if (!(pt_log[i].t >= lo && pt_log[i].t < hi))
                    continue;
                const Eigen::Vector3d p = pt_log[i].p;
                pt->intensity = static_cast<laszip_U16>(100 + pt_ring[i]);
                pt->gps_time = (pt_log[i].t + t_boot) * 1e9;
                laszip_F64 xyz[3] = { p.x(), p.y(), p.z() };
                laszip_set_coordinates(w, xyz);
                laszip_write_point(w);
                laszip_update_inventory(w);
            }
            laszip_close_writer(w);
            laszip_destroy(w);
        }
        std::printf("session written to %s\n", export_dir.c_str());
#else
        std::fprintf(stderr, "--export-session needs LASzip at build time\n");
#endif
    }
    return 0;
}
