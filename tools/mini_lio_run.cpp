// mini_lio_run: run mini_lio on a Mandeye session (LAZ + IMU csv) and write an HDMapping-style result folder.
//
//   mini_lio_run <session dir | files...> [-o out_dir] [options]       (see --help)

#include "mandeye_io.hpp"
#include "session_writer.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace fs = std::filesystem;
using namespace mini_lio;

static void usage()
{
    std::puts(
        "usage: mini_lio_run <session dir | files...> [options]\n"
        "  input: *.laz/*.las + *.csv (paired by sorted order); points are used as recorded (no calibration)\n"
        "  -o DIR            output folder (default: <session>/lio_result_N, next free N)\n"
        "  --imu-id N        IMU id to use (default 0)\n"
        "  --use-first-file  also use points of the first laz (upstream drops them)\n"
        "  --vqf             VQF instead of Madgwick/Fusion\n"
        "  --imu-rate HZ     IMU rate for the AHRS (default: from the csv timestamps)\n"
        "  --iters N         LM iterations per chunk (default 100 as upstream; 8-15 is much faster)\n"
        "  --window S        rolling map of the last ~S seconds instead of 200 m of travel\n"
        "  --max-seconds S   stop after S seconds of lidar data (after the first used file)\n"
        "  --chunks-per-scan N   chunks (0.1 s each) per scan_lio_N.laz (default 51 as upstream)\n"
        "  --min-out-range R drop output points closer than R m to the sensor (default 0.5)\n"
        "  --no-laz          write trajectories only\n"
        "  --single-thread");
}

int main(int argc, char** argv)
{
    Params P;
    std::vector<std::string> inputs;
    std::string out_dir;
    int imu_id = 0, chunks_per_scan = 51;
    bool use_first = false, write_laz = true;
    double max_seconds = 0.0, min_out = 0.5, imu_rate = 0.0;
    for (int i = 1; i < argc; ++i)
    {
        const auto arg = [&](const char* n) { return !std::strcmp(argv[i], n); };
        const auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : (usage(), std::exit(2), "") ; };
        if (arg("-o"))
            out_dir = val();
        else if (arg("--imu-id"))
            imu_id = std::atoi(val());
        else if (arg("--use-first-file"))
            use_first = true;
        else if (arg("--vqf"))
            P.use_vqf = true;
        else if (arg("--imu-rate"))
            imu_rate = std::atof(val());
        else if (arg("--iters"))
            P.nr_iter = std::atoi(val());
        else if (arg("--window"))
            P.window_seconds = std::atof(val());
        else if (arg("--max-seconds"))
            max_seconds = std::atof(val());
        else if (arg("--chunks-per-scan"))
            chunks_per_scan = std::max(1, std::atoi(val()));
        else if (arg("--min-out-range"))
            min_out = std::atof(val());
        else if (arg("--no-laz"))
            write_laz = false;
        else if (arg("--single-thread"))
            P.multithread = false;
        else if (arg("--help") || arg("-h"))
            return usage(), 0;
        else if (argv[i][0] == '-')
            return std::fprintf(stderr, "unknown option %s\n", argv[i]), usage(), 2;
        else
            inputs.push_back(argv[i]);
    }
    if (inputs.empty())
        return usage(), 2;

    // ---- discover files
    std::vector<std::string> files;
    fs::path session_dir;
    for (const auto& in : inputs)
    {
        if (fs::is_directory(in))
        {
            session_dir = in;
            for (const auto& e : fs::directory_iterator(in))
                if (e.is_regular_file())
                    files.push_back(e.path().string());
        }
        else
        {
            files.push_back(in);
            if (session_dir.empty())
                session_dir = fs::path(in).parent_path();
        }
    }
    mandeye::SessionFiles sf = mandeye::classify(files);
    const size_t n_files = std::min(sf.laz.size(), sf.csv.size());
    if (n_files == 0)
        return std::fprintf(stderr, "need at least one laz + csv pair (found %zu laz, %zu csv)\n", sf.laz.size(), sf.csv.size()), 1;
    if (sf.laz.size() != sf.csv.size())
        std::fprintf(stderr, "WARNING: %zu laz vs %zu csv files, using the first %zu pairs\n", sf.laz.size(), sf.csv.size(), n_files);

    // ---- IMU: all files, time sorted
    std::vector<mandeye::ImuRow> imu;
    for (size_t i = 0; i < n_files; ++i)
    {
        auto v = mandeye::load_imu(sf.csv[i], imu_id);
        imu.insert(imu.end(), v.begin(), v.end());
    }
    std::stable_sort(imu.begin(), imu.end(), [](const auto& a, const auto& b) { return a.s.t < b.s.t; });
    if (imu.size() < 100)
        return std::fprintf(stderr, "only %zu IMU samples for imu id %d\n", imu.size(), imu_id), 1;
    std::printf("session: %zu laz/csv pairs, %zu IMU samples (imu id %d), %.1f s\n", n_files, imu.size(), imu_id, imu.back().s.t - imu.front().s.t);

    // VQF integrates with a fixed sample time: use the real rate (Livox 200 Hz, Hesai JT ~430 Hz).
    // Average over the whole span: some IMUs (Hesai JT) timestamp samples in bursts, so the
    // median interval is far shorter than the real sample period.
    if (imu_rate <= 0.0 && imu.back().s.t > imu.front().s.t)
        imu_rate = std::round((imu.size() - 1) / (imu.back().s.t - imu.front().s.t));
    if (imu_rate > 0.0)
        P.imu_rate_hz = imu_rate;
    std::printf("IMU rate: %.0f Hz\n", P.imu_rate_hz);

    if (out_dir.empty())
    {
        int n = 0;
        while (fs::exists(session_dir / ("lio_result_" + std::to_string(n))))
            ++n;
        out_dir = (session_dir / ("lio_result_" + std::to_string(n))).string();
    }

    // t -> unix time for the trajectory csv
    std::vector<std::pair<double, double>> t_unix;
    t_unix.reserve(imu.size());
    for (const auto& r : imu)
        t_unix.emplace_back(r.s.t, r.unix_s);
    const auto unix_of_t = [&](double t)
    {
        auto it = std::lower_bound(t_unix.begin(), t_unix.end(), std::make_pair(t, -1e300));
        if (it == t_unix.end())
            --it;
        return it->second;
    };

    P.collect_output_points = write_laz;
    mandeye::SessionWriter writer(out_dir, chunks_per_scan, min_out, write_laz, unix_of_t);
    Lio lio(P);
    size_t n_chunks = 0, n_opt = 0;
    double ms_sum = 0.0;
    lio.setCallback(
        [&](const ChunkResult& r)
        {
            ++n_chunks;
            n_opt += r.optimized;
            ms_sum += r.ms;
            writer.add(r);
        });

    double prev_t = -1e300;
    for (const auto& r : imu)
        if (r.s.t > prev_t)
        {
            lio.addImu(r.s);
            prev_t = r.s.t;
        }

    // ---- stream the laz files
    const auto c0 = std::chrono::steady_clock::now();
    double t_first_pt = -1.0, t_last_pt = 0.0;
    size_t total_pts = 0;
    for (size_t i = 0; i < n_files; ++i)
    {
        if (i == 0 && !use_first)
        {
            std::printf("skipping points of %s (as upstream, use --use-first-file to keep)\n", sf.laz[i].c_str());
            continue;
        }
        std::vector<LidarPoint> pts;
        if (!mandeye::load_laz(sf.laz[i], P.min_range_xy, P.max_range_xy, pts))
            return 1;
        if (pts.empty())
            continue;
        if (t_first_pt < 0)
            t_first_pt = pts.front().t;
        if (max_seconds > 0.0)
        {
            const double t_lim = t_first_pt + max_seconds;
            pts.erase(std::remove_if(pts.begin(), pts.end(), [&](const LidarPoint& p) { return p.t > t_lim; }), pts.end());
        }
        if (pts.empty())
            break; // everything past --max-seconds
        total_pts += pts.size();
        t_last_pt = pts.empty() ? t_last_pt : pts.back().t;
        lio.addPoints(pts);
        lio.process();
        std::printf("%s: %zu points, chunks so far %zu\n", fs::path(sf.laz[i]).filename().string().c_str(), pts.size(), n_chunks);
        if (max_seconds > 0.0 && t_last_pt >= t_first_pt + max_seconds)
            break;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    if (n_chunks == 0)
        return std::fprintf(stderr, "no chunk processed: too few points (need > %d after filtering, in the IMU time range)\n", P.initial_points), 1;
    if (!writer.finish())
        return std::fprintf(stderr, "writing the result failed\n"), 1;

    const auto& tr = lio.trajectory();
    double len = 0.0;
    for (size_t i = 20; i < tr.size(); i += 20)
        len += (tr[i].T.translation() - tr[i - 20].T.translation()).norm();
    std::printf(
        "done: %zu chunks (%zu optimized), %zu poses, ~%.1f m, %.1f s of lidar data in %.1f s (x%.2f real time), %.1f ms/chunk\n", n_chunks, n_opt, tr.size(), len,
        t_last_pt - t_first_pt, wall, (t_last_pt - t_first_pt) / std::max(wall, 1e-9), ms_sum / n_chunks);
    std::printf("output: %s  (%zu scans, %zu points)\n", out_dir.c_str(), writer.scans(), writer.total_points());
    return 0;
}
