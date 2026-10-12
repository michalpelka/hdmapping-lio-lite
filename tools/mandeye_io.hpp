// Mandeye session input: IMU csv and lidar LAZ (via LASzip). Mirrors HDMapping step 1
// (lidar_odometry.cpp load_data, lidar_odometry_utils.cpp load_imu / load_point_cloud). No calibration: points are used as recorded.
#pragma once

#include <Eigen/Dense>
#include <laszip/laszip_api.h>
#include <mini_lio/types.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mandeye
{

// ---------------------------------------------------------------- IMU csv
struct ImuRow
{
    mini_lio::ImuSample s; // t [s], gyro [rad/s], acc [g]
    double unix_s = 0.0;
};

inline std::vector<std::string> split_fields(const std::string& line)
{
    std::vector<std::string> f;
    std::string cur;
    for (char c : line)
    {
        if (c == ',' || c == ' ' || c == '\t' || c == '\r')
        {
            if (!cur.empty())
                f.push_back(cur), cur.clear();
        }
        else
            cur += c;
    }
    if (!cur.empty())
        f.push_back(cur);
    return f;
}

// With header (timestamp,timestampUnix,gyroX..accZ[,imuId]) or legacy headerless "ts gx gy gz ax ay az [imuId [unix]]".
// Rows of other IMUs are skipped, as upstream (imuId < 0 / absent column means "all").
inline std::vector<ImuRow> load_imu(const std::string& path, int imu_to_use)
{
    std::vector<ImuRow> out;
    std::ifstream f(path);
    if (!f)
    {
        std::cerr << "cannot open " << path << "\n";
        return out;
    }
    std::string line;
    std::getline(f, line);
    const auto head = split_fields(line);
    std::map<std::string, size_t> col;
    for (size_t i = 0; i < head.size(); ++i)
        col[head[i]] = i;
    const bool has_header = col.count("timestamp") > 0;
    const auto num = [](const std::vector<std::string>& v, size_t i) { return i < v.size() ? std::strtod(v[i].c_str(), nullptr) : 0.0; };

    const auto handle_legacy = [&](const std::string& l)
    {
        const auto v = split_fields(l);
        if (v.size() < 7)
            return;
        const double ts = num(v, 0);
        const int id = v.size() > 7 ? static_cast<int>(num(v, 7)) : 0;
        if (!(ts > 0) || id != imu_to_use)
            return;
        ImuRow r;
        r.s.t = ts / 1e9;
        r.s.gyro = { num(v, 1), num(v, 2), num(v, 3) };
        r.s.acc = { num(v, 4), num(v, 5), num(v, 6) };
        r.unix_s = v.size() > 8 ? num(v, 8) / 1e9 : 0.0;
        out.push_back(r);
    };

    if (!has_header)
    {
        handle_legacy(line);
        while (std::getline(f, line))
            handle_legacy(line);
        return out;
    }
    for (const char* k : { "gyroX", "gyroY", "gyroZ", "accX", "accY", "accZ" })
        if (!col.count(k))
        {
            std::cerr << path << ": missing column " << k << "\n";
            return out;
        }
    const bool has_id = col.count("imuId") > 0, has_unix = col.count("timestampUnix") > 0;
    while (std::getline(f, line))
    {
        const auto v = split_fields(line);
        if (v.size() < head.size())
            continue;
        if (has_id && static_cast<int>(num(v, col["imuId"])) != imu_to_use)
            continue;
        ImuRow r;
        r.s.t = num(v, col["timestamp"]) / 1e9;
        r.s.gyro = { num(v, col["gyroX"]), num(v, col["gyroY"]), num(v, col["gyroZ"]) };
        r.s.acc = { num(v, col["accX"]), num(v, col["accY"]), num(v, col["accZ"]) };
        r.unix_s = has_unix ? num(v, col["timestampUnix"]) / 1e9 : 0.0;
        out.push_back(r);
    }
    return out;
}

// ---------------------------------------------------------------- LAZ points
// gps_time is nanoseconds (or seconds in some sessions, detected per file from the time span),
// intensity kept, time-sorted. Points with gps_time 0 and with xy range outside
// (min_xy, max_xy) are dropped, as upstream.
inline bool load_laz(const std::string& path, double min_xy, double max_xy, std::vector<mini_lio::LidarPoint>& out)
{
    laszip_POINTER rd = nullptr;
    if (laszip_create(&rd))
        return false;
    laszip_BOOL compressed = 0;
    if (laszip_open_reader(rd, path.c_str(), &compressed))
    {
        std::cerr << "cannot open " << path << "\n";
        laszip_destroy(rd);
        return false;
    }
    laszip_header* h = nullptr;
    laszip_point* pt = nullptr;
    laszip_get_header_pointer(rd, &h);
    laszip_get_point_pointer(rd, &pt);
    const uint64_t n = h->number_of_point_records ? h->number_of_point_records : h->extended_number_of_point_records;
    out.clear();
    out.reserve(n);
    std::set<int> ids;
    for (uint64_t i = 0; i < n; ++i)
    {
        if (laszip_read_point(rd))
            break;
        if (pt->gps_time == 0)
            continue;
        ids.insert(pt->user_data);
        const Eigen::Vector3d pf(h->x_offset + h->x_scale_factor * pt->X, h->y_offset + h->y_scale_factor * pt->Y, h->z_offset + h->z_scale_factor * pt->Z);
        const double rxy = pf.head<2>().norm();
        if (!(rxy > min_xy && rxy < max_xy))
            continue;
        mini_lio::LidarPoint lp;
        lp.p = pf;
        lp.t = pt->gps_time; // raw, scaled below
        lp.intensity = pt->intensity;
        out.push_back(lp);
    }
    laszip_close_reader(rd);
    if (ids.size() > 1)
        std::cerr << "WARNING: " << path << " has " << ids.size() << " lidar ids; no calibration is applied, points are used as recorded\n";
    laszip_destroy(rd);
    // A file covers a few seconds: a span of ~1e9 means ns, a span below 1000 means seconds already.
    if (!out.empty())
    {
        const auto [lo, hi] = std::minmax_element(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.t < b.t; });
        const double scale = (hi->t - lo->t) < 1e3 ? 1.0 : 1e-9;
        for (auto& lp : out)
            lp.t *= scale;
    }
    std::stable_sort(out.begin(), out.end(), [](const mini_lio::LidarPoint& a, const mini_lio::LidarPoint& b) { return a.t < b.t; });
    return true;
}

// ---------------------------------------------------------------- session discovery
struct SessionFiles
{
    std::vector<std::string> laz, csv;
};

// Sort, classify by extension; laz and csv are paired by sorted position (as upstream).
inline SessionFiles classify(std::vector<std::string> files)
{
    namespace fs = std::filesystem;
    std::sort(files.begin(), files.end());
    SessionFiles s;
    for (const auto& fn : files)
    {
        std::string ext = fs::path(fn).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".csv")
            s.csv.push_back(fn);
        else if (ext == ".laz" || ext == ".las")
            s.laz.push_back(fn);
    }
    return s;
}

} // namespace mandeye
