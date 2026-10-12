// Writes mini_lio output in the layout of HDMapping step 1 (lidar_odometry.cpp save_result):
//   scan_lio_N.laz           points in the frame of the first pose of scan N (LAS 1.2, format 1, scale 1e-4, gps_time ns)
//   trajectory_lio_N.csv     poses relative to the first pose of scan N
//   session_poses.mrp / session_ini_poses.mri   first pose of every scan (scan frame -> world)
//   session.mjs              session description, opens in HDMapping step 2
#pragma once

#include <Eigen/Dense>
#include <laszip/laszip_api.h>
#include <mini_lio/lio.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace mandeye
{

class SessionWriter
{
public:
    SessionWriter(std::string out_dir, int chunks_per_scan, double min_output_range, bool write_laz, std::function<double(double)> unix_of_t)
        : dir_(std::move(out_dir)), chunks_per_scan_(chunks_per_scan), min_range_(min_output_range), write_laz_(write_laz), unix_of_t_(std::move(unix_of_t))
    {
        std::filesystem::create_directories(dir_);
    }

    void add(const mini_lio::ChunkResult& r)
    {
        const int base = static_cast<int>(poses_.size());
        poses_.insert(poses_.end(), r.poses.begin(), r.poses.end());
        for (const auto& op : r.output_points)
        {
            points_.push_back(op);
            points_.back().pose += base;
        }
        point_sizes_.push_back(static_cast<int>(r.output_points.size()));
        if (++chunks_in_scan_ >= chunks_per_scan_)
            flush();
    }

    // writes the pending partial scan and the session files
    bool finish()
    {
        if (chunks_in_scan_ > 0)
            flush();
        if (!ok_)
            return false;
        write_poses("session_poses.mrp");
        write_poses("session_ini_poses.mri");
        {
            std::ofstream f(std::filesystem::path(dir_) / "point_sizes_per_chunk.json");
            f << "[";
            for (size_t i = 0; i < point_sizes_.size(); ++i)
                f << (i ? ", " : "") << point_sizes_[i];
            f << "]\n";
        }
        std::ofstream f(std::filesystem::path(dir_) / "session.mjs");
        const auto abs = [&](const std::string& n) { return (std::filesystem::absolute(dir_) / n).string(); };
        f << "{\n  \"Session Settings\": {\n"
          << "    \"offset_x\": 0.0,\n    \"offset_y\": 0.0,\n    \"offset_z\": 0.0,\n"
          << "    \"folder_name\": \"" << std::filesystem::absolute(dir_).string() << "\",\n"
          << "    \"out_folder_name\": \"" << std::filesystem::absolute(dir_).string() << "\",\n"
          << "    \"poses_file_name\": \"" << abs("session_poses.mrp") << "\",\n"
          << "    \"initial_poses_file_name\": \"" << abs("session_ini_poses.mri") << "\",\n"
          << "    \"out_poses_file_name\": \"" << abs("session_poses.mrp") << "\",\n"
          << "    \"lidar_odometry_version\": \"mini_lio\",\n"
          << "    \"point_sizes_path\": \"" << abs("point_sizes_per_chunk.json") << "\"\n  },\n  \"laz_file_names\": [\n";
        for (size_t i = 0; i < names_.size(); ++i)
            f << "    { \"file_name\": \"" << abs(names_[i]) << "\" }" << (i + 1 < names_.size() ? "," : "") << "\n";
        f << "  ]\n}\n";
        return ok_ && f.good();
    }

    size_t scans() const { return names_.size(); }
    size_t total_points() const { return total_points_; }

private:
    void flush()
    {
        namespace fs = std::filesystem;
        const size_t idx = names_.size();
        const std::string laz_name = "scan_lio_" + std::to_string(idx) + ".laz";
        const Eigen::Affine3d T0 = poses_.front().T, T0inv = T0.inverse();
        scan_poses_.push_back(T0);
        names_.push_back(laz_name);

        if (write_laz_)
        {
            std::vector<Eigen::Affine3d> to_local(poses_.size());
            for (size_t i = 0; i < poses_.size(); ++i)
                to_local[i] = T0inv * poses_[i].T;
            ok_ &= write_laz(fs::path(dir_) / laz_name, to_local);
        }

        std::ofstream f(fs::path(dir_) / ("trajectory_lio_" + std::to_string(idx) + ".csv"));
        f << "timestamp_nanoseconds pose00 pose01 pose02 pose03 pose10 pose11 pose12 pose13 pose20 pose21 pose22 pose23 "
             "timestampUnix_nanoseconds om_rad fi_rad ka_rad\n";
        for (const auto& ps : poses_)
        {
            const Eigen::Affine3d rel = T0inv * ps.T;
            const mini_lio::TaitBryanPose tb = mini_lio::tb_from_affine(ps.T);
            f << std::setprecision(20) << ps.t * 1e9 << " " << std::setprecision(10);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    f << rel(r, c) << " ";
            f << std::setprecision(20) << unix_of_t_(ps.t) * 1e9 << " " << tb.om << " " << tb.fi << " " << tb.ka << "\n";
        }
        ok_ &= f.good();
        poses_.clear();
        points_.clear();
        chunks_in_scan_ = 0;
    }

    bool write_laz(const std::filesystem::path& path, const std::vector<Eigen::Affine3d>& to_local)
    {
        laszip_POINTER w = nullptr;
        if (laszip_create(&w))
            return false;
        laszip_header* h = nullptr;
        laszip_get_header_pointer(w, &h);
        h->file_source_ID = 4711;
        h->global_encoding = 1;
        h->version_major = 1;
        h->version_minor = 2;
        h->point_data_format = 1;
        h->point_data_record_length = 28;
        h->number_of_point_records = 0;
        h->x_scale_factor = h->y_scale_factor = h->z_scale_factor = 0.0001;
        h->x_offset = h->y_offset = h->z_offset = 0.0;
        if (laszip_open_writer(w, path.string().c_str(), 1))
        {
            std::cerr << "cannot write " << path << "\n";
            laszip_destroy(w);
            return false;
        }
        laszip_point* pt = nullptr;
        laszip_get_point_pointer(w, &pt);
        size_t n = 0;
        for (const auto& op : points_)
        {
            if (op.p.norm() <= min_range_)
                continue;
            const Eigen::Vector3d g = to_local[op.pose] * op.p;
            pt->intensity = op.intensity;
            pt->return_number = 1;
            pt->number_of_returns = 1;
            pt->gps_time = op.t * 1e9;
            laszip_F64 xyz[3] = { g.x(), g.y(), g.z() };
            if (laszip_set_coordinates(w, xyz) || laszip_write_point(w) || laszip_update_inventory(w))
            {
                std::cerr << "laszip write error in " << path << "\n";
                laszip_close_writer(w);
                laszip_destroy(w);
                return false;
            }
            ++n;
        }
        total_points_ += n;
        const bool ok = !laszip_close_writer(w);
        laszip_destroy(w);
        return ok;
    }

    void write_poses(const char* name)
    {
        std::ofstream f(std::filesystem::path(dir_) / name);
        f << std::setprecision(17) << scan_poses_.size() << "\n";
        for (size_t i = 0; i < scan_poses_.size(); ++i)
        {
            f << names_[i] << "\n";
            for (int r = 0; r < 3; ++r)
                f << scan_poses_[i](r, 0) << " " << scan_poses_[i](r, 1) << " " << scan_poses_[i](r, 2) << " " << scan_poses_[i](r, 3) << "\n";
            f << "0 0 0 1\n";
        }
    }

    std::string dir_;
    int chunks_per_scan_;
    double min_range_;
    bool write_laz_;
    std::function<double(double)> unix_of_t_;
    std::vector<mini_lio::PoseStamped> poses_;
    std::vector<mini_lio::OutputPoint> points_;
    std::vector<int> point_sizes_;
    std::vector<std::string> names_;
    std::vector<Eigen::Affine3d> scan_poses_;
    int chunks_in_scan_ = 0;
    size_t total_points_ = 0;
    bool ok_ = true;
};

} // namespace mandeye
