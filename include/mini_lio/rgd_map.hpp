// Regular-grid-decomposition NDT map (HDMapping update_rgd_ray, same update rules).
#pragma once

#include "types.hpp"

#include <ankerl/unordered_dense.h>

#include <algorithm>

namespace mini_lio
{

struct Bucket
{
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d cov_inverse = Eigen::Matrix3d::Zero();
    Eigen::Vector3d mean = Eigen::Vector3d::Zero(); // first point of the cell (upstream never moves it)
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    uint32_t n = 0;
    uint8_t hits = 0; // times a ray passed through the cell (stale/dynamic cells are skipped at >= 20)
};

// Truncating int16 cell index, exactly like get_rgd_index_3d (cell 0 is twice as wide; +-3.2 km at 0.1 m).
inline uint64_t rgd_key(const Eigen::Vector3d& p, const Eigen::Vector3d& b)
{
    const auto x = static_cast<uint16_t>(static_cast<int16_t>(p.x() / b.x()));
    const auto y = static_cast<uint16_t>(static_cast<int16_t>(p.y() / b.y()));
    const auto z = static_cast<uint16_t>(static_cast<int16_t>(p.z() / b.z()));
    return (static_cast<uint64_t>(x) << 32) | (static_cast<uint64_t>(y) << 16) | static_cast<uint64_t>(z);
}

class RgdMap
{
public:
    explicit RgdMap(double resolution = 0.1)
        : b_(resolution, resolution, resolution)
        , b_norm_(b_.norm())
    {
    }

    void clear() { map_.clear(); }
    size_t size() const { return map_.size(); }
    double resolution() const { return b_.x(); }

    const Bucket* find(const Eigen::Vector3d& p) const { return findKey(key(p)); }
    uint64_t key(const Eigen::Vector3d& p) const { return rgd_key(p, b_); }
    const Bucket* findKey(uint64_t k) const
    {
        const auto it = map_.find(k);
        return it == map_.end() ? nullptr : &it->second;
    }

    const ankerl::unordered_dense::map<uint64_t, Bucket>& cells() const { return map_; }

    void update(const std::vector<Eigen::Vector3d>& pts, const Eigen::Vector3d& viewport, bool ray_cast)
    {
        for (const auto& p : pts)
        {
            const uint64_t key = rgd_key(p, b_);
            auto it = map_.find(key);
            if (it != map_.end())
            {
                Bucket& bk = it->second;
                bk.n++;
                const Eigen::Vector3d d = bk.mean - p;
                const Eigen::Matrix3d cov_update = d * d.transpose();
                const double n = static_cast<double>(bk.n);
                const auto upd = [&]()
                {
                    bk.cov = bk.cov * (n - 1) / n + cov_update * (n - 1) / (n * n);
                    bk.cov_inverse = bk.cov.inverse();
                };
                if (bk.n == 2)
                {
                    upd();
                }
                else if (bk.n == 3)
                {
                    upd();
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(bk.cov, Eigen::ComputeEigenvectors);
                    const Eigen::Matrix3d ev = es.eigenvectors();
                    Eigen::Vector3d nv = ev.col(1).cross(ev.col(2));
                    nv.normalize();
                    if (nv.dot(viewport - bk.mean) < 0.0)
                        nv = -nv;
                    bk.normal = nv;
                }
                else if (bk.n > 3)
                {
                    if (bk.normal.dot(viewport - bk.mean) >= 0.0) // only integrate points seen from the front
                        upd();
                }
            }
            else
            {
                Bucket bk;
                bk.mean = p;
                bk.cov = Eigen::Matrix3d::Identity() * 0.03 * 0.03;
                bk.cov_inverse = bk.cov.inverse();
                bk.n = 1;
                map_.emplace(key, bk);
            }

            if (ray_cast)
            {
                Eigen::Vector3d dir = viewport - p;
                const double dist = dir.norm();
                int steps = static_cast<int>(std::ceil(dist / b_norm_));
                dir.normalize();
                steps -= 1;
                if (steps > 1)
                {
                    for (int j = 1; j < steps; ++j)
                    {
                        const Eigen::Vector3d q = p + dir * (b_norm_ * (j + 1));
                        auto qi = map_.find(rgd_key(q, b_));
                        if (qi != map_.end() && qi->second.hits < 30)
                            qi->second.hits++;
                    }
                }
            }
        }
    }

private:
    Eigen::Vector3d b_;
    double b_norm_;
    ankerl::unordered_dense::map<uint64_t, Bucket> map_;
};

} // namespace mini_lio
