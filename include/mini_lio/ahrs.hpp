// Orientation from the IMU: VQF or Fusion (Madgwick), same usage as HDMapping calculate_trajectory().
#pragma once

#include "types.hpp"

#include <vqf.hpp>

extern "C"
{
#include <Fusion.h>
}

#include <algorithm>
#include <memory>
#include <vector>

namespace mini_lio
{

class Ahrs
{
public:
    explicit Ahrs(const Params& p)
        : p_(p)
    {
        if (p_.use_vqf)
        {
            VQFParams vp;
            vp.tauAcc = p_.vqf_tau_acc > 0.0 ? p_.vqf_tau_acc : 3.0;
#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
            vp.motionBiasEstEnabled = p_.vqf_motion_bias_est;
            vp.biasSigmaMotion = p_.vqf_bias_sigma_motion;
            vp.biasVerticalForgettingFactor = p_.vqf_bias_vertical_forgetting;
#endif
            vp.restBiasEstEnabled = p_.vqf_rest_bias_est;
            vp.biasSigmaInit = p_.vqf_bias_sigma_init;
            vp.biasForgettingTime = p_.vqf_bias_forgetting_time;
            vp.biasClip = p_.vqf_bias_clip;
            vp.biasSigmaRest = p_.vqf_bias_sigma_rest;
            vp.restMinT = p_.vqf_rest_min_t;
            vp.restFilterTau = p_.vqf_rest_filter_tau;
            vp.restThGyr = p_.vqf_rest_th_gyr;
            vp.restThAcc = p_.vqf_rest_th_acc;
            vqf_ = std::make_unique<VQF>(vp, 1.0 / p_.imu_rate_hz);
        }
        else
        {
            FusionAhrsInitialise(&fusion_);
            fusion_.settings.convention = FusionConventionNwu;
            fusion_.settings.gain = static_cast<float>(p_.fusion_gain);
        }
    }

    // Feed one sample, returns the sensor->world rotation after the update.
    // With fusion_remove_gyro_bias the first 1000 samples are buffered (ready() == false) and replayed.
    bool update(const ImuSample& s, std::vector<std::pair<double, Eigen::Quaterniond>>& out)
    {
        if (!p_.use_vqf && p_.fusion_remove_gyro_bias && !bias_ready_)
        {
            warmup_.push_back(s);
            if (warmup_.size() < kBiasSamples)
                return false;
            std::vector<double> gx, gy, gz;
            for (const auto& w : warmup_)
            {
                gx.push_back(w.gyro.x() * kRadToDeg);
                gy.push_back(w.gyro.y() * kRadToDeg);
                gz.push_back(w.gyro.z() * kRadToDeg);
            }
            const size_t m = kBiasSamples / 2;
            std::nth_element(gx.begin(), gx.begin() + m, gx.end());
            std::nth_element(gy.begin(), gy.begin() + m, gy.end());
            std::nth_element(gz.begin(), gz.begin() + m, gz.end());
            bias_dps_ = Eigen::Vector3d(gx[m], gy[m], gz[m]);
            bias_ready_ = true;
            for (const auto& w : warmup_)
                out.emplace_back(w.t, step(w));
            warmup_.clear();
            return true;
        }
        out.emplace_back(s.t, step(s));
        return true;
    }

    const Eigen::Vector3d& gyroBiasDps() const { return bias_dps_; }

private:
    static constexpr double kRadToDeg = 180.0 / M_PI;
    static constexpr size_t kBiasSamples = 1000;

    Eigen::Quaterniond step(const ImuSample& s)
    {
        if (p_.use_vqf)
        {
            constexpr double g = 9.80665;
            vqf_real_t gyr[3] = { s.gyro.x(), s.gyro.y(), s.gyro.z() };
            vqf_real_t acc[3] = { s.acc.x() * g, s.acc.y() * g, s.acc.z() * g };
            vqf_->update(gyr, acc);
            vqf_real_t q[4];
            vqf_->getQuat6D(q);
            return Eigen::Quaterniond(q[0], q[1], q[2], q[3]);
        }

        const FusionVector gyro = { static_cast<float>(s.gyro.x() * kRadToDeg - bias_dps_.x()),
                                    static_cast<float>(s.gyro.y() * kRadToDeg - bias_dps_.y()),
                                    static_cast<float>(s.gyro.z() * kRadToDeg - bias_dps_.z()) };
        const FusionVector acc = { static_cast<float>(s.acc.x()), static_cast<float>(s.acc.y()), static_cast<float>(s.acc.z()) };
        const float dt = first_ ? 1.0f / 200.0f : static_cast<float>(s.t - last_t_);
        first_ = false;
        last_t_ = s.t;
        FusionAhrsUpdateNoMagnetometer(&fusion_, gyro, acc, dt);
        const FusionQuaternion q = FusionAhrsGetQuaternion(&fusion_);
        return Eigen::Quaterniond(q.element.w, q.element.x, q.element.y, q.element.z);
    }

    Params p_;
    std::unique_ptr<VQF> vqf_;
    FusionAhrs fusion_;
    bool first_ = true;
    double last_t_ = 0.0;
    bool bias_ready_ = false;
    Eigen::Vector3d bias_dps_ = Eigen::Vector3d::Zero();
    std::vector<ImuSample> warmup_;
};

} // namespace mini_lio
