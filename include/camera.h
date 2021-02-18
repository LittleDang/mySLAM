#pragma once
#include <Eigen/Dense>
#include "config.h"

namespace mySLAM
{
    class camera
    {
    private:
        Eigen::Matrix3f K;     //内参数矩阵
        Eigen::Matrix3f inv_K; //内参数矩阵的逆
        camera()
        {
        }

    public:
        camera(const float &fx, const float &fy, const float &cx, const float &cy)
        {
            K << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f;
            inv_K = K.inverse();
        }
        Eigen::Vector2f world2Pixel(const Eigen::Vector3f &worldPt) const
        {
            Eigen::Vector3f i = K * worldPt;
            i /= i[2];
            return Eigen::Vector2f(i[0], i[1]);
        }
        Eigen::Vector3f pixel2World(const Eigen::Vector2f &pixel, const float &z) const
        {
            Eigen::Vector3f i(pixel[0], pixel[1], 1);
            i *= z;
            return inv_K * i;
        }
        Eigen::Vector2f operator*(const Eigen::Vector3f &worldPt) const
        {
            return this->world2Pixel(worldPt);
        }
    };
} // namespace mySLAM