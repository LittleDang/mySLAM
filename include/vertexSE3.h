#pragma once
#include "g2o/core/base_vertex.h"
#include <Eigen/Dense>
#include <sophus/se3.hpp>
//SE3理论上应该是6维的
class vertexSE3 : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    vertexSE3()
    {
    }
    //不需要io
    virtual bool read(std::istream &is)
    {
        return true;
    }
    virtual bool write(std::ostream &os) const
    {

        return os.good();
    }

    virtual void setToOriginImpl()
    {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update)
    {
        //对李代数求导，一个6维的更新量
        Sophus::Vector6d updateVectord = Sophus::Vector6d::ConstMapType(update);
        Sophus::SE3d updateSE3 = Sophus::SE3d::exp(updateVectord);
        _estimate = updateSE3 * _estimate;
    }
};