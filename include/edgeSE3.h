#pragma once
#include "g2o/core/base_binary_edge.h"
#include "vertexSE3.h"

class edgeSE3 : public g2o::BaseBinaryEdge<6, Sophus::SE3d, vertexSE3, vertexSE3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    edgeSE3()
    {
    }

    void computeError()
    {
        const vertexSE3 *v1 = static_cast<const vertexSE3 *>(_vertices[0]);
        const vertexSE3 *v2 = static_cast<const vertexSE3 *>(_vertices[1]);
        _error = ((v1->estimate().inverse() * v2->estimate()).inverse() * _measurement).log();
    }

    void setMeasurement(const Sophus::SE3d &m)
    {
        _measurement = m;
    }

    virtual bool read(std::istream &is)
    {
        return true;
    }
    virtual bool write(std::ostream &os) const
    {
        return true;
    }
};