#pragma once
#include "config.h"
#include "edgeSE3.h"
#include "frame.h"
#include "vertexSE3.h"
#include "vo.h"
#include <DBoW3/DBoW3.h>
namespace mySLAM
{
    struct loopData
    {
        int index1;
        int index2;
        Eigen::Matrix4f T;
        loopData(int i1, int i2, Eigen::Matrix4f _T)
        {
            index1 = i1;
            index2 = i2;
            T = _T;
        }
    };
    class loop
    {
    public:
        loop(std::vector<Eigen::Matrix4f> _tfs, std::vector<std::shared_ptr<frame>> _keyFrames);
        std::vector<Eigen::Matrix4f> computeResult();
        std::vector<loopData> getLoops();

        static void getRT(Eigen::Matrix4f &tf, Eigen::Matrix3d &R, Eigen::Vector3d &T);
        static void matrix4d2matrix4f(Eigen::Matrix4d &src, Eigen::Matrix4f &dst);

    private:
        std::vector<Sophus::SE3d> tfs;
        std::vector<Sophus::SE3d> tfs_absolute;

        std::vector<std::shared_ptr<frame>> keyFrames;
        DBoW3::Vocabulary vocab;
        double vocScores;
        std::vector<loopData> lds;
        int orbThreshold;
    };
} // namespace mySLAM
