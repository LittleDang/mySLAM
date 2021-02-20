#pragma once
#include "config.h"
#include <frame.h>
#include <memory>
#include <vector>
namespace mySLAM
{
    class vo
    {
    private:
        //存放所有的帧，其实存进来的就是关键帧了，没存进来的就不配称为关键帧
        std::shared_ptr<frame> lastFrame;
        std::shared_ptr<frame> currentFrame;

        std::vector<Eigen::Matrix4f> tfs;              //记录下每一帧的相对位姿，给后端用的
        std::vector<std::shared_ptr<frame>> keyFrames; //记录下关键帧，用来回环检测

        std::shared_ptr<camera> cam_ptr;
        int id;
        //参数
        bool isFirstFrame;
        int errorCounts;
        int errorThreshold;
        int orbThreshold;
        int keyFrameMod;

    public:
        std::vector<Eigen::Matrix4f> getTFs();
        std::vector<std::shared_ptr<frame>> getKeyFrames();
        vo(std::shared_ptr<camera> &camPtr);
        Eigen::Matrix4f addFrame(std::shared_ptr<frame> &f);
        //返回的是T12
        static void estimateRigid3D(std::vector<Eigen::Vector3f> &pt1,
                                    std::vector<Eigen::Vector3f> &pt2, Eigen::Matrix3f &R, Eigen::Vector3f &t);
        static bool estimateRigid3DRansac(std::vector<Eigen::Vector3f> &pt1,
                                          std::vector<Eigen::Vector3f> &pt2,
                                          int maxIters, float error, std::vector<int> &inlines, Eigen::Matrix3f &R, Eigen::Vector3f &t);
        static float estimateReprojection(std::vector<Eigen::Vector3f> &pt1,
                                          std::vector<Eigen::Vector3f> &pt2,
                                          Eigen::Matrix3f &R, Eigen::Vector3f &t,
                                          std::vector<int> inlines);
    };
} // namespace mySLAM