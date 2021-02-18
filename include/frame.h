#pragma once
#include "config.h"
#include <Eigen/Dense>
#include <camera.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
namespace mySLAM
{

    class frame
    {
    private:
        std::shared_ptr<camera> cam_ptr; //指向一个cam,只有一个
        Eigen::Matrix4f P;               //外参数 Pw_c
        Eigen::Matrix4f inv_P;           //外参数 P_cw

        std::unique_ptr<cv::Mat> color; //彩色图
        std::unique_ptr<cv::Mat> depth; //深度图

        std::vector<cv::KeyPoint> keyPts;      //属于这一帧的关键点
        cv::Mat desptrors;                     //关键点对应的描述子
        std::vector<Eigen::Vector3f> worldPts; //对应的相机坐标系里面的三维坐标

        int keyFrameCounts;

        //参数
        float minDepth;
        float maxDepth;
        float depthFactor;
        int keyFrameMod; //每隔多少帧取一张关键帧
        int id;

    protected:
    public:
        frame(std::shared_ptr<camera> &camPtr,
              std::unique_ptr<cv::Mat> &_color, std::unique_ptr<cv::Mat> &_depth);
        void computeORB(); //计算orb特征
        Eigen::Matrix4f getTf();
        void setTf(Eigen::Matrix4f Pwc);
        std::shared_ptr<std::vector<cv::DMatch>> matchFrames(std::shared_ptr<frame> &f, float eta);
        const cv::Mat &getColor();
        const cv::Mat &getDepth();
        const std::shared_ptr<camera> &getCamPtr();
        float getDepthVaule(int x, int y);
        Eigen::Vector3f getWorldPt(int x, int y);
        Eigen::Vector3f getWorldPt(int index);
        const std::vector<cv::KeyPoint> &getKeyPoints();
        void setId(int _id);
        int getId();
        cv::Mat getDesptrors();
    };

} // namespace mySLAM