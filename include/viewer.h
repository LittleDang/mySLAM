#pragma once
#include "config.h"
#include "frame.h"
#include "loop.h"
#include <Eigen/Dense>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h> //体素滤波相关
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <sophus/se3.hpp>
#include <vector>
#include <visualization_msgs/Marker.h>

namespace mySLAM
{
    class viewer
    {
    private:
        ros::NodeHandle n;

        ros::Publisher bPathPub; //显示优化之前的路径
        ros::Publisher bMapPub;  //显示优化之前的稠密地图

        ros::Publisher aPathPub; //显示优化之后的路径
        ros::Publisher aMapPub;  //显示优化之后的稠密地图

        ros::Publisher loopPub; //显示回环用的

        std::vector<Sophus::SE3f> bTfs;
        std::vector<Sophus::SE3f> aTfs;

        std::vector<std::shared_ptr<frame>> frames;
        std::vector<loopData> lps;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr;

        sensor_msgs::PointCloud2 pclMsg;

    public:
        viewer(std::string loop = "myloop", std::string aPath = "myapath",
               std::string aMap = "myamap", std::string bPath = "mybpath",
               std::string bMap = "mybmap");

        void clear();
        void addFrame(std::shared_ptr<frame> &f);
        void addLoop(std::vector<loopData> &_lps);

        //优化后的
        void addTfAAbsolute(Eigen::Matrix4f &tf); //绝对的tf
        //发布优化后的
        void publishAPath();
        void publishAMap();

        //优化前的
        void addBTf(Eigen::Matrix4f &tf);         //f-f的tf，最后也是转化成绝对的tf
        void addTfBAbsolute(Eigen::Matrix4f &tf); //绝对的tf
        //发布前优化的
        void publishBPath();
        void publishBMap();
    };
} // namespace mySLAM