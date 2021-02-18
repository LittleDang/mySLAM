#include "viewer.h"
namespace mySLAM
{
    viewer::viewer(std::string loop, std::string aPath, std::string aMap,
                   std::string bPath, std::string bMap)
    {
        n = ros::NodeHandle();
        bPathPub = n.advertise<nav_msgs::Path>(bPath, 1000);
        aPathPub = n.advertise<nav_msgs::Path>(aPath, 1000);

        bMapPub = n.advertise<sensor_msgs::PointCloud2>(bMap, 1);
        aMapPub = n.advertise<sensor_msgs::PointCloud2>(aMap, 1);

        loopPub = n.advertise<visualization_msgs::Marker>(loop, 100);
        cloudPtr =
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    }
    void viewer::clear()
    {
        bTfs.clear();
        aTfs.clear();
        frames.clear();
    }
    void viewer::addFrame(std::shared_ptr<frame> &f)
    {
        if (f->getId() % config::getInstance()->getData<int>("keyFrameMod") != 0)
            frames.push_back(NULL);
        else
            frames.push_back(f);
    }
    void viewer::addLoop(std::vector<loopData> &_lps)
    {
        lps = _lps;
    }

    void viewer::addTfAAbsolute(Eigen::Matrix4f &tf)
    {
        Eigen::Matrix3f R = tf.block<3, 3>(0, 0).matrix();
        Eigen::Vector3f t = tf.block<3, 1>(0, 3).matrix();
        Sophus::SE3f tSE3(R, t);
        aTfs.push_back(tSE3);
    }

    void viewer::addBTf(Eigen::Matrix4f &tf)
    {
        Eigen::Matrix3f R = tf.block<3, 3>(0, 0).matrix();
        Eigen::Vector3f t = tf.block<3, 1>(0, 3).matrix();
        Sophus::SE3f tSE3(R, t);
        if (bTfs.empty())
            bTfs.push_back(tSE3);
        else
        {
            Sophus::SE3f newSE3 = bTfs[bTfs.size() - 1] * tSE3;
            bTfs.push_back(newSE3);
        }
    }
    void viewer::addTfBAbsolute(Eigen::Matrix4f &tf)
    {
        Eigen::Matrix3f R = tf.block<3, 3>(0, 0).matrix();
        Eigen::Vector3f t = tf.block<3, 1>(0, 3).matrix();
        Sophus::SE3f tSE3(R, t);
        bTfs.push_back(tSE3);
    }

    void viewer::publishBPath()
    {
        nav_msgs::Path path;
        for (int i = 0; i < bTfs.size(); i++)
        {
            geometry_msgs::PoseStamped p;
            p.pose.position.x = bTfs[i].translation().x();
            p.pose.position.y = bTfs[i].translation().y();
            p.pose.position.z = bTfs[i].translation().z();
            Eigen::Quaternionf q;
            q = bTfs[i].rotationMatrix();
            p.pose.orientation.x = q.x();
            p.pose.orientation.y = q.y();
            p.pose.orientation.z = q.z();
            p.pose.orientation.w = q.w();
            path.poses.push_back(p);
        }
        path.header.frame_id = "map";
        bPathPub.publish(path);

        if (!lps.empty())
        {
            visualization_msgs::Marker line_list;
            line_list.header.frame_id = "map";
            line_list.header.stamp = ros::Time::now();
            line_list.ns = "lines";
            line_list.action = visualization_msgs::Marker::ADD;
            line_list.pose.orientation.w = 1.0;
            line_list.id = 2;
            line_list.type = visualization_msgs::Marker::LINE_LIST;
            // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
            line_list.scale.x = 0.01;
            // Line list is red
            line_list.color.b = 1.0;
            line_list.color.a = 1.0;
            // Create the vertices for the points and lines
            for (int i = 0; i < lps.size(); i++)
            {
                int i1 = lps[i].index1;
                int i2 = lps[i].index2;
                geometry_msgs::Point p1, p2;

                p1.x = path.poses[i1].pose.position.x;
                p1.y = path.poses[i1].pose.position.y;
                p1.z = path.poses[i1].pose.position.z;
                line_list.points.push_back(p1);

                p2.x = path.poses[i2].pose.position.x;
                p2.y = path.poses[i2].pose.position.y;
                p2.z = path.poses[i2].pose.position.z;
                line_list.points.push_back(p2);
            }
            loopPub.publish(line_list);
            SUCCESS_OUTPUT("send loop");
        }
    }
    void viewer::publishAPath()
    {
        nav_msgs::Path path;
        for (int i = 0; i < aTfs.size(); i++)
        {
            geometry_msgs::PoseStamped p;
            p.pose.position.x = aTfs[i].translation().x();
            p.pose.position.y = aTfs[i].translation().y();
            p.pose.position.z = aTfs[i].translation().z();
            Eigen::Quaternionf q;
            q = aTfs[i].rotationMatrix();
            p.pose.orientation.x = q.x();
            p.pose.orientation.y = q.y();
            p.pose.orientation.z = q.z();
            p.pose.orientation.w = q.w();
            path.poses.push_back(p);
        }
        path.header.frame_id = "map";
        aPathPub.publish(path);
    }

    void viewer::publishAMap()
    {

        if (frames.size() != aTfs.size())
        {
            ERROR_OUTPUT("frames.size()!=aTfs.size()");
            exit(0);
        }
        OPEN_GREEN;
        std::cout << "begin to generate dense map" << std::endl;
        for (int k = 0; k < frames.size(); k++)
        {
            if (frames[k] == NULL)
                continue;
            std::cout << "\rCurrent progress:" << k + 1 << "/" << frames.size();
            auto depth = frames[k]->getDepth();
            auto color = frames[k]->getColor();

            for (int i = 0; i < depth.rows; i++)
            {
                for (int j = 0; j < depth.cols; j++)
                {
                    Eigen::Vector3f w = frames[k]->getWorldPt(j, i);
                    Eigen::Vector3f worldPt = aTfs[k] * w;
                    pcl::PointXYZRGB p;
                    p.x = worldPt.x();
                    p.y = worldPt.y();
                    p.z = worldPt.z();
                    p.r = color.at<cv::Vec3b>(i, j)[2];
                    p.g = color.at<cv::Vec3b>(i, j)[1];
                    p.b = color.at<cv::Vec3b>(i, j)[0];
                    cloudPtr->push_back(p);
                }
            }
        }
        std::cout << "\nbegin to filter ... ";
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloudPtr);
        sor.setLeafSize(0.01f, 0.01f, 0.01f); //体素大小设置为30*30*30cm
        sor.filter(*cloudPtr);
        pcl::toROSMsg(*cloudPtr, pclMsg);
        std::cout << pclMsg.data.size() << std::endl;
        pclMsg.header.frame_id = "map";
        aMapPub.publish(pclMsg);
        std::cout << "done" << std::endl;
        CLOSE_COLOR;
    }

    void viewer::publishBMap()
    {
        if (frames.size() != bTfs.size())
        {
            ERROR_OUTPUT("frames.size()!=bTfs.size()");
            exit(0);
        }
        OPEN_GREEN;
        std::cout << "begin to generate dense map" << std::endl;
        for (int k = 0; k < frames.size(); k++)
        {
            if (frames[k] == NULL)
                continue;
            std::cout << "\rCurrent progress:" << k + 1 << "/" << frames.size();
            auto depth = frames[k]->getDepth();
            auto color = frames[k]->getColor();

            for (int i = 0; i < depth.rows; i++)
            {
                for (int j = 0; j < depth.cols; j++)
                {
                    Eigen::Vector3f w = frames[k]->getWorldPt(j, i);
                    Eigen::Vector3f worldPt = bTfs[k] * w;
                    pcl::PointXYZRGB p;
                    p.x = worldPt.x();
                    p.y = worldPt.y();
                    p.z = worldPt.z();
                    p.r = color.at<cv::Vec3b>(i, j)[2];
                    p.g = color.at<cv::Vec3b>(i, j)[1];
                    p.b = color.at<cv::Vec3b>(i, j)[0];
                    cloudPtr->push_back(p);
                }
            }
        }
        std::cout << "\nbegin to filter ... ";
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloudPtr);
        sor.setLeafSize(0.01f, 0.01f, 0.01f); //体素大小设置为30*30*30cm
        sor.filter(*cloudPtr);
        pcl::toROSMsg(*cloudPtr, pclMsg);
        pclMsg.header.frame_id = "map";
        bMapPub.publish(pclMsg);
        std::cout << "done" << std::endl;
    }

}; // namespace mySLAM
