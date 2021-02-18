#include "frame.h"
namespace mySLAM
{
    void frame::computeORB()
    {
        //默认的参数，暂时不管，其实可以从config读入
        cv::Ptr<cv::ORB> detector = cv::ORB::create();
        detector->detectAndCompute(*color, cv::Mat(), keyPts, desptrors);
        std::vector<cv::KeyPoint> tKeyPts;
        cv::Mat tDesptors;
        for (size_t i = 0; i < keyPts.size(); i++)
        {
            float tempDepth = getDepthVaule(keyPts[i].pt.x, keyPts[i].pt.y);
            if (tempDepth < minDepth || tempDepth > maxDepth)
                continue;
            Eigen::Vector3f worldPt = cam_ptr->pixel2World(
                Eigen::Vector2f(keyPts[i].pt.x, keyPts[i].pt.y), tempDepth);

            //把点筛选一遍
            tKeyPts.push_back(keyPts[i]);
            tDesptors.push_back(desptrors.row(i));
            worldPts.push_back(worldPt);
        }
        keyPts = tKeyPts;
        desptrors = tDesptors;
    }
    std::shared_ptr<std::vector<cv::DMatch>> frame::matchFrames(std::shared_ptr<frame> &f, float eta)
    {

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        std::shared_ptr<std::vector<cv::DMatch>> matches = std::make_shared<std::vector<cv::DMatch>>();

        matcher->match(desptrors, f->desptrors, *matches);

        std::shared_ptr<std::vector<cv::DMatch>> goodMatches = std::make_shared<std::vector<cv::DMatch>>();
        float maxdist = 0;
        for (unsigned int i = 0; i < matches->size(); ++i)
        {
            maxdist = std::max(maxdist, (*matches)[i].distance);
        }
        for (unsigned int i = 0; i < matches->size(); ++i)
        {
            if ((*matches)[i].distance < maxdist * eta)
                goodMatches->push_back((*matches)[i]);
        }

        return goodMatches;
    }

    frame::frame(std::shared_ptr<camera> &camPtr, std::unique_ptr<cv::Mat> &_color, std::unique_ptr<cv::Mat> &_depth)
    {
        cam_ptr = camPtr;
        color = std::move(_color);
        depth = std::move(_depth);
        P = Eigen::Matrix4f::Identity();
        inv_P = Eigen::Matrix4f::Identity();

        //这个之后适合改成从配置文件读入
        minDepth = config::getInstance()->getData<float>("minDepth");
        maxDepth = config::getInstance()->getData<float>("maxDepth");
        depthFactor = config::getInstance()->getData<float>("DepthMapFactor");
        keyFrameMod = config::getInstance()->getData<int>("keyFrameMod");
        keyFrameCounts = 0;
        id = -1;
    }
    Eigen::Matrix4f frame::getTf()
    {
        return P;
    }
    void frame::setTf(Eigen::Matrix4f Pwc)
    {
        P = Pwc;
        inv_P = P.inverse();
    }
    const cv::Mat &frame::getColor() { return *color; }
    const cv::Mat &frame::getDepth() { return *depth; }
    const std::shared_ptr<camera> &frame::getCamPtr() { return cam_ptr; }
    float frame::getDepthVaule(int x, int y)
    {
        short d = depth->at<short>(y, x);
        float r = d / depthFactor;
        if (r < minDepth || r > maxDepth)
            return -1;
        return r;
    }
    Eigen::Vector3f frame::getWorldPt(int x, int y)
    {
        float d = getDepthVaule(x, y);
        if (d < 0)
            return Eigen::Vector3f(0, 0, 0);
        return cam_ptr->pixel2World(Eigen::Vector2f(x, y), d);
    }
    Eigen::Vector3f frame::getWorldPt(int index)
    {
        return worldPts[index];
    }
    const std::vector<cv::KeyPoint> &frame::getKeyPoints()
    {
        return keyPts;
    }
    void frame::setId(int _id)
    {
        id = _id;
    }
    int frame::getId()
    {
        return id;
    }
    cv::Mat frame::getDesptrors()
    {
        return desptrors;
    }

} // namespace mySLAM