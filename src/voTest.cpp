#include "vo.h"
#include "viewer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
using namespace mySLAM;
using namespace Eigen;

void callback(int event, int x, int y, int flags, void *param);

void LoadImage(vector<string> &vstrImageFilenamesRGB,
               vector<string> &vstrImageFilenamesD,
               vector<double> &vTimestamps, string &path);

shared_ptr<camera> createCam();
std::shared_ptr<frame> cf;

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "viewer");
    config::setFileName("/home/dang/mySLAM/code/TUM1.yaml"); //全局调用的，必须要第一行执行比较好

    string path = "/home/dang/ORB_SLAM2/dataset/rgbd_dataset_freiburg1_desk/";
    cf = NULL;
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    LoadImage(vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps, path);
    auto camPtr = createCam();
    bool stop = false;
    namedWindow("color");
    namedWindow("depth");

    setMouseCallback("color", callback);
    setMouseCallback("depth", callback);
    vo myVO(camPtr);
    viewer toRos("mypath", "mymap");
    for (size_t i = 0; i < vstrImageFilenamesRGB.size(); i++)
    {
        std::unique_ptr<cv::Mat> color(
            new Mat(cv::imread(path + vstrImageFilenamesRGB[i], CV_LOAD_IMAGE_UNCHANGED)));
        std::unique_ptr<cv::Mat> depth(
            new Mat(cv::imread(path + vstrImageFilenamesD[i], CV_LOAD_IMAGE_UNCHANGED)));
        std::shared_ptr<frame> f = std::make_shared<frame>(camPtr, color, depth);
        f->computeORB();
        auto tf = myVO.addFrame(f);
        toRos.addBTf(tf);
        toRos.addFrame(f);
        toRos.publishBPath();
        cf = f;
        imshow("color", f->getColor());
        imshow("depth", f->getDepth());
        uchar key = waitKey(30);
        if (key == 'q')
            break;
        if (key == ' ')
            stop = true;
        if (stop)
            waitKey(0);
        stop = false;
    }
    toRos.publishBMap();
    return 0;
}

void LoadImage(vector<string> &vstrImageFilenamesRGB,
               vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps, string &path)
{
    ifstream file_in;
    file_in.open(path + "associations.txt");
    while (!file_in.eof())
    {
        string s;
        getline(file_in, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
    file_in.close();
}
shared_ptr<camera> createCam()
{
    auto configPtr = config::getInstance();
    float fx = configPtr->getData<float>("Camera.fx");
    float fy = configPtr->getData<float>("Camera.fy");
    float cx = configPtr->getData<float>("Camera.cx");
    float cy = configPtr->getData<float>("Camera.cy");
    return make_shared<camera>(fx, fy, cx, cy);
}
void callback(int event, int x, int y, int flags, void *param)
{
    if (event == EVENT_LBUTTONUP)
    {
        auto worldPt = cf->getWorldPt(x, y);
        if (worldPt.z() > 0)
            cout << "pixel=[" << x << "," << y << "]. world=["
                 << worldPt.x() << "," << worldPt.y() << "," << worldPt.z() << endl;
        else
            cout << "\033[0;31merror:out of range\033[0m" << endl;
    }
}