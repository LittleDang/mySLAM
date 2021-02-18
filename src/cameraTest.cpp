#include "camera.h"
#include "config.h"
#include <iostream>
using namespace std;
using namespace mySLAM;
using namespace Eigen;
int main(int argc, char *argv[])
{
    config::setFileName("/home/dang/mySLAM/code/TUM1.yaml");
    auto configPtr = config::getInstance();
    float fx = configPtr->getData<float>("Camera.fx");
    float fy = configPtr->getData<float>("Camera.fy");
    float cx = configPtr->getData<float>("Camera.cx");
    float cy = configPtr->getData<float>("Camera.cy");
    camera cam(fx, fy, cx, cy);
    while (!cin.eof())
    {
        float x, y, z;
        cin >> x >> y >> z;
        Eigen::Vector3f WorldPt(x, y, z);
        Eigen::Vector2f pixel = cam * WorldPt;
        cout << "world pixel:" << pixel.transpose() << endl;
        cout << "reproject error:" << (cam.pixel2World(pixel, z) - WorldPt).norm() << endl;
    }
    return 0;
}