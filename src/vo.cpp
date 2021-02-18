#include "vo.h"
namespace mySLAM
{
    Eigen::Matrix4f vo::addFrame(std::shared_ptr<frame> &f)
    {
        f->setId(id);
        if (id % keyFrameMod == 0)
        {
            keyFrames.push_back(f);
        }
        id++;

        currentFrame = f;
        if (isFirstFrame)
        {
            isFirstFrame = false;
            lastFrame = currentFrame;
            errorCounts = 0;
            tfs.push_back(Eigen::Matrix4f::Identity());
            return Eigen::Matrix4f::Identity();
        }
        else
        {
            auto matches = currentFrame->matchFrames(lastFrame, 0.4);
            if (errorCounts >= errorThreshold)
            {
                ERROR_OUTPUT("lose.so quit");
                exit(0);
            }
            if (matches->size() < orbThreshold)
            {
                errorCounts++;
                lastFrame = currentFrame;
                cv::waitKey(0);
                tfs.push_back(Eigen::Matrix4f::Identity());
                return Eigen::Matrix4f::Identity();
            }
            errorCounts = 0;

            // cv::Mat board;
            // cv::drawMatches(currentFrame->getColor(),
            //                 currentFrame->getKeyPoints(),
            //                 lastFrame->getColor(),
            //                 lastFrame->getKeyPoints(),
            //                 *matches, board);

            // cv::imshow("matches", board);
            // cv::waitKey(0);

            std::vector<Eigen::Vector3f> ps1;
            std::vector<Eigen::Vector3f> ps2;
            for (size_t i = 0; i < matches->size(); i++)
            {
                ps1.push_back(currentFrame->getWorldPt(matches->at(i).queryIdx));
                ps2.push_back(lastFrame->getWorldPt(matches->at(i).trainIdx));
            }
            //SUCCESS_OUTPUT(ps1.size());
            //SUCCESS_OUTPUT(ps2.size());

            //这个函数返回的tf是 ps2=tf*ps1
            //ps2是上一帧的，所以这个tf就是Plc
            //pl=plc*Pc
            Eigen::Matrix3f R;
            Eigen::Vector3f t;
            estimateRigid3D(ps2, ps1, R, t);
            lastFrame = currentFrame;
            Eigen::Matrix4f resultTF = Eigen::Matrix4f::Identity();
            resultTF.block<3, 3>(0, 0) = R;
            resultTF.block<3, 1>(0, 3) = t;
            tfs.push_back(resultTF);
            return resultTF;
        }
    }
    void vo::estimateRigid3D(std::vector<Eigen::Vector3f> &pt1,
                             std::vector<Eigen::Vector3f> &pt2, Eigen::Matrix3f &R, Eigen::Vector3f &t)
    {
        std::vector<Eigen::Vector3f> q1, q2;
        Eigen::Vector3f c1 = Eigen::Vector3f::Zero(), c2 = Eigen::Vector3f::Zero();
        float n = pt1.size();
        for (size_t i = 0; i < n; i++)
        {
            c1 += pt1[i];
            c2 += pt2[i];
        }
        c1 /= n;
        c2 /= n;

        for (size_t i = 0; i < n; i++)
        {
            q1.push_back(pt1[i] - c1);
            q2.push_back(pt2[i] - c2);
        }

        Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < n; i++)
            W += q1[i] * q2[i].transpose();

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();
        R = U * V.transpose();
        if (R.determinant() < 0.5)
        {
            R = Eigen::Matrix3f::Identity();
            t = Eigen::Vector3f::Zero();
            return;
        }
        t = c1 - R * c2;
    }

    std::vector<Eigen::Matrix4f> vo::getTFs()
    {
        return tfs;
    }
    std::vector<std::shared_ptr<frame>> vo::getKeyFrames()
    {
        return keyFrames;
    }
    vo::vo(std::shared_ptr<camera> &camPtr)
    {
        isFirstFrame = true;
        errorCounts = 0;
        errorThreshold = config::getInstance()->getData<int>("errorThreshold");
        orbThreshold = config::getInstance()->getData<int>("orbThreshold");
        keyFrameMod = config::getInstance()->getData<int>("keyFrameMod");
        cam_ptr = camPtr;
        id = 0;
    }

    float vo::estimateReprojection(std::vector<Eigen::Vector3f> &pt1,
                                   std::vector<Eigen::Vector3f> &pt2, Eigen::Matrix3f &R, Eigen::Vector3f &t)
    {
        float re = 0;
        for (int i = 0; i < pt1.size(); i++)
            re += (R * pt1[i] + t - pt2[i]).norm();
        re /= pt1.size();
        return re;
    }

} // namespace mySLAM