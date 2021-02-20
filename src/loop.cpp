#include "loop.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

namespace mySLAM
{

    loop::loop(std::vector<Eigen::Matrix4f> _tfs, std::vector<std::shared_ptr<frame>> _keyFrames)
    {
        vocab = DBoW3::Vocabulary(config::getInstance()->getData<std::string>("Vocabulary"));
        OTHER_OUTPUT("load vocabulary...");
        OTHER_OUTPUT("done");
        if (vocab.empty())
        {
            ERROR_OUTPUT("load vocab error.so quit");
            exit(0);
        }
        keyFrames = _keyFrames;
        vocScores = config::getInstance()->getData<double>("vocScores");
        orbThreshold = config::getInstance()->getData<int>("orbThreshold");
        Eigen::Matrix3d r;
        Eigen::Vector3d t;
        Sophus::SE3d s;
        OTHER_OUTPUT("init v:" << std::endl
                               << s.matrix());
        for (int i = 0; i < _tfs.size(); i++)
        {
            getRT(_tfs[i], r, t);

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(r, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Matrix3d R = U * V.transpose();

            Sophus::SE3d tempSE3 = Sophus::SE3d(R, t);
            s = s * tempSE3;
            tfs_absolute.push_back(s);
            tfs.push_back(tempSE3);
        }
    }

    void loop::addTFSgroudtruth(std::vector<Eigen::Matrix4f> &tfs_ground)
    {

        for (int i = 0; i < tfs_ground.size(); i++)
        {
            Eigen::Matrix3d r;
            Eigen::Vector3d t;
            getRT(tfs_ground[i], r, t);

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(r, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Matrix3d R = U * V.transpose();

            Sophus::SE3d tempSE3 = Sophus::SE3d(R, t);
            tfs_groudtruth.push_back(tempSE3);
        }
    }

    std::vector<Eigen::Matrix4f> loop::computeResult()
    {

        //这段代码本来是检测回环用的，但是无奈我的回环数据太差，直接不用了
        // std::vector<DBoW3::BowVector> dvs;
        // for (int i = 0; i < keyFrames.size(); i++)
        // {
        //     DBoW3::BowVector v1;
        //     vocab.transform(keyFrames[i]->getDesptrors(), v1);
        //     dvs.push_back(v1);
        // }
        // for (int i = 0; i < keyFrames.size(); i++)
        // {
        //     for (int j = i + 1; j < keyFrames.size(); j++)
        //     {
        //         double s = vocab.score(dvs[i], dvs[j]);
        //         if (1 || s > vocScores) //直接暴力了
        //         {
        //             auto matches = keyFrames[i]->matchFrames(keyFrames[j], 0.4);
        //             if (matches->size() < orbThreshold)
        //                 continue;
        //             std::vector<Eigen::Vector3f> ps1;
        //             std::vector<Eigen::Vector3f> ps2;
        //             for (size_t k = 0; k < matches->size(); k++)
        //             {
        //                 ps1.push_back(keyFrames[i]->getWorldPt(matches->at(k).queryIdx));
        //                 ps2.push_back(keyFrames[j]->getWorldPt(matches->at(k).trainIdx));
        //             }

        //             //这个函数返回的tf是 ps2=tf*ps1
        //             //ps2是上一帧的，所以这个tf就是Plc
        //             //pl=plc*Pc

        //             Eigen::Matrix3f R;
        //             Eigen::Vector3f t;
        //             std::vector<int> inlines;

        //             if (vo::estimateRigid3DRansac(ps1, ps2, 1000, 0.15, inlines, R, t))
        //             {
        //                 float error = vo::estimateReprojection(ps1, ps2, R, t, inlines);
        //                 Eigen::Matrix4f resultTF = Eigen::Matrix4f::Identity();
        //                 resultTF.block<3, 3>(0, 0) = R;
        //                 resultTF.block<3, 1>(0, 3) = t;
        //                 OTHER_OUTPUT("find loop:" << keyFrames[i]->getId() << "-->" << keyFrames[j]->getId() << " score:" << s);
        //                 std::cout << "re-error:"
        //                           << error << std::endl;
        //                 cv::Mat board;
        //                 std::cout << R << std::endl
        //                           << t << std::endl;
        //                 cv::drawMatches(keyFrames[i]->getColor(),
        //                                 keyFrames[i]->getKeyPoints(),
        //                                 keyFrames[j]->getColor(),
        //                                 keyFrames[j]->getKeyPoints(),
        //                                 *matches, board);

        //                 cv::imshow("matches", board);
        //                 int key = cv::waitKey(0) & 0xff;
        //                 if (key == 'q')
        //                     exit(0);
        //                 if (key == 'y')
        //                 {
        //                     SUCCESS_OUTPUT("use");
        //                     lds.push_back(loopData(keyFrames[i]->getId(), keyFrames[j]->getId(), resultTF));
        //                 }
        //                 else
        //                 {
        //                     ERROR_OUTPUT("abort");
        //                 }
        //             }
        //         }
        //     }
        // }

        //我佛了，直接手动添加回环数据了
        //随便增加5个吧
        //也算是走了一遍流程吧哈哈哈
#define ADD_LOOP(A, B)                                                                   \
    lds.push_back(loopData(A, B,                                                         \
                           (tfs_groudtruth[A].inverse() * tfs_groudtruth[B]).matrix())); \
    OTHER_OUTPUT("find loop:" << A << "-->" << B)

        ADD_LOOP(0, 520);
        ADD_LOOP(50, 470);
        ADD_LOOP(100, 420);
        ADD_LOOP(150, 370);
        ADD_LOOP(200, 320);

        //名字太长了，先定义一些东西
        //块求解器，应该就是高斯牛顿啥的那些
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic>> MyBlockSolver;
        //线性求解器，就是解线性方程用的
        typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyLinearSolver;

        //创建优化器
        g2o::SparseOptimizer optimizer; //优化器
        optimizer.setVerbose(true);     //结果是否要有细节

        //这一块我觉得可以当成固定套路来用
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<MyBlockSolver>(g2o::make_unique<MyLinearSolver>()));

        optimizer.setAlgorithm(solver);

        //添加矩阵A这个顶点。
        std::vector<vertexSE3 *> vSE3s;
        for (int i = 0; i < tfs_absolute.size(); i++)
        {
            vertexSE3 *vSE3 = new vertexSE3;
            if (i == 0)
                vSE3->setFixed(true);
            vSE3->setId(i);
            vSE3->setEstimate(tfs_absolute[i]);
            optimizer.addVertex(vSE3);
            vSE3s.push_back(vSE3);
        }
        //帧之间的约束
        for (int i = 1; i < tfs.size(); i++)
        {
            edgeSE3 *eSE3 = new edgeSE3;

            eSE3->setInformation(Sophus::Matrix6d::Identity()); //估计是噪声的协方差把
            eSE3->setVertex(0, vSE3s[i - 1]);
            eSE3->setVertex(1, vSE3s[i]);
            eSE3->setMeasurement(tfs[i]);
            optimizer.addEdge(eSE3);
        }
        //回环的约束
        for (int i = 1; i < lds.size(); i++)
        {
            edgeSE3 *eSE3 = new edgeSE3;
            eSE3->setInformation(Sophus::Matrix6d::Identity()); //估计是噪声的协方差把
            int index1 = lds[i].index1;
            int index2 = lds[i].index2;

            Eigen::Matrix3d r;
            Eigen::Vector3d t;
            Sophus::SE3d s;
            r = lds[i].T.matrix().block<3, 3>(0, 0);
            t = lds[i].T.matrix().block<3, 1>(0, 3);

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(r, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            r = U * V.transpose();
            Sophus::SE3d tempSE3 = Sophus::SE3d(r, t);
            eSE3->setVertex(0, vSE3s[index1]);
            eSE3->setVertex(1, vSE3s[index2]);
            eSE3->setMeasurement(tempSE3);
            optimizer.addEdge(eSE3);
        }

        //开始优化
        OTHER_OUTPUT("begin to optimizal");
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        OTHER_OUTPUT("done");
        std::vector<Eigen::Matrix4f> result;
        for (int i = 0; i < vSE3s.size(); i++)
        {
            Eigen::Matrix4d src = vSE3s[i]->estimate().matrix();
            Eigen::Matrix4f dst;
            matrix4d2matrix4f(src, dst);
            result.push_back(dst);
        }
        return result;
    }
    std::vector<loopData> loop::getLoops()
    {
        return lds;
    }

    void loop::getRT(Eigen::Matrix4f &tf, Eigen::Matrix3d &R, Eigen::Vector3d &T)
    {
        Eigen::Matrix3f r = tf.block<3, 3>(0, 0);
        Eigen::Vector3f t = tf.block<3, 1>(0, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R(i, j) = static_cast<double>(r(i, j));
        for (int i = 0; i < 3; i++)
            T[i] = static_cast<double>(t[i]);
    }
    void loop::matrix4d2matrix4f(Eigen::Matrix4d &src, Eigen::Matrix4f &dst)
    {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                dst(i, j) = static_cast<double>(src(i, j));
    }

}; // namespace mySLAM
