#pragma once
#include <eigen3/Eigen/Eigen>

using namespace std;

namespace Monster
{

    class MeshView
    {

    private:
        // make it non-instantiable
        MeshView() {}
        ~MeshView() {}

    public:
        static double VIEW_RADIUS;

    public:
        static bool loadViewPoints(string fileName, vector<Eigen::Vector3d> &viewPoints, vector<vector<int>> *ptrViewGroups = 0);
        static bool buildViewMatrix(Eigen::Vector3d &inViewPoint, Eigen::Matrix4d &outViewMat);
        static bool buildProjMatrix(Eigen::Matrix4d &outMat,
                                    double l = -VIEW_RADIUS, double r = VIEW_RADIUS,
                                    double b = -VIEW_RADIUS, double t = VIEW_RADIUS,
                                    double n = 0.1, double f = VIEW_RADIUS * 2.0);
        static bool buildImageMatrix(Eigen::Matrix4d &outMat, int imageWidth, int imageHeight, double shift = -0.5);

    private:
        // static bool error(string s);
    };
} // namespace Monster