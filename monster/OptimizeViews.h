#pragma once

#include <vector>
#include <string>

#include "MapsData.h"
#include "Types.h"
using namespace std;

namespace Monster
{
    class OptimizeViews
    {
    public:
        bool init(const string &sketchViews,
                  const string &sketchFolder,
                  const string &mapFolder,
                  const string &resultFolder,
                  const string &viewPointFile);
        bool process(bool skipOptimization = false, bool symmetrization = false);

    private:
        bool establishAlignmentOrder();
        bool optimizePointCloud(string saveFolder = "");
        bool prunePointCloud();
        bool extractPointCloud(string saveFolder = "");
        bool alignPointCloud(string saveFolder = "");
        bool wrapPointCloud();
        bool unionPointCloud();
        bool symmetrizePointCloud();
        bool trimPointCloud();

        vector<Eigen::Vector3d> mViewPoints;
        vector<vector<int>> mViewGroups;
        vector<vector<int>> mViewNeighbors;
        vector<int> mViewOrder;
        vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mViewMatrices;   // object space-->camera space-->image space
        vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mRotateMatrices; // object space-->camera space
        vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mCameraMatrices;

        MapsData mViewMaps;
        vector<Eigen::Matrix3Xd> mViewPointCloudsPosition;
        vector<Eigen::Matrix3Xd> mViewPointCloudsNormal;
        TPointSet mPointCloud; //Union of points from all views

        string mResultFolder;
    };
} // namespace Monster