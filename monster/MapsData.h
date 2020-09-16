#pragma once

#include <vector>
#include <string>
#include <eigen3/Eigen/Eigen>
#include "Types.h"

using namespace std;
using namespace Eigen;

namespace Monster
{

    class MapsData
    {
    public:
        MapsData() : mSketchSize(0), mMapSize(0){};

        bool loadSketch(const string &sketchViews, const string &sketchFolder);
        bool loadMap(const string &mapFolder, int numMaps, string prefix = "pred");
        bool loadMask(const string &mapFolder, int numMaps);
        bool parseSketch(const string &imgName, MatD &sketch);
        bool parseDepthNormal(const string &imgName, MatI &mask, MatD &depth, Mat3D &normal);
        bool parseMask(const string &imgName, MatD &maskProb);

        static bool visualizePoints(const string &fileName, const Eigen::Matrix3Xd &positions, const Eigen::Matrix3Xd &normals);
        static bool visualizeMask(string fileName, const MatI &mask);
        static bool visualizeDepth(string fileName, const MatD &depth);
        static bool visualizeNormal(string fileName, const Mat3D &normal);

        int mSketchSize;
        int mMapSize;

        vector<MatD> mSketches;
        vector<MatI> mMasks;
        vector<MatD> mMaskProbs;
        vector<MatD> mDepths;
        vector<Mat3D> mNormals;
    };
} // namespace Monster