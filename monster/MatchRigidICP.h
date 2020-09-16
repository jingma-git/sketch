#pragma once

//Iterative Closet Point

#include <vector>
#include <string>
#include <eigen3/Eigen/Eigen>
#include "TheaKDTreeHelper.h"

using namespace std;

namespace Monster
{
    class MatchRigidICP
    {
    public:
        // sourceP: 3xN source point positions
        // sourceN: 3xN source point normals
        // targetP: 3xN target point positions
        // targetN: 3xN target point normals
        // transformation: initial and output transformation
        // alignment: whether `source' and `target' is roughly aligned through `transformation'
        static bool run(
            int iteration,
            const Eigen::Matrix3Xd &sourceP,
            const Eigen::Matrix3Xd &sourceN,
            const Eigen::Matrix3Xd &targetP,
            const Eigen::Matrix3Xd &targetN,
            Eigen::Affine3d &transformation,
            bool aligned = false);

    private:
        static bool buildKDTree(const Eigen::Matrix3Xd &points,
                                SKDTree &tree, SKDTreeData &data);
        static bool initAlignment(const Eigen::Matrix3Xd &source,
                                  const Eigen::Matrix3Xd target,
                                  Eigen::Affine3d &transformation);
        static bool findNearestNeighbors(const SKDTree &tree,
                                         const Eigen::Matrix3Xd &inPoints,
                                         vector<int> &outIndices);
        static bool sliceMatrices(const Eigen::Matrix3Xd &inMatrix,
                                  const vector<int> &inIndices,
                                  Eigen::Matrix3Xd &outMatrix);

        // if the the distance from source to target is less than certain threshold
        // keep the indices of source in outIndices
        static bool findMatchedNeighbors(const Eigen::Matrix3Xd &inSourceP,
                                         const Eigen::Matrix3Xd &inSourceN,
                                         const Eigen::Matrix3Xd &inTargetP,
                                         const Eigen::Matrix3Xd &inTargetN,
                                         vector<int> &outIndices,
                                         bool aligned);
        static bool extractTransformation(const Eigen::Matrix3Xd &source,
                                          const Eigen::Matrix3Xd &target,
                                          Eigen::Affine3d &transformation);
        static double error(const Eigen::Matrix3Xd &source, const Eigen::Matrix3Xd &target);
    };
} // namespace Monster