#pragma once

#include <vector>
#include <eigen3/Eigen/Eigen>
#include "CMLHelper.h"

using namespace std;

namespace Monster
{
    typedef vector<vector<double>> MatD;
    typedef vector<vector<int>> MatI;
    typedef vector<vector<vec3d>> Mat3D;

    struct TPointSet
    {
        vector<vec3> positions;
        vector<vec3> normals;

        int amount;
        TPointSet() : amount(0) {}
        TPointSet(const TPointSet &ps) : positions(ps.positions), normals(ps.normals), amount(ps.amount) {}
    };

    struct TTriangleMesh : TPointSet
    {
        vector<vec3i> indices;
        TTriangleMesh() {}
        TTriangleMesh(const TTriangleMesh &tm) : TPointSet(tm), indices(tm.indices) {}
    };
} // namespace Monster