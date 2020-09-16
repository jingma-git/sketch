#pragma once

#include <vector>
#include <string>
#include "Types.h"

using namespace std;

namespace Monster
{
    class OptimizeDepths
    {
    private:
        OptimizeDepths() {}
        ~OptimizeDepths() {}

    public:
        static bool optimize(
            const vector<MatI> &masks,
            const vector<MatD> &depths,
            const vector<Mat3D> &normals,
            const MatD &maskProbs,
            MatI &outMasks,
            MatD &outDepths,
            Mat3D &outNormals);
    };
} // namespace Monster