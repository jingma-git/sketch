#pragma once

#include <string>
#include "Types.h"
using namespace std;

namespace Monster
{
    class MeshIO
    {
    private:
        MeshIO() {}
        ~MeshIO() {}

    public:
        static bool savePointSet(string fileName, TPointSet &points, bool ascii = true);
    };
} // namespace Monster