#pragma once

#include <vector>
#include <string>

#include "CMLHelper.h"

using namespace std;

namespace Monster
{

    class ExtractContour
    {

    private:
        ExtractContour() {}
        ~ExtractContour() {}

    public:
        static bool extract(vector<vector<double>> &sketch, vector<vec2d> &contour, string visualFolder = "");

    private:
        static bool extractOutline(vector<vector<bool>> &sketch, vector<vec2i> &outline);
        static bool dilateMask(vector<vector<bool>> &mask);
        static bool erodeMask(vector<vector<bool>> &mask);

    private:
        static string mVisualFolder;
    };
} // namespace Monster