#pragma once
#include <string>
#include <vector>
#include <unordered_set>

#include <eigen3/Eigen/Eigen>

#include "MapsData.h"
using namespace std;

namespace Monster
{
    class OptimizeMesh
    {
    public:
        bool init(const string &sketchViews,
                  const string &sketchFolder,
                  const string &mapFolder,
                  const string &resultFolder,
                  const string &viewPointFile);
        bool process();

    private:
        bool preProcessSketch();
        void extractSketchContour();

    private:
        string mResultFolder;
        string mVisualFolder;

        TTriangleMesh mMesh;
        MapsData mViewMaps;

        vector<Eigen::Vector3d> mViewPoints;
        vector<vector<int>> mViewGroups;
        vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mViewMatrices;   // object space -> sketch space
        vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mRotateMatrices; // object space -> sketch space

        vector<vector<vec2d>> mSketchContours;        // pixel coordinate : # of contour pixels : # of views
        vector<vector<vector<vec2d>>> mSketchStrokes; // sample coordinate : # of stroke samples : # of strokes : # of views

        vector<vec3d> mContourHandles; // deformation handle position : # of vertices
        vector<bool> mContourFlags;    // whether lying on contour : # of vertices

        vector<bool> mStrokeFlags; // whether lying on stroke : # of vertices

        vector<unordered_set<int>> mMeshGraph;
    };
} // namespace Monster