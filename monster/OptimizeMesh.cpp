#include <iostream>

#include "OptimizeMesh.h"
#include "MeshView.h"
#include "ExtractContour.h"
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

using namespace Monster;
bool OptimizeMesh::init(const string &sketchViews,
                        const string &sketchFolder,
                        const string &mapFolder,
                        const string &resultFolder,
                        const string &viewPointFile)
{
    cout << "Loading data..." << endl;

    mResultFolder = resultFolder;
    mVisualFolder = mResultFolder + "/visual/";
    if (!fs::exists(mVisualFolder))
    {
        fs::create_directories(mVisualFolder);
    }

    // load view points

    if (!MeshView::loadViewPoints(viewPointFile, mViewPoints, &mViewGroups))
        return false;

    // load sketch data

    int numMaps = (int)mViewPoints.size();
    if (!mViewMaps.loadSketch(sketchViews, sketchFolder))
        return false;
    if (!preProcessSketch())
        return false;
    return true;
}

bool OptimizeMesh::process()
{
    return false;
}

bool OptimizeMesh::preProcessSketch()
{
    extractSketchContour();
    return true;
}

void OptimizeMesh::extractSketchContour()
{
    cout << "Extracting contour..." << endl;

    int numSketches = (int)mViewMaps.mSketches.size();
    mSketchContours.resize(numSketches);

    for (int sketchID = 0; sketchID < numSketches; sketchID++)
    {
        MatD &sketch = mViewMaps.mSketches[sketchID];
        vector<vec2d> &contour = mSketchContours[sketchID];
        string visualFolder = mVisualFolder + "contour-" + to_string(sketchID) + "/";
        if (!fs::exists(visualFolder))
        {
            fs::create_directories(visualFolder);
        }

        ExtractContour::extract(sketch, contour, visualFolder);
    }
}