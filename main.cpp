#include <iostream>
#include <boost/filesystem.hpp>
#include <vector>

#include "monster/OptimizeViews.h"
#include "monster/ProjectViews.h"
#include "monster/PlyLoader.h"
#include "monster/OptimizeMesh.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace Monster;

int main()
{
    vector<vec3> vertices, normals;
    vector<vec3i> indices;
    PlyLoader::loadMesh("data/3.ply", &vertices, &normals, &indices);

    ProjectViews pv;
    pv.init(512);
    pv.loadMesh(&vertices, &normals, &indices);
    pv.bindBuffers();
    // pv.render();
    pv.cleanContext();

    // int stage = 1;
    // string sketchViews = "FS";
    // string sketchFolder = "data/CharacterDraw/sketch/m1/";
    // string mapFolder = "data/CharacterDraw/pred/images/m1/";
    // string resultFolder = "data/CharacterDraw/result/m1/";
    // string viewPointFile = "data/CharacterDraw/view/view.off";
    // bool bSkipOptimization = false;
    // bool bSymmetrization = false;

    // if (fs::exists(resultFolder))
    // {
    //     fs::create_directories(resultFolder);
    // }

    // if (stage == 1)
    // {
    //     OptimizeViews ov;
    //     ov.init(sketchViews, sketchFolder, mapFolder, resultFolder, viewPointFile);
    //     ov.process(bSkipOptimization, bSymmetrization);
    // }
    // else if (stage == 2)
    // {
    //     OptimizeMesh om;
    //     om.init(sketchViews, sketchFolder, mapFolder, resultFolder, viewPointFile);
    //     // if (!om.process())
    //     //     return error("");
    // }
    return 0;
}