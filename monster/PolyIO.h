#pragma once

#include <string>
#include <vector>

#include "CMLHelper.h"

using namespace std;

class PolyIO
{
public:
    PolyIO();
    void cleanUp();

    bool addPoint(const vector<vec3> *vertices,
                  vector<vec3> *normals = 0, matrix transform = cml::identity_4x4(),
                  vec3i color = vec3i(0, 0, 0));
    bool output(string fileName, bool ascii = true);

public:
    vector<vec3> mVertices; //TODO: change vertices to float
    vector<vec3> mNormals;
    vector<vec3i> mColors;
    vector<vec3i> mFaceIndices;
    vector<vec2i> mEdgeIndices;
    vector<vec3i> mFaceColors;
    vector<string> mComments;

    bool hasNormal;
    bool hasColor;
    bool hasFaceColor;
};