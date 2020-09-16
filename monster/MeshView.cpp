#include <fstream>
#include <iostream>

#include "MeshView.h"

using namespace std;
using namespace Monster;

double MeshView::VIEW_RADIUS = 2.5;
bool MeshView::buildProjMatrix(Eigen::Matrix4d &outMat,
                               double l, double r,
                               double b, double t,
                               double n, double f)
{
    // ref: https://www.opengl.org/sdk/docs/man2/xhtml/glOrtho.xml

    outMat << 2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l),
        0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b),
        0.0, 0.0, -2.0 / (f - n), -(f + n) / (f - n),
        0.0, 0.0, 0.0, 1.0;

    return true;
}

bool MeshView::buildImageMatrix(Eigen::Matrix4d &outMat, int imageWidth, int imageHeight, double shift)
{

    // x: [-1, 1) => [0, W)
    // y: [1, -1) => [0, H)
    // z: keep as-is

    outMat << imageWidth / 2.0, 0.0, 0.0, imageWidth / 2.0 + shift,
        0.0, -imageHeight / 2.0, 0.0, imageHeight / 2.0 + shift,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0;

    return true;
}

bool MeshView::loadViewPoints(string fileName, vector<Eigen::Vector3d> &viewPoints, vector<vector<int>> *ptrViewGroups)
{
    // view points from mesh file
    cout << fileName << endl;
    ifstream file(fileName);
    string header;
    getline(file, header);
    if (file.fail())
    {
        cerr << "invalid mesh file" << endl;
        return false;
    }

    int numPoints, numGroups, numDontcare;
    file >> numPoints >> numGroups >> numDontcare;
    viewPoints.resize(numPoints);
    for (int pointID = 0; pointID < numPoints; pointID++)
    {
        Eigen::Vector3d &point = viewPoints[pointID];
        file >> point[0] >> point[1] >> point[2];
    }
    if (ptrViewGroups)
    {
        ptrViewGroups->resize(numGroups);
        for (int groupID = 0; groupID < numGroups; groupID++)
        {
            vector<int> &group = (*ptrViewGroups)[groupID];
            int groupSize;
            file >> groupSize;
            group.resize(groupSize);
            for (int k = 0; k < groupSize; k++)
                file >> group[k];
        }
    }
    file.close();

    return true;
}

bool MeshView::buildViewMatrix(Eigen::Vector3d &inViewPoint, Eigen::Matrix4d &outViewMat)
{

    // ref: http ://www.ibm.com/support/knowledgecenter/ssw_aix_53/com.ibm.aix.opengl/doc/openglrf/gluLookAt.htm

    Eigen::Vector3d E = inViewPoint;  // eye
    Eigen::Vector3d C(0.0, 0.0, 0.0); // center
    Eigen::Vector3d U(0.0, 1.0, 0.0); // up
    Eigen::Vector3d L = (C - E).normalized();
    Eigen::Vector3d S = L.cross(U);
    if (S.norm() == 0)
    {
        U = Eigen::Vector3d(0.0, 0.0, -1.0);
        S = L.cross(U);
    }
    S = S.normalized();
    Eigen::Vector3d Up = S.cross(L);
    Eigen::Matrix4d R;
    R.setIdentity();
    for (int k = 0; k < 3; k++)
        R(0, k) = S[k];
    for (int k = 0; k < 3; k++)
        R(1, k) = Up[k];
    for (int k = 0; k < 3; k++)
        R(2, k) = -L[k];
    Eigen::Matrix4d T;
    T.setIdentity();
    for (int k = 0; k < 3; k++)
        T(k, 3) = -E[k];
    outViewMat = R * T;

    return true;
}