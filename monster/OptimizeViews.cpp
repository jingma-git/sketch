#include "OptimizeViews.h"
#include "MeshView.h"
#include "ProjectViews.h"
#include "MatchRigidICP.h"
#include "OptimizeDepths.h"
#include "TheaKDTreeHelper.h"
#include "MeshKDTree.h"
#include "MeshIO.h"
#include "MeshCompute.h"

#include <unordered_set>
#include <iostream>
#include <float.h>

#include <eigen3/Eigen/Eigen>
#include <boost/filesystem.hpp>

using namespace Monster;
using namespace std;
namespace fs = boost::filesystem;

bool OptimizeViews::init(const string &sketchViews,
                         const string &sketchFolder,
                         const string &mapFolder,
                         const string &resultFolder,
                         const string &viewPointFile)
{
    mResultFolder = resultFolder;
    if (!fs::exists(mResultFolder))
    {
        fs::create_directories(mResultFolder);
    }

    if (!MeshView::loadViewPoints(viewPointFile, mViewPoints, &mViewGroups))
    {
        return false;
    }
    if (!establishAlignmentOrder())
        return false;

    int numMaps = 14;
    mViewMaps.loadSketch(sketchViews, sketchFolder);
    mViewMaps.loadMap(mapFolder, numMaps, "pred");
    mViewMaps.loadMask(mapFolder, numMaps);
    int mapSize = mViewMaps.mMapSize;

    Eigen::Matrix4d projMat, imgMat;
    if (!MeshView::buildProjMatrix(projMat))
        return false;
    if (!MeshView::buildImageMatrix(imgMat, mapSize, mapSize))
        return false;

    int numViews = (int)mViewPoints.size();
    mViewMatrices.resize(numViews);
    mRotateMatrices.resize(numViews);
    mCameraMatrices.resize(numViews);

    for (int viewID = 0; viewID < numViews; viewID++)
    {
        Eigen::Matrix4d viewMat;
        if (!MeshView::buildViewMatrix(mViewPoints[viewID], viewMat))
            return false;
        mCameraMatrices[viewID] = viewMat;
        mViewMatrices[viewID] = imgMat * projMat * viewMat;
        mRotateMatrices[viewID].setIdentity();
        mRotateMatrices[viewID].topLeftCorner(3, 3) = viewMat.topLeftCorner(3, 3);
    }

    return true;
}

bool OptimizeViews::process(bool skipOptimization, bool symmetrization)
{
    if (!skipOptimization)
    {
        int numIterations = 2;
        for (int iterID = 0; iterID < numIterations; iterID++)
        {
            cout << "----------------optimization " << iterID << " ---------------" << endl;

            string bkResultFolder = mResultFolder + "iter-" + to_string(iterID) + "/";
            if (!fs::exists(bkResultFolder))
                fs::create_directories(bkResultFolder);

            if (iterID > 0)
            {
                optimizePointCloud(bkResultFolder);
            }

            prunePointCloud();
            extractPointCloud(bkResultFolder);
            alignPointCloud(bkResultFolder);
        }
    }
    else
    {
        extractPointCloud();
    }

    wrapPointCloud();
    auto &baseP = mViewPointCloudsPosition[mViewOrder[0]];
    auto &baseN = mViewPointCloudsNormal[mViewOrder[0]];
    MapsData::visualizePoints(mResultFolder + "wrapped.ply", baseP, baseN);

    unionPointCloud();
    MeshIO::savePointSet(mResultFolder + "union.ply", mPointCloud);

    if (symmetrization)
    {
        symmetrizePointCloud();
    }

    if (!skipOptimization)
    {
        trimPointCloud();
    }
}

bool OptimizeViews::trimPointCloud()
{

    cout << "Trimming point clouds..." << endl;

    int numPoints = mPointCloud.amount;
    vector<bool> pointFlags(numPoints, true);

    // check sparse points

    SKDTree tree;
    SKDTreeData treeData;
    if (!MeshKDTree::buildKdTree(mPointCloud.positions, tree, treeData))
        return false;

    vec3 bsCenter;
    float bsRadius;
    if (!MeshCompute::computeBoundingSphere(mPointCloud, bsCenter, bsRadius))
        return false;
    double distBound = bsRadius * 0.05;
    int inspectNeighbors = 10;

    vector<double> pointDists(numPoints, 0.0);
#pragma omp parallel for
    for (int pointID = 0; pointID < numPoints; pointID++)
    {
        vec3 point = mPointCloud.positions[pointID];
        SKDT::NamedPoint queryPoint(point[0], point[1], point[2]);
        Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(inspectNeighbors);
        tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult, distBound);
        if (queryResult.size() < inspectNeighbors)
        {
            pointDists[pointID] = DBL_MAX;
        }
        else
        {
            int nbID = (int)tree.getElements()[queryResult[queryResult.size() - 1].getIndex()].id;
            double nbDist = (mPointCloud.positions[nbID] - point).length();
            //double nbDist = 0;
            //for (int queryID = 0; queryID < queryResult.size(); queryID++) {
            //	int nbID = (int)tree.getElements()[queryResult[queryID].getIndex()].id;
            //	nbDist += (mPointCloud.positions[nbID] - point).length();
            //}
            //if (nbDist) nbDist /= queryResult.size();
            //else nbDist = DBL_MAX;
            pointDists[pointID] = nbDist;
        }
    }

    vector<double> tmpDists = pointDists;
    int midPoint = numPoints / 2; // UNDONE: trimming parameters
    nth_element(tmpDists.begin(), tmpDists.begin() + midPoint, tmpDists.end());
    double midDist = tmpDists[midPoint];
    double distThreshold = midDist * 2.0;

#pragma omp parallel for
    for (int pointID = 0; pointID < numPoints; pointID++)
    {
        if (pointDists[pointID] > distThreshold)
        {
            pointFlags[pointID] = false;
        }
    }

    // update points

    vector<vec3> newPositions(0), newNormals(0);
    for (int pointID = 0; pointID < numPoints; pointID++)
    {
        if (pointFlags[pointID])
        {
            newPositions.push_back(mPointCloud.positions[pointID]);
            newNormals.push_back(mPointCloud.normals[pointID]);
        }
    }
    mPointCloud.positions.swap(newPositions);
    mPointCloud.normals.swap(newNormals);
    mPointCloud.amount = (int)mPointCloud.positions.size();
    numPoints = mPointCloud.amount;
    if (!MeshIO::savePointSet(mResultFolder + "trim.ply", mPointCloud))
        return false;

    return true;
}

bool OptimizeViews::symmetrizePointCloud()
{

    cout << "Symmetrizing point clouds..." << endl;

    int numPoints = mPointCloud.amount;

    // flip point cloud horizontally

    TPointSet flippedPointCloud = mPointCloud;
    vec3 center(0.0f, 0.0f, 0.0f);
    for (vec3 pos : mPointCloud.positions)
    {
        center += pos;
    }
    center *= 1.0f / numPoints;
    for (vec3 &position : flippedPointCloud.positions)
    {
        position[0] = center[0] * 2.0f - position[0];
    }
    for (vec3 &normal : flippedPointCloud.normals)
    {
        normal[0] = -normal[0];
    }

    // align point cloud

    Eigen::Matrix3Xd matSP(3, numPoints);
    Eigen::Matrix3Xd matSN(3, numPoints);
    Eigen::Matrix3Xd matTP(3, numPoints);
    Eigen::Matrix3Xd matTN(3, numPoints);

    for (int sampleID = 0; sampleID < numPoints; sampleID++)
    {
        vec3 p = flippedPointCloud.positions[sampleID];
        vec3 n = flippedPointCloud.normals[sampleID];
        matSP.col(sampleID) = Eigen::Vector3d(p[0], p[1], p[2]);
        matSN.col(sampleID) = Eigen::Vector3d(n[0], n[1], n[2]);
    }
    for (int sampleID = 0; sampleID < numPoints; sampleID++)
    {
        vec3 p = mPointCloud.positions[sampleID];
        vec3 n = mPointCloud.normals[sampleID];
        matTP.col(sampleID) = Eigen::Vector3d(p[0], p[1], p[2]);
        matTN.col(sampleID) = Eigen::Vector3d(n[0], n[1], n[2]);
    }

    Eigen::Affine3d xform;
    xform.setIdentity();
    if (!MatchRigidICP::run(10, matSP, matSN, matTP, matTN, xform))
        return false;
    //if (!MatchRigidICP::visualize("symmetrize.ply", matSP, matTP, xform)) return false;

    // append symmetrized points

    Eigen::Matrix3d rotation = xform.rotation();
    for (int sampleID = 0; sampleID < numPoints; sampleID++)
    {
        vec3 p = flippedPointCloud.positions[sampleID];
        vec3 n = flippedPointCloud.normals[sampleID];
        Eigen::Vector3d pv(p[0], p[1], p[2]);
        Eigen::Vector3d nv(n[0], n[1], n[2]);
        pv = xform * pv;
        nv = rotation * nv;
        mPointCloud.positions.push_back(vec3d(pv[0], pv[1], pv[2]));
        mPointCloud.normals.push_back(vec3d(nv[0], nv[1], nv[2]));
    }
    mPointCloud.amount = (int)mPointCloud.positions.size();

    return true;
}

bool OptimizeViews::unionPointCloud()
{
    cout << "Combining point clouds..." << endl;

    int numViews = (int)mViewPoints.size();

    mPointCloud.positions.clear();
    mPointCloud.normals.clear();
    for (int viewID = 0; viewID < numViews; viewID++)
    {
        int numPoints = (int)mViewPointCloudsPosition[viewID].cols();
        for (int pointID = 0; pointID < numPoints; pointID++)
        {
            Eigen::Vector3d p = mViewPointCloudsPosition[viewID].col(pointID);
            Eigen::Vector3d n = mViewPointCloudsNormal[viewID].col(pointID);
            mPointCloud.positions.push_back(vec3d(p[0], p[1], p[2]));
            mPointCloud.normals.push_back(vec3d(n[0], n[1], n[2]));
        }
    }
    mPointCloud.amount = (int)mPointCloud.positions.size();

    return true;
}

bool OptimizeViews::wrapPointCloud()
{
    int numWrapViews = (int)mViewMaps.mSketches.size();
    int numProcessViews = (int)mViewPoints.size();
    int mapSize = mViewMaps.mMapSize;

    cout << "Wrapping";
    for (int wrapViewID = 0; wrapViewID < numWrapViews; wrapViewID++)
    {
        cout << " " << wrapViewID;
        MatI &wrapMask = mViewMaps.mMasks[wrapViewID];

        //build KD tree for mask point
        vector<vec3> maskPoints(0);
        maskPoints.reserve(mapSize * mapSize);
        vector<vec2i> maskIdx(0);
        maskIdx.reserve(mapSize * mapSize);
        for (int row = 0; row < mapSize; row++)
        {
            for (int col = 0; col < mapSize; col++)
            {
                if (wrapMask[row][col])
                {
                    maskPoints.push_back(vec3((float)col, (float)row, 0.0f));
                    maskIdx.push_back(vec2i(row, col));
                }
            }
        }

        SKDTree tree;
        SKDTreeData treeData;
        MeshKDTree::buildKdTree(maskPoints, tree, treeData);

        // compute offset
        vector<vector<vec2i>> wrapOffset(mapSize, vector<vec2i>(mapSize, vec2i(0, 0)));
#pragma omp parallel for
        for (int pixelID = 0; pixelID < mapSize * mapSize; pixelID++)
        {
            int row = pixelID / mapSize;
            int col = pixelID % mapSize;
            if (wrapMask[row][col])
            {
                wrapOffset[row][col] = vec2i(0, 0);
            }
            else
            {
                SKDT::NamedPoint queryPoint((float)col, (float)row, 0.0f);
                Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(1);
                tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult);
                int maskID = (int)tree.getElements()[queryResult[0].getIndex()].id;
                wrapOffset[row][col] = maskIdx[maskID] - vec2i(row, col);
            }
        }

        // wrap points for each view

        for (int processViewID = 0; processViewID < numProcessViews; processViewID++)
        {
            // project point clouds
            Eigen::Matrix3Xd &points = mViewPointCloudsPosition[processViewID];
            int numPoints = points.cols();
            Eigen::Matrix4Xd homoPoints(4, numPoints);
            homoPoints << points, Eigen::MatrixXd::Ones(1, numPoints);

            Eigen::Matrix4Xd projPoints = mViewMatrices[processViewID] * homoPoints;
#pragma omp parallel for
            for (int i = 0; i < numPoints; i++)
            {

                int row = projPoints(1, i);
                int col = projPoints(0, i);

                if (!wrapMask[row][col])
                {
                    vec2i offset = wrapOffset[row][col];
                    projPoints(0, i) += offset[1];
                    projPoints(1, i) += offset[0];
                }
            }

            homoPoints = mViewMatrices[processViewID].inverse() * projPoints;
            points.swap(homoPoints.topRows(3));
        }
    }

    cout << endl;
}

bool OptimizeViews::optimizePointCloud(string saveFolder)
{
    int numViews = (int)mViewPoints.size();
    int mapSize = mViewMaps.mMapSize;

    vector<MatI> optimizeMasks(numViews);
    vector<MatD> optimizedDepths(numViews);
    vector<Mat3D> optimizedNormals(numViews);
    ProjectViews pv;
    if (!pv.init(mapSize))
        return false;

    Eigen::Matrix4d imgMatrix;
    MeshView::buildImageMatrix(imgMatrix, 256, 256);
    Eigen::Matrix4d projMatrix;
    MeshView::buildProjMatrix(projMatrix);

    cout
        << "Optimizing ";
    for (int viewID = 0; viewID < numViews; viewID++)
    {
        cout << viewID << " ";
        int numNeighbors = (int)(mViewNeighbors[viewID].size());
        vector<MatI> viewMasks(numNeighbors + 1);
        vector<MatD> viewDepths(numNeighbors + 1);
        vector<Mat3D> viewNormals(numNeighbors + 1);
        viewMasks[0] = mViewMaps.mMasks[viewID];
        viewDepths[0] = mViewMaps.mDepths[viewID];
        viewNormals[0] = mViewMaps.mNormals[viewID];

        if (false && saveFolder != "")
        {
            MapsData::visualizeMask(saveFolder + to_string(viewID) + "-mask-orig.png", viewMasks[0]);
            MapsData::visualizeDepth(saveFolder + to_string(viewID) + "-depth-orig.png", viewDepths[0]);
            MapsData::visualizeNormal(saveFolder + to_string(viewID) + "-normal-orig.png", viewNormals[0]);
        }

        for (int neighborID = 0; neighborID < numNeighbors; neighborID++)
        {
            int nbViewID = mViewNeighbors[viewID][neighborID];

            Eigen::Matrix4d viewMat = mViewMatrices[viewID] * mViewMatrices[nbViewID].inverse();
            Eigen::Matrix4d rotMat = mRotateMatrices[viewID] * mRotateMatrices[nbViewID].transpose();

            pv.loadMaps(mViewMaps.mMasks[nbViewID], mViewMaps.mDepths[nbViewID], mViewMaps.mNormals[nbViewID]);
            pv.project(viewMat, rotMat, viewMasks[neighborID + 1], viewDepths[neighborID + 1], viewNormals[neighborID + 1]);

            if (false)
            {
                MapsData::visualizeMask(saveFolder + to_string(viewID) + "-mask-bproj-" + to_string(neighborID) + ".png", mViewMaps.mMasks[nbViewID]);
                MapsData::visualizeDepth(saveFolder + to_string(viewID) + "-depth-bproj-" + to_string(neighborID) + ".png", mViewMaps.mDepths[nbViewID]);
                MapsData::visualizeNormal(saveFolder + to_string(viewID) + "-normal-bproj-" + to_string(neighborID) + ".png", mViewMaps.mNormals[nbViewID]);

                MapsData::visualizeMask(saveFolder + to_string(viewID) + "-mask-proj-" + to_string(neighborID) + ".png", viewMasks[neighborID + 1]);
                MapsData::visualizeDepth(saveFolder + to_string(viewID) + "-depth-proj-" + to_string(neighborID) + ".png", viewDepths[neighborID + 1]);
                MapsData::visualizeNormal(saveFolder + to_string(viewID) + "-normal-proj-" + to_string(neighborID) + ".png", viewNormals[neighborID + 1]);
            }
        }

        OptimizeDepths::optimize(viewMasks, viewDepths, viewNormals, mViewMaps.mMaskProbs[viewID],
                                 optimizeMasks[viewID], optimizedDepths[viewID], optimizedNormals[viewID]);
    }

    pv.cleanContext();
    cout << endl;
    // output optimized results
    for (int viewID = 0; viewID < numViews; viewID++)
    {
        mViewMaps.mMasks[viewID].swap(optimizeMasks[viewID]);
        mViewMaps.mDepths[viewID].swap(optimizedDepths[viewID]);
        mViewMaps.mNormals[viewID].swap(optimizedNormals[viewID]);
    }

    return true;
}

bool OptimizeViews::prunePointCloud()
{
    int numViews = (int)mViewPoints.size();
    int numSketchViews = (int)mViewMaps.mSketches.size();
    int mapSize = mViewMaps.mMapSize;

    cout << "Pruning ";
    int kernel = 3; // mask dilation kernel size
    for (int pruneViewID = numSketchViews; pruneViewID < numViews; pruneViewID++)
    {
        cout << " " << pruneViewID;
        Eigen::Matrix4d xform = mViewMatrices[pruneViewID].inverse();
        for (int h = 0; h < mapSize; h++)
        {
            for (int w = 0; w < mapSize; w++)
            {
                if (!mViewMaps.mMasks[pruneViewID][h][w])
                    continue;
                double d = mViewMaps.mDepths[pruneViewID][h][w];
                Eigen::Vector4d p = xform * Eigen::Vector4d((double)w, (double)h, d, 1.0);
                int pruneVote = 0;
                int keepVote = 0;
                for (int maskViewID : mViewNeighbors[pruneViewID])
                {
                    Eigen::Vector4d q = mViewMatrices[maskViewID] * p; // project the point into this views
                    int u = (int)q[0];                                 // x
                    int v = (int)q[1];                                 // y
                    bool valid = false;
                    for (int uu = max(0, u - kernel); uu <= min(mapSize - 1, u + kernel); uu++)
                    {
                        for (int vv = max(0, v - kernel); vv <= min(mapSize - 1, v + kernel); vv++)
                        {
                            if (mViewMaps.mMasks[maskViewID][vv][uu])
                            {
                                valid = true;
                                break;
                            }
                        }
                        if (valid)
                            break;
                    }

                    int vote = 1;
                    if (maskViewID < numSketchViews)
                        vote = 3; //give more votes for sketch views
                    if (valid)
                        keepVote += vote;
                    else
                    {
                        pruneVote += vote;
                    }
                }
                if (pruneVote > keepVote)
                {
                    mViewMaps.mMasks[pruneViewID][h][w] = false;
                    mViewMaps.mDepths[pruneViewID][h][w] = 1.0;
                    mViewMaps.mNormals[pruneViewID][h][w] = vec3d(1.0, 1.0, 1.0);
                }
            }
        }
    }
    cout << endl;
}

bool OptimizeViews::extractPointCloud(string saveFolder)
{
    int numViews = (int)mViewPoints.size();
    int mapSize = mViewMaps.mMapSize;

    mViewPointCloudsPosition.resize(numViews);
    mViewPointCloudsNormal.resize(numViews);

    cout << "Extracting";
    for (int viewID = 0; viewID < numViews; viewID++)
    {
        cout << " " << viewID;

        int numPoints = 0;
        for (int h = 0; h < mapSize; h++)
        {
            for (int w = 0; w < mapSize; w++)
            {
                if (mViewMaps.mMasks[viewID][h][w])
                    numPoints++;
            }
        }

        Eigen::Matrix3Xd &matP = mViewPointCloudsPosition[viewID];
        Eigen::Matrix3Xd &matN = mViewPointCloudsNormal[viewID];
        matP.resize(3, numPoints);
        matN.resize(3, numPoints);

        Eigen::Matrix4d xformMat = mViewMatrices[viewID].inverse();
        Eigen::Matrix4d rotateMat = mRotateMatrices[viewID].transpose();

        int pointID = 0;
        for (int h = 0; h < mapSize; h++)
        {
            for (int w = 0; w < mapSize; w++)
            {
                if (!mViewMaps.mMasks[viewID][h][w])
                    continue;

                double vd = mViewMaps.mDepths[viewID][h][w];
                vec3d &vn = mViewMaps.mNormals[viewID][h][w];
                Eigen::Vector4d p = xformMat * Eigen::Vector4d(double(w), double(h), vd, 1.0);
                Eigen::Vector4d n = rotateMat * Eigen::Vector4d(vn[0], vn[1], vn[2], 1.0);

                matP.col(pointID) = p.topRows(3);
                matN.col(pointID) = n.topRows(3);
                pointID++;
            }
        }

        matP = matP.cwiseMax(-MeshView::VIEW_RADIUS).cwiseMin(MeshView::VIEW_RADIUS);

        if (saveFolder != "")
        {
            string shapeName = saveFolder + to_string(viewID) + ".ply";
            MapsData::visualizePoints(shapeName, matP, matN);
        }
    }
    cout << endl;
}

bool OptimizeViews::establishAlignmentOrder()
{

    int numViews = (int)mViewPoints.size();

    vector<unordered_set<int>> neighbors(numViews, unordered_set<int>());
    for (vector<int> &group : mViewGroups)
    {
        for (int k = 0; k < (int)group.size(); k++)
        {
            int v1 = group[k];
            int v2 = group[(k + 1) % (int)group.size()];
            neighbors[v1].insert(v2);
            neighbors[v2].insert(v1);
        }
    }

    mViewNeighbors.resize(numViews);
    for (int viewID = 0; viewID < numViews; viewID++)
    {
        mViewNeighbors[viewID].assign(neighbors[viewID].begin(), neighbors[viewID].end());
    }

    vector<bool> visited(numViews, false);
    visited[0] = true;
    mViewOrder.assign(1, 0);
    int head = 0;
    while (head < (int)mViewOrder.size())
    {
        for (int nb : neighbors[mViewOrder[head]])
        {
            if (!visited[nb])
            {
                mViewOrder.push_back(nb);
                visited[nb] = true;
            }
        }
        head++;
    }

    cout << "Fusing order: ";
    for (int view : mViewOrder)
        cout << view << " ";
    cout << endl;

    return true;
}

bool OptimizeViews::alignPointCloud(string saveFolder)
{
    Eigen::Matrix3Xd &baseMatP = mViewPointCloudsPosition[mViewOrder[0]];
    Eigen::Matrix3Xd &baseMatN = mViewPointCloudsNormal[mViewOrder[0]];

    cout << "Aligning";
    for (int orderID = 1; orderID < (int)mViewOrder.size(); orderID++)
    {
        int viewID = mViewOrder[orderID];
        cout << " " << viewID;

        Eigen::Affine3d xform;
        xform.setIdentity();

        Eigen::Matrix3Xd &viewMatP = mViewPointCloudsPosition[viewID];
        Eigen::Matrix3Xd &viewMatN = mViewPointCloudsNormal[viewID];

        MatchRigidICP::run(3, viewMatP, viewMatN, baseMatP, baseMatN, xform, true);
        // transform points
        viewMatP = xform * viewMatP;
        viewMatN = xform.rotation() * viewMatN;
        mViewPointCloudsPosition[viewID] = viewMatP;
        mViewPointCloudsNormal[viewID] = viewMatN;

        // add in points
        Eigen::Matrix3Xd extMatP(3, baseMatP.cols() + viewMatP.cols());
        Eigen::Matrix3Xd extMatN(3, baseMatN.cols() + viewMatN.cols());
        extMatP << baseMatP, viewMatP;
        extMatN << baseMatN, viewMatN;
        baseMatP.swap(extMatP);
        baseMatN.swap(extMatN);

        // add in transformation
        Eigen::Matrix4d alignMat = xform.inverse().matrix();
        Eigen::Matrix4d rotateMat = Eigen::Matrix4d::Identity();
        rotateMat.topLeftCorner(3, 3) = alignMat.topLeftCorner(3, 3);
        mViewMatrices[viewID] *= alignMat;
        mRotateMatrices[viewID] *= rotateMat;
    }
    cout << endl;

    if (saveFolder != "")
        if (!MapsData::visualizePoints(saveFolder + "align.ply", baseMatP, baseMatN))
            return false;

    return true;
}