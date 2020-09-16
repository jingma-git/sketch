#include "MatchRigidICP.h"
#include "TheaKDTreeHelper.h"
#include <float.h>
using namespace Monster;

bool MatchRigidICP::initAlignment(const Eigen::Matrix3Xd &source,
                                  const Eigen::Matrix3Xd target,
                                  Eigen::Affine3d &transformation)
{
    Eigen::Vector3d centerS = (transformation * source).rowwise().mean();
    Eigen::Vector3d centT = target.rowwise().mean();
    transformation.pretranslate(centT - centerS);
    return true;
}

bool MatchRigidICP::run(
    int iteration,
    const Eigen::Matrix3Xd &sourceP,
    const Eigen::Matrix3Xd &sourceN,
    const Eigen::Matrix3Xd &targetP,
    const Eigen::Matrix3Xd &targetN,
    Eigen::Affine3d &transformation,
    bool aligned)
{
    SKDTree tree;
    SKDTreeData treeData;

    buildKDTree(targetP, tree, treeData);

    if (!aligned)
    {
        initAlignment(sourceP, targetP, transformation);
    }

    Eigen::Affine3d intialTransformation = transformation;

    //ICP iteration
    double lastError = DBL_MAX;
    for (int iterID = 0; iterID < iteration; iterID++)
    {
        Eigen::Matrix3d rotation = transformation.rotation();
        Eigen::Matrix3Xd matXSP = transformation * sourceP;
        Eigen::Matrix3Xd matXSN = rotation * sourceN;

        Eigen::Matrix3Xd matTP, matTN;
        vector<int> slices;
        findNearestNeighbors(tree, matXSP, slices);
        sliceMatrices(targetP, slices, matTP);
        sliceMatrices(targetN, slices, matTN);
        findMatchedNeighbors(matXSP, matXSN, matTP, matTN, slices, false);
        if (slices.empty())
            break;
        sliceMatrices(matXSP, slices, matXSP);
        sliceMatrices(matXSN, slices, matXSN);
        sliceMatrices(matTP, slices, matTP);
        sliceMatrices(matTN, slices, matTN);

        //Align matched points
        Eigen::Affine3d newTransformation;
        extractTransformation(matXSP, matTP, newTransformation);
        transformation = newTransformation * transformation;

        matXSP = transformation * sourceP;
        double currentError = error(targetP, matXSP);
        //cout << " iter " << iterID << ": " << currentError;
        if (currentError > lastError * 0.99)
        {
            // cout << " break";
            break;
        }
        //cout << endl;
        lastError = currentError;
    }

    if (!transformation.matrix().allFinite())
    {
        transformation = intialTransformation;
        initAlignment(sourceP, targetP, transformation);
    }

    return true;
}

double MatchRigidICP::error(const Eigen::Matrix3Xd &source, const Eigen::Matrix3Xd &target)
{
    SKDTree tree;
    SKDTreeData treeData;
    buildKDTree(target, tree, treeData);

    Eigen::Matrix3Xd matched;
    vector<int> neighbors;
    findNearestNeighbors(tree, source, neighbors);
    sliceMatrices(target, neighbors, matched);
    double outError = (matched - source).squaredNorm() / source.cols();
    return outError;
}

bool MatchRigidICP::extractTransformation(const Eigen::Matrix3Xd &source,
                                          const Eigen::Matrix3Xd &target,
                                          Eigen::Affine3d &transformation)
{
    Eigen::Vector3d vecSCenter = source.rowwise().mean();
    Eigen::Vector3d vecTCenter = target.rowwise().mean();
    Eigen::Matrix3Xd transSource = source.colwise() - vecSCenter;
    Eigen::Matrix3Xd transTarget = target.colwise() - vecTCenter;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(transTarget * transSource.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d matRotate = svd.matrixU() * svd.matrixV().transpose();
    Eigen::Vector3d vecTranslate = vecTCenter - matRotate * vecSCenter;
    transformation.setIdentity();
    transformation.prerotate(matRotate);
    transformation.pretranslate(vecTranslate);
    return true;
}

bool MatchRigidICP::findMatchedNeighbors(const Eigen::Matrix3Xd &inSourceP,
                                         const Eigen::Matrix3Xd &inSourceN,
                                         const Eigen::Matrix3Xd &inTargetP,
                                         const Eigen::Matrix3Xd &inTargetN,
                                         vector<int> &outIndices,
                                         bool aligned)
{
    if (inSourceP.cols() == 0)
        return true;
    const double rejectDistanceThreshold = 5.0;

    Eigen::ArrayXd vecD = (inSourceP - inTargetP).colwise().norm().array();
    Eigen::ArrayXd vecN = (inSourceN.transpose() * inTargetN).diagonal().array().abs();

    double maxDist;
    if (aligned)
    {
        // bounding box diagonal length
        double bbLength = (inSourceP.rowwise().maxCoeff() - inSourceP.rowwise().minCoeff()).norm();
        maxDist = bbLength * 0.05;
    }
    else
    {
        // median length
        vector<double> vDist(vecD.data(), vecD.data() + vecD.size());
        nth_element(vDist.begin(), vDist.begin() + vecD.size() / 2, vDist.end());
        maxDist = vDist[vecD.size() / 2] * rejectDistanceThreshold;
    }
    auto filter = vecD < maxDist && vecN > 0.5;
    outIndices.clear();
    outIndices.reserve((int)filter.count());
    for (int j = 0; j < filter.size(); j++)
    {
        if (filter(j))
        {
            outIndices.push_back(j);
        }
    }
}

bool MatchRigidICP::sliceMatrices(const Eigen::Matrix3Xd &inMatrix,
                                  const vector<int> &inIndices,
                                  Eigen::Matrix3Xd &outMatrix)
{
    Eigen::Matrix3Xd tmp;
    tmp.resize(inMatrix.rows(), inIndices.size());
    for (int j = 0; j < (int)inIndices.size(); j++)
    {
        tmp.col(j) = inMatrix.col(inIndices[j]);
    }
    outMatrix.swap(tmp);
    return true;
}

bool MatchRigidICP::findNearestNeighbors(const SKDTree &tree,
                                         const Eigen::Matrix3Xd &inPoints,
                                         vector<int> &outIndices)
{
    outIndices.resize(inPoints.cols());
#pragma omp parallel for
    for (int j = 0; j < inPoints.cols(); j++)
    {
        SKDT::NamedPoint queryPoint((float)inPoints(0, j), (float)inPoints(1, j), (float)inPoints(2, j));
        Thea::BoundedSortedArray<SKDTree::Neighbor> queryResult(1);
        tree.kClosestElements<Thea::MetricL2>(queryPoint, queryResult);
        if (queryResult.isEmpty())
            outIndices[j] = 0;
        else
        {
            outIndices[j] = (int)tree.getElements()[queryResult[0].getIndex()].id;
        }
    }
}

bool MatchRigidICP::buildKDTree(const Eigen::Matrix3Xd &points,
                                SKDTree &tree, SKDTreeData &data)
{
    data.resize(points.cols());
    for (int i = 0; i < (int)points.cols(); i++)
    {
        data[i] = SKDT::NamedPoint((float)points(0, i), (float)points(1, i), (float)points(2, i), (size_t)i);
    }
    tree.init(data.begin(), data.end());
    return true;
}