#include "MapsData.h"
#include "../common/common.h"
#include "../common/CvUtil.h"
#include "Types.h"
#include "MeshIO.h"

#include <opencv4/opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>
#include <boost/filesystem.hpp>

using namespace Monster;
namespace fs = boost::filesystem;
using namespace Eigen;

bool MapsData::loadMap(const string &mapFolder, int numMaps, string prefix)
{
    vector<string> imgNames;
    for (int viewID = 0; viewID < numMaps; viewID++)
    {
        string suffix = prefix + "-dn14--" + to_string(viewID) + ".png";
        string name = mapFolder + suffix;
        if (!fs::exists(name))
        {
            cerr << name << " does not exist!\n";
            return false;
        }
        imgNames.push_back(name);
    }

    int numViews = (int)imgNames.size();

    mMasks.resize(numViews);
    mDepths.resize(numViews);
    mNormals.resize(numViews);

    cout << __FILE__ << " " << __LINE__ << " Loading data Depth&Normal...\n";
    for (int viewID = 0; viewID < numViews; viewID++)
    {
        if (!parseDepthNormal(imgNames[viewID], mMasks[viewID], mDepths[viewID], mNormals[viewID]))
            return false;
    }

    return true;
}

bool MapsData::parseDepthNormal(const string &imgName, MatI &mask, MatD &depth, Mat3D &normal)
{
    if (!fs::exists(imgName))
    {
        cerr << imgName << " doesn't exist!\n";
        return false;
    }

    cv::Mat img = cv::imread(imgName, cv::IMREAD_UNCHANGED);
    mask.resize(img.rows);
    depth.resize(img.rows);
    normal.resize(img.rows);

    for (int h = 0; h < img.rows; h++)
    {
        mask[h].resize(img.cols);
        depth[h].resize(img.cols);
        normal[h].resize(img.cols);

        for (int w = 0; w < img.cols; w++)
        {
            bool mk = true; //mask keep
            vec3d n(1.0, 1.0, 1.0);
            double d = 1.0;

            if (img.channels() == 1)
            {
                ushort c = img.at<short>(h, w);
            }
            else if (img.channels() == 4)
            {
                const cv::Vec4s &bgra = img.at<cv::Vec4s>(h, w);
                ushort b = bgra.val[0];
                ushort g = bgra.val[1];
                ushort r = bgra.val[2];
                ushort a = bgra.val[3];

                double x = r / (double)SMAX - 1.0;
                double y = g / (double)SMAX - 1.0;
                double z = b / (double)SMAX - 1.0;
                d = a / (double)SMAX - 1.0;
                n = vec3d(x, y, z);
                double nl = n.length();
                if (nl == 0 || nl > 1.5)
                {
                    mk = false;
                }
                else
                {
                    n /= nl;
                }
            }
            else
            {
                cout << "unsupported channels\n";
                exit(-1);
            }

            if (d >= 0.9)
            {
                mk = false;
            }

            if (!mk)
            {
                d = 1.0;
                n = vec3d(1.0, 1.0, 1.0);
            }

            mask[h][w] = mk;
            depth[h][w] = d;
            normal[h][w] = n;
        }
    }

    if (img.channels() == 1)
    {
        // TODO computeNormalFromDepth
    }
    return true;
}

bool MapsData::loadSketch(const string &sketchViews, const string &sketchFolder)
{
    if (!fs::exists(sketchFolder))
    {
        cerr << sketchFolder << " does not exist\n";
        return false;
    }

    string styleID = "0";

    int numSketches = sketchViews.length();
    vector<string> sketchNames;
    for (int viewID = 0; viewID < numSketches; viewID++)
    {
        string viewSketchName = sketchFolder + "sketch-" + sketchViews[viewID] + "-" + styleID + ".png";
        if (fs::exists(viewSketchName))
            sketchNames.push_back(viewSketchName);
    }

    mSketches.resize(numSketches);
    for (int sketchID = 0; sketchID < numSketches; sketchID++)
    {
        if (!parseSketch(sketchNames[sketchID], mSketches[sketchID]))
            return false;
    }

    if (numSketches)
    {
        if (mSketchSize >= 0)
        {
            mSketchSize = (int)mSketches.size();
        }
    }
    return true;
}

bool MapsData::parseSketch(const string &imgName, MatD &sketch)
{
    if (!fs::exists(imgName))
        return false;

    cv::Mat buffer = cv::imread(imgName, cv::IMREAD_UNCHANGED);
    sketch.resize(buffer.rows);
    for (int row = 0; row < buffer.rows; row++)
    {
        sketch[row].resize(buffer.cols);
        for (int col = 0; col < buffer.cols; col++)
        {
            double intensity = 0.0;

            if (buffer.channels() == 3)
            {
                const cv::Vec3b &bgr = buffer.at<cv::Vec3b>(row, col);
                for (int ch = 0; ch < 3; ch++)
                {
                    intensity += 1.0 - bgr.val[ch] / 255.0;
                }
                intensity /= 3;
            }
            else if (buffer.channels() == 1)
            {
                intensity = 1.0 - buffer.at<uchar>(row, col) / 255.0;
            }
            else
            {
                cerr << "sketch image unsupported channels!" << endl;
                return false;
            }

            if (intensity > 0.1)
            {
                sketch[row][col] = intensity;
            }
        }
    }
    return true;
}

bool MapsData::loadMask(const string &mapFolder, int numMaps)
{
    vector<string> imgNames;
    for (int viewID = 0; viewID < numMaps; viewID++)
    {
        string suffix = "mask-dn14--" + to_string(viewID) + ".png";
        string name = mapFolder + suffix;
        imgNames.push_back(name);
    }

    int numViews = imgNames.size();
    mMaskProbs.resize(numViews);
    cout << __FILE__ << " " << __LINE__ << " Load masks...\n";
    for (int viewID = 0; viewID < numViews; viewID++)
    {
        if (!parseMask(imgNames[viewID], mMaskProbs[viewID]))
            return false;
    }

    if (numViews)
    {
        mMapSize = mMaskProbs[0].size();
    }
    return true;
}

bool MapsData::parseMask(const string &imgName, MatD &maskProb)
{
    if (!fs::exists(imgName))
        return false;

    cv::Mat buffer = cv::imread(imgName, cv::IMREAD_UNCHANGED);
    maskProb.assign(buffer.rows, vector<double>(buffer.cols));
    for (int row = 0; row < buffer.rows; row++)
    {
        for (int col = 0; col < buffer.cols; col++)
        {
            maskProb[row][col] = buffer.at<ushort>(row, col) / (double)US_MAX;
        }
    }
    return true;
}

bool MapsData::visualizePoints(const string &fileName,
                               const Eigen::Matrix3Xd &positions, const Eigen::Matrix3Xd &normals)
{
    TPointSet ps;
    ps.positions.resize((int)positions.cols());
    ps.normals.resize((int)normals.cols());

    for (int i = 0; i < (int)positions.cols(); i++)
    {
        Eigen::Vector3d p = positions.col(i);
        ps.positions[i] = vec3d(p[0], p[1], p[2]);
    }

    for (int i = 0; i < (int)normals.cols(); i++)
    {
        Eigen::Vector3d n = normals.col(i);
        ps.normals[i] = vec3d(n[0], n[1], n[2]);
    }

    ps.amount = (int)ps.positions.size();

    MeshIO::savePointSet(fileName, ps);
}

bool MapsData::visualizeMask(string fileName, const MatI &mask)
{
    int height = mask.size();
    int width = mask[0].size();
    cv::Mat mat = cv::Mat::zeros(height, width, CV_8UC1);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (mask[i][j])
                mat.at<uchar>(i, j) = 255;
        }
    }

    return cv::imwrite(fileName, mat);
}

bool MapsData::visualizeDepth(string fileName, const MatD &depth)
{
    int height = depth.size();
    int width = depth[0].size();
    cv::Mat mat = cv::Mat::zeros(height, width, CV_8UC1);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double d = depth[i][j];
            mat.at<uchar>(i, j) = (d * 0.5 + 0.5) * 255;
        }
    }

    return cv::imwrite(fileName, mat);
}

bool MapsData::visualizeNormal(string fileName, const Mat3D &normal)
{
    int height = normal.size();
    int width = normal[0].size();
    cv::Mat mat = cv::Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            const vec3d &n = normal[i][j];
            cv::Vec3b &color = mat.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; c++)
            {
                color[c] = (n[c] * 0.5 + 0.5) * 255;
            }
        }
    }

    return cv::imwrite(fileName, mat);
}