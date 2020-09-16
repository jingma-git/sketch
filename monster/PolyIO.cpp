#include "PolyIO.h"
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

PolyIO::PolyIO()
{
    cleanUp();
}

void PolyIO::cleanUp()
{
    mVertices.clear();
    mNormals.clear();
    mColors.clear();
    mFaceIndices.clear();
    mEdgeIndices.clear();
    mFaceColors.clear();
    mComments.clear();

    hasNormal = false;
    hasColor = false;
    hasFaceColor = false;
}

bool PolyIO::addPoint(const vector<vec3> *vertices,
                      vector<vec3> *normals, matrix transform,
                      vec3i color)
{
    for (auto v : *vertices)
    {
        //TODO: transform vertex
        mVertices.push_back(v);
    }

    if (normals)
    {
        assert(vertices->size() == normals->size());

        //TODO: transform normal
        for (auto n : *normals)
        {
            n.normalize();
            mNormals.push_back(n);
        }
        hasNormal = true;
    }
    else
    {
        for (int i = 0; i < vertices->size(); i++)
        {
            mNormals.push_back(vec3(0.f, 1.f, 0.f));
        }
    }

    if (color != vec3i(0, 0, 0))
    {
        hasColor = true;
    }

    for (int i = 0; i < vertices->size(); i++)
    {
        mColors.push_back(color);
    }
    return true;
}

bool PolyIO::output(string fileName, bool ascii)
{
    // https://blog.csdn.net/shine_cherise/article/details/79435774
    if (hasNormal && mNormals.size() != mVertices.size())
        return false;
    if (hasColor && mColors.size() != mVertices.size())
        return false;

    ofstream outFile;

    outFile.open(fileName);
    {
        outFile << "ply" << endl;
        if (ascii)
        {
            outFile << "format ascii 1.0" << endl;
        }
        else
        {
            outFile << "format binary_little_endian 1.0" << endl;
        }
        if (!mComments.empty())
        {
            for (auto comment : mComments)
            {
                outFile << "comment " << comment << endl;
            }
        }
        outFile << "element vertex " << mVertices.size() << endl;
        outFile << "property float x" << endl;
        outFile << "property float y" << endl;
        outFile << "property float z" << endl;

        if (hasNormal)
        {
            outFile << "property float nx" << endl;
            outFile << "property float ny" << endl;
            outFile << "property float nz" << endl;
        }

        if (hasColor)
        {
            outFile << "property uchar red" << endl;
            outFile << "property uchar green" << endl;
            outFile << "property uchar blue" << endl;
            outFile << "property uchar alpha" << endl;
        }

        if (mFaceIndices.size())
        {
            outFile << "element face " << mFaceIndices.size() << endl;
            outFile << "property list uchar int vertex_indices" << endl;

            if (hasFaceColor)
            {
                outFile << "property uchar red" << endl;
                outFile << "property uchar green" << endl;
                outFile << "property uchar blue" << endl;
                outFile << "property uchar alpha" << endl;
            }
        }

        if (mEdgeIndices.size())
        {
            outFile << "element edge " << mEdgeIndices.size() << endl;

            outFile << "property int vertex1" << endl;
            outFile << "property int vertex2" << endl;
        }

        outFile << "end_header" << endl;
    }

    if (ascii)
    {

        for (int i = 0; i < (int)(mVertices.size()); i++)
        {

            outFile << mVertices[i] << " ";

            if (hasNormal)
            {
                outFile << mNormals[i] << " ";
            }

            if (hasColor)
            {
                outFile << mColors[i] << " 255 ";
            }

            outFile << endl;
        }

        if (mFaceIndices.size())
        {
            int numFaces = (int)mFaceIndices.size();
            for (int i = 0; i < numFaces; i++)
            {
                outFile << "3 " << mFaceIndices[i] << endl;
                if (hasFaceColor)
                {
                    if (i < (int)mFaceColors.size())
                    {
                        outFile << mFaceColors[i] << " 255" << endl;
                    }
                    else
                    {
                        outFile << "0 0 0 255" << endl;
                    }
                }
            }
        }

        if (mEdgeIndices.size())
        {
            int numEdges = (int)mEdgeIndices.size();
            for (int i = 0; i < numEdges; i++)
            {
                outFile << mEdgeIndices[i] << endl;
            }
        }
    }
    else
    {
        outFile.close();

        outFile.open(fileName, ios::app | ios::binary);
        {
            for (int i = 0; i < (int)(mVertices.size()); i++)
            {

                outFile.write((char *)(mVertices[i].data()), sizeof(vec3));

                if (hasNormal)
                {
                    outFile.write((char *)(mNormals[i].data()), sizeof(vec3));
                }

                if (hasColor)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        outFile.put((unsigned char)(mColors[i][c]));
                    }
                    outFile.put(255u); // alpha as padding
                }
            }

            if (mFaceIndices.size())
            {
                int numFaces = (int)mFaceIndices.size();
                for (int i = 0; i < numFaces; i++)
                {
                    outFile.put(3u);
                    outFile.write((char *)(mFaceIndices[i].data()), sizeof(vec3i));
                    if (hasFaceColor)
                    {
                        if (i < (int)mFaceColors.size())
                        {
                            for (int c = 0; c < 3; c++)
                            {
                                outFile.put((unsigned char)(mFaceColors[i][c]));
                            }
                            outFile.put(255u); // alpha as padding
                        }
                        else
                        {
                            for (int c = 0; c < 3; c++)
                            {
                                outFile.put(0u);
                            }
                            outFile.put(255u); // alpha as padding
                        }
                    }
                }
            }

            if (mEdgeIndices.size())
            {
                int numEdges = (int)mEdgeIndices.size();
                for (int i = 0; i < numEdges; i++)
                {
                    outFile.write((char *)(mEdgeIndices[i].data()), sizeof(vec2i));
                }
            }
        }
    }

    outFile.close();
    return true;
}