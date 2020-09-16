#include "MeshIO.h"
#include "PolyIO.h"
#include <iostream>
using namespace Monster;

bool MeshIO::savePointSet(string fileName, TPointSet &ps, bool ascii)
{
    string ext = fileName.substr(fileName.find_last_of(".") + 1);
    if (ext == "ply")
    {
        PolyIO polyIO;
        polyIO.addPoint(&ps.positions, &ps.normals);
        polyIO.output(fileName, true);
    }
    else
    {
        cout << "unsupported point set format" << endl;
        return false;
    }
    return true;
}