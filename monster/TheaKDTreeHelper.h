#pragma once

/*
	Added (for VS2015 compatibility):
		TheaKDTree/Vector.hpp#L23:
			#include <algorithm>
	Modified (removed boost library dependency):
		TheaKDTree/Util.hpp
		TheaKDTree/KDTree3.hpp			
*/

#include "TheaKDTree/KDTree3.hpp"

namespace SKDT
{
    struct NamedPoint
    {
        G3D::Vector3 position;
        size_t id;

        NamedPoint() {}
        NamedPoint(float x, float y, float z) : position(x, y, z) {}
        NamedPoint(float x, float y, float z, const size_t _id) : position(x, y, z), id(_id) {}
    };
} // namespace SKDT

namespace TKDT
{
    struct NamedTriangle
    {
        G3D::Vector3 v[3];
        size_t id;

        NamedTriangle() {}
        NamedTriangle(G3D::Vector3 const &v0_, G3D::Vector3 const &v1_, G3D::Vector3 const &v2_, size_t _id)
        {
            v[0] = v0_;
            v[1] = v1_;
            v[2] = v2_;
            id = _id;
        }
        NamedTriangle(NamedTriangle const &src)
        {
            v[0] = src.v[0];
            v[1] = src.v[1];
            v[2] = src.v[2];
            id = src.id;
        }

        // Get the i'th vertex. This is the operative function that a vertex triple needs to have.
        G3D::Vector3 const &getVertex(int i) const { return v[i]; }
    };
} // namespace TKDT

namespace Thea
{
    template <>
    struct PointTraits3<SKDT::NamedPoint>
    {
        static G3D::Vector3 const &getPosition(SKDT::NamedPoint const &np) { return np.position; }
    };

    template <>
    struct IsPoint3<SKDT::NamedPoint>
    {
        static bool const value = true;
    };
} // namespace Thea

typedef Thea::KDTree3<SKDT::NamedPoint> SKDTree;
typedef std::vector<SKDT::NamedPoint> SKDTreeData;

typedef Thea::Triangle3<TKDT::NamedTriangle> TKDTreeElement;
typedef Thea::KDTree3<TKDTreeElement> TKDTree;
typedef std::vector<TKDTreeElement> TKDTreeData;