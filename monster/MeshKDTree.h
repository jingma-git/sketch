
#pragma once

#include "TheaKDTreeHelper.h"

#include "Types.h"

using namespace std;

namespace Monster
{

	class MeshKDTree
	{

	private:
		// make it non-instantiable
		MeshKDTree() {}
		~MeshKDTree() {}

	public:
		static bool buildKdTree(vector<vec3> &points, SKDTree &tree, SKDTreeData &treeData); // kd tree of points
		static bool buildKdTree(TTriangleMesh &mesh, TKDTree &tree, TKDTreeData &treeData);	 // kd tree of face triangles
		static bool buildKdTree(TTriangleMesh &mesh, SKDTree &tree, SKDTreeData &treeData);	 // kd tree of face center points

		static bool distanceToPoints(vector<vec3> &points, SKDTree &tree, vec3 &point, double &distance);
		static bool distanceToMesh(TTriangleMesh &mesh, SKDTree &faceTree, vec3 &point, double &distance);
		static bool checkInsideMesh(TKDTree &tree, vec3 &point, bool &insideFlag, double eps = 1e-5);
	};
} // namespace Monster