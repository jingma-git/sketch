
#pragma once

#include "Types.h"

namespace Monster
{

	class MeshCompute
	{

	private:
		// make it non-instantiable
		MeshCompute() {}
		~MeshCompute() {}

	public:
		static bool recomputeNormals(TTriangleMesh &mesh);
		static bool recomputeNormals(vector<vec3> &points, vector<vec3> &normals, int numNeighbors = 20, double neighborDist = 0.1);
		static bool computeDihedralAngle(vec3 center1, vec3 normal1, vec3 center2, vec3 normal2, double &angle);
		static bool computeAABB(TPointSet &mesh, vec3 &bbMin, vec3 &bbMax);
		static bool computeBoundingSphere(TPointSet &mesh, vec3 &center, float &radius);
		static bool computeFaceArea(TTriangleMesh &mesh, double &area);
		static bool computeMassCenter(TTriangleMesh &mesh, vec3 &center);
	};
} // namespace Monster