#pragma once

#include "BVH/BVHTree.h"
#include "BVH/MBVHTree.h"
#include <Structures.h>

namespace rfw
{
class CPUMesh
{
  public:
	CPUMesh();
	CPUMesh(const CPUMesh &other);
	~CPUMesh();
	
	void setGeometry(const Mesh &mesh);

	BVHTree *bvh = nullptr;
	MBVHTree *mbvh = nullptr;
	rfw::Triangle *triangles = nullptr;

  private:
	int vertexCount;
	int triangleCount;
};
} // namespace rfw