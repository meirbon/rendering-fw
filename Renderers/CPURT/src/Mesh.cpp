#include "Mesh.h"

using namespace rfw;

CPUMesh::CPUMesh()
{
	bvh = nullptr;
	mbvh = nullptr;
	vertexCount = 0;
	triangleCount = 0;
}

CPUMesh::CPUMesh(const CPUMesh &other)
{
	memcpy(this, &other, sizeof(CPUMesh));
	memset(const_cast<CPUMesh *>(&other), 0, sizeof(CPUMesh));
}

CPUMesh::~CPUMesh()
{
	delete bvh;
	delete mbvh;

	bvh = nullptr;
	mbvh = nullptr;
}

void rfw::CPUMesh::setGeometry(const Mesh &mesh)
{
	triangles = const_cast<rfw::Triangle *>(mesh.triangles);
	const bool rebuild = vertexCount != mesh.vertexCount;

	vertexCount = mesh.vertexCount;
	triangleCount = mesh.triangleCount;

	if (!bvh)
	{
		if (mesh.hasIndices())
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
		else
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount);

		bvh->constructBVH(true);
		mbvh = new MBVHTree(bvh);
		mbvh->constructBVH(true);
	}
	else
	{
		if (rebuild) // Full rebuild of BVH
		{
			delete bvh;
			delete mbvh;

			if (mesh.hasIndices())
				bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
			else
				bvh = new BVHTree(mesh.vertices, mesh.vertexCount);

			bvh->constructBVH();
			mbvh = new MBVHTree(bvh);
			mbvh->constructBVH();
		}
		else // Keep same BVH but refit all nodes
		{
#if 0
			delete mbvh;
			if (mesh.hasIndices())
				bvh->refit(mesh.vertices, mesh.indices);
			else
				bvh->refit(mesh.vertices);

			mbvh = new MBVHTree(bvh);
			mbvh->constructBVH();
#else
			if (mesh.hasIndices())
				mbvh->refit(mesh.vertices, mesh.indices);
			else
				mbvh->refit(mesh.vertices);
#endif
		}
	}
}
