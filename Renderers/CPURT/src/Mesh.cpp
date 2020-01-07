#include "PCH.h"

using namespace rfw;

#define REFIT 0

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
	const bool rebuild = !bvh || (vertexCount != mesh.vertexCount);

	vertexCount = mesh.vertexCount;
	triangleCount = mesh.triangleCount;

#if REFIT
	if (rebuild) // Full rebuild of BVH
	{
		delete bvh;
		delete mbvh;

		if (mesh.hasIndices())
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
		else
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount);

		bvh->construct_bvh();
		mbvh = new MBVHTree(bvh);
		mbvh->constructBVH();
	}
	else // Keep same BVH but refit nodes
	{
		if (mesh.hasIndices())
			mbvh->refit(mesh.vertices, mesh.indices);
		else
			mbvh->refit(mesh.vertices);
	}
#else
	delete bvh;
	delete mbvh;

	if (mesh.hasIndices())
		bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
	else
		bvh = new BVHTree(mesh.vertices, mesh.vertexCount);

	bvh->construct_bvh();
	mbvh = new MBVHTree(bvh);
	mbvh->construct_bvh();
#endif
}
