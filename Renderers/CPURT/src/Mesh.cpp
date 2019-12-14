//
// Created by meirn on 11/12/2019.
//

#include "Mesh.h"

using namespace rfw;

void rfw::CPUMesh::setGeometry(const Mesh &mesh)
{
	triangles = const_cast<rfw::Triangle *>(mesh.triangles);
	vertexCount = mesh.vertexCount;
	triangleCount = mesh.triangleCount;

	if (!bvh)
	{
		if (mesh.hasIndices())
		{
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
			bvh->constructBVH();
			mbvh = new MBVHTree(bvh);
			mbvh->constructBVH();
		}
		else
		{
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount);
			bvh->constructBVH();
			mbvh = new MBVHTree(bvh);
			mbvh->constructBVH();
		}
	}
	else
	{
		if (mesh.hasIndices())
		{
			if (mesh.vertexCount == vertexCount && mesh.triangleCount == triangleCount)
			{
				delete mbvh;
				bvh->setVertices(mesh.vertices);
				bvh->constructBVH();
				mbvh = new MBVHTree(bvh);
				mbvh->constructBVH();
			}
			else
			{
				delete bvh;
				delete mbvh;

				bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
				bvh->constructBVH();
				mbvh = new MBVHTree(bvh);
				mbvh->constructBVH();
			}
		}
		else
		{
			if (mesh.vertexCount == vertexCount && mesh.triangleCount == triangleCount)
			{
				delete mbvh;
				bvh->setVertices(mesh.vertices);
				bvh->constructBVH();
				mbvh = new MBVHTree(bvh);
				mbvh->constructBVH();
			}
			else
			{
				delete bvh;
				delete mbvh;

				bvh = new BVHTree(mesh.vertices, mesh.vertexCount);
				bvh->constructBVH();
				mbvh = new MBVHTree(bvh);
				mbvh->constructBVH();
			}
		}
	}
}
