#include "SceneNode.h"

#include <utility>

#include "SceneObject.h"
#include "MeshSkin.h"

rfw::SceneNode::SceneNode(SceneObject *obj, std::string n, rfw::utils::ArrayProxy<int> c, rfw::utils::ArrayProxy<int> meshIds,
						  rfw::utils::ArrayProxy<int> skinIds, rfw::utils::ArrayProxy<std::vector<TmpPrim>> meshes, Transform T, glm::mat4 transform)
	: object(obj), name(std::move(n))
{
	childIndices.resize(c.size());
	memcpy(childIndices.data(), c.data(), c.size() * sizeof(int));

	translation = T.translation;
	rotation = T.rotation;
	scale = T.scale;
	matrix = transform;

	for (int i = 0; i < meshIds.size(); i++)
	{
		int meshID = meshIds[i];
		if (meshID > -1)
		{
			const auto newIdx = object->meshes.size();
			object->meshes.emplace_back();
			auto &m = object->meshes.at(newIdx);
			m.object = object;
			for (const auto &prim : meshes.at(meshID))
				m.addPrimitive(prim.indices, prim.vertices, prim.normals, prim.uvs, prim.poses, prim.joints, prim.weights, prim.matID);
			meshIDs.push_back(newIdx);

			if (skinIds.has(i))
				skinIDs.push_back(skinIds[i]);
			else
				skinIDs.push_back(-1);
		}
	}

	if (!meshIDs.empty())
	{
		const auto morphTargets = glm::max(object->meshes.at(meshIDs.at(0)).poses.size(), size_t(1)) - 1;
		if (morphTargets > 0)
			weights.resize(morphTargets, 0.0f);
	}

	calculateTransform();
}

bool rfw::SceneNode::update(glm::mat4 accumulatedTransform)
{
	bool changed = false;
	if (transformed)
		calculateTransform();

	combinedTransform = accumulatedTransform * localTransform;

	for (size_t s = childIndices.size(), i = 0; i < s; i++)
	{
		rfw::SceneNode &child = object->nodes.at(childIndices.at(i));
		changed = child.update(combinedTransform);
	}

	if (!meshIDs.empty())
	{
		for (int i = 0; i < meshIDs.size(); i++)
		{
			const int meshID = meshIDs.at(i);

			if (morphed)
			{
				object->meshes.at(meshID).setPose(weights);
				morphed = false;
				changed = true;
			}
			else if (skinIDs.at(i) != -1)
			{
				auto &skin = object->skins.at(skinIDs.at(i));

				for (int s = static_cast<int>(skin.jointNodes.size()), j = 0; j < s; j++)
				{
					auto &jointNode = object->nodes.at(skin.jointNodes.at(j));

// Create a row major matrix for SIMD accelerated scene updates
#if ROW_MAJOR_MESH_SKIN
					skin.jointMatrices.at(j) = rowMajor4(combinedTransform * jointNode.combinedTransform * skin.inverseBindMatrices.at(j));
#else
					skin.jointMatrices.at(j) = combinedTransform * jointNode.combinedTransform * skin.inverseBindMatrices.at(j);
#endif
				}

				object->meshes.at(meshID).setPose(skin);
				changed = true;
			}
			else if (!hasUpdatedStatic)
			{
				object->meshes.at(meshID).setTransform(combinedTransform);
				hasUpdatedStatic = true;
				changed = true;
			}
		}
	}

	return changed;
}

void rfw::SceneNode::calculateTransform()
{
	const mat4 T = translate(mat4(1.0f), translation);
	const mat4 R = mat4_cast(rotation);
	const mat4 S = glm::scale(mat4(1.0f), scale);

	localTransform = T * R * S * matrix;
	transformed = false;
}
