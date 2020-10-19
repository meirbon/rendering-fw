#include <rfw/rfw.h>

#include <rfw/internal.h>

#define ALLOW_INDEXED_ANIM_DATA 1

using namespace rfw;
using namespace geometry;
using namespace gltf;

SceneNode::SceneNode(SceneObject *obj, std::string n, rfw::utils::array_proxy<int> c, rfw::utils::array_proxy<int> meshIds,
						  rfw::utils::array_proxy<int> skinIds, rfw::utils::array_proxy<std::vector<TmpPrim>> meshes, Transform T, glm::mat4 transform)
	: object(obj), name(std::move(n))
{
	childIndices.resize(c.size());
	memcpy(childIndices.data(), c.data(), c.size() * sizeof(int));

	translation = T.translation;
	rotation = T.rotation;
	scale = T.scale;
	matrix = transform;

	calculate_transform();

	for (int i = 0; i < meshIds.size(); i++)
	{
		int meshID = meshIds[i];
		if (meshID > -1)
		{
			const auto newIdx = object->meshes.size();
			object->meshes.emplace_back(*object);
			object->meshTranforms.emplace_back(localTransform);

			auto &m = object->meshes.at(newIdx);
			for (const auto &prim : meshes.at(meshID))
				m.add_primitive(prim.indices, prim.vertices, prim.normals, prim.uvs, prim.poses, prim.joints, prim.weights, prim.matID);
			meshIDs.emplace_back(static_cast<int>(newIdx));

			if (skinIds.has(i))
				skinIDs.push_back(skinIds[i]);
			else
				skinIDs.push_back(-1);
		}
	}

	if (!meshIDs.empty())
	{
		const auto morphTargets = glm::max(object->meshes[meshIDs[0]].poses.size(), size_t(1)) - 1;
		if (morphTargets > 0)
			weights.resize(morphTargets, 0.0f);
	}
}

bool SceneNode::update(rfw::simd::matrix4 accumulatedTransform)
{
	bool changed = transformed;
	if (transformed)
		calculate_transform();

	combinedTransform = accumulatedTransform * localTransform;

	for (size_t s = childIndices.size(), i = 0; i < s; i++)
	{
		auto &child = object->nodes.at(childIndices.at(i));
		changed = child.update(combinedTransform);
	}

	if (!meshIDs.empty())
	{
		for (int i = 0; i < meshIDs.size(); i++)
		{
			const int meshID = meshIDs[i];

			object->changedMeshNodeTransforms[meshID] = true;

			if (changed || !hasUpdatedStatic)
			{
				object->meshTranforms[meshID] = combinedTransform;
				hasUpdatedStatic = true;
				changed = true;
			}

			if (morphed)
			{
				object->meshes.at(meshID).set_pose(weights);
				morphed = false;
				changed = true;
			}

			if (skinIDs.at(i) != -1)
			{
				auto &skin = object->skins.at(skinIDs[i]);
				const auto inverse_transform = combinedTransform.inversed();
				for (int s = static_cast<int>(skin.jointNodes.size()), j = 0; j < s; j++)
				{
					const auto &jointNode = object->nodes[skin.jointNodes[j]];
					skin.jointMatrices[j] = inverse_transform * jointNode.combinedTransform * skin.inverseBindMatrices[j];
				}

				object->meshes[meshID].set_pose(skin);
				changed = true;
			}
		}
	}

	return changed;
}

void SceneNode::calculate_transform()
{
	const mat4 T = translate(mat4(1.0f), translation);
	const mat4 R = glm::mat4(rotation);
	const mat4 S = glm::scale(mat4(1.0f), scale);

	localTransform = T * R * S * matrix;
	transformed = false;
}
