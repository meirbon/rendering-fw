#include "SceneNode.h"

#include "SceneObject.h"
#include "MeshSkin.h"

rfw::SceneNode::SceneNode(SceneObject *obj, std::string n, rfw::utils::ArrayProxy<int> c) : object(obj), name(n)
{
	childIndices.resize(c.size());
	memcpy(childIndices.data(), c.data(), c.size() * sizeof(int));

	translation = vec3(0);
	rotation = glm::identity<glm::quat>();
	scale = vec3(1);
	matrix = glm::identity<glm::mat4>();
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

				for (size_t s = skin.joints.size(), j = 0; j < s; j++)
				{
					auto &jointNode = object->nodes.at(skin.joints.at(j));

					// Create a row major matrix for SIMD accelerated scene updates
					skin.jointMatrices.at(j) = glm::rowMajor4(combinedTransform * jointNode.combinedTransform * skin.inverseBindMatrices.at(j));
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
