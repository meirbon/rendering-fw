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

	if (meshID > -1)
	{
		if (morphed)
		{
			object->meshes.at(meshID).setPose(weights);
			morphed = false;
		}
		else if (skinID > -1)
		{
			auto &skin = object->skins.at(skinID);

			for (size_t s = skin.joints.size(), j = 0; j < s; j++)
			{
				auto &jointNode = object->nodes.at(skin.joints.at(j));
				skin.jointMatrices.at(j) =
					combinedTransform * jointNode.combinedTransform * skin.inverseBindMatrices.at(j);
			}

			object->meshes.at(meshID).setPose(skin);
		}
		else if (!hasUpdatedStatic)
		{
			object->meshes.at(meshID).setTransform(combinedTransform);
			hasUpdatedStatic = true;
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
