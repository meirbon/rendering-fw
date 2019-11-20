#include "SceneNode.h"

#include "SceneObject.h"
#include "MeshSkin.h"

rfw::SceneNode::SceneNode(SceneObject *obj, std::string n, glm::mat4 transform, rfw::utils::ArrayProxy<int> c)
	: object(obj), name(n), localTransform(transform), matrix(transform)
{
	childIndices.resize(c.size());
	memcpy(childIndices.data(), c.data(), c.size() * sizeof(int));
}

bool rfw::SceneNode::update(const glm::mat4 &accumulatedTransform)
{
	bool changed = false;

	if (transformed)
	{
		calculateTransform();
		transformed = false;
	}

	this->combinedTransform = accumulatedTransform * localTransform;

	for (size_t s = childIndices.size(), i = 0; i < s; i++)
	{
		rfw::SceneNode &child = object->nodes.at(childIndices.at(i));
		changed = child.update(accumulatedTransform);
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

void rfw::SceneNode::calculateTransform() {}
