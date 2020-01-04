#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <Texture.h>

#include <assimp/material.h>

#include <Structures.h>
#include <DeviceStructures.h>

namespace rfw
{
class MaterialList;
class HostMaterial
{
  public:
	struct MapProps
	{
		int textureID = -1;					  // texture ID; -1 denotes empty slot
		float valueScale = 1;				  // texture value scale, only sensible for normal maps
		glm::vec2 uvscale = glm::vec2(1.0f);  // uv coordinate scale
		glm::vec2 uvoffset = glm::vec2(0.0f); // uv coordinate offset
	};
	HostMaterial()
	{
		for (auto &i : map)
			i.textureID = -1;
	}
	enum
	{
		SMOOTH = 1,		   // material uses normal interpolation
		HASALPHA = 2,	   // material textures use alpha channel
		ANISOTROPIC = 4,   // material has anisotropic roughness
		FROM_MTL = 128,	   // changes are persistent for these, not for others
		ISCONDUCTOR = 256, // rough conductor
		ISDIELECTRIC = 512 // rough dielectric. If 256 and 512 not specified: diffuse.
	};

	std::string name = "unnamed"; // material name, not for unique identification
	std::string origin;			  // origin: file from which the data was loaded, with full path
	int ID = -1;				  // unique integer ID of this material
	uint flags = SMOOTH;		  // material properties
	uint refCount = 1;			  // the number of models that use this material

	glm::vec3 color = glm::vec3(1.f);
	glm::vec3 absorption = glm::vec3(0.f);
	float metallic = 0.0f;
	float subsurface = 0.0f;
	float specular = 0.5f;
	float roughness = 0.5f;
	float specularTint = 0.0f;
	float anisotropic = 0.0f;
	float sheen = 0.0f;
	float sheenTint = 0.0f;
	float clearcoat = 0.0f;
	float clearcoatGloss = 1.0f;
	float transmission = 0.0f;
	float eta = 1.0f;
	float custom0 = 0.0f;
	float custom1 = 0.0f;
	float custom2 = 0.0f;
	float custom3 = 0.0f;
	MapProps map[11]; // bitmap data
	// field for the BuildMaterialList method of HostMesh
	bool visited = false; // last mesh that checked this material

	void setFlag(MatPropFlags flag) { flags |= (1u << uint(flag)); }
	[[nodiscard]] bool hasFlag(MatPropFlags flag) const { return (flags & (1u << uint(flag))); }

	std::pair<Material, MaterialTexIds> convertToDeviceMaterial(MaterialList *list) const;

	[[nodiscard]] bool isEmissive() const { return any(greaterThan(color, vec3(1.0f))); }
};

class MaterialList
{
  public:
	~MaterialList();

	uint add(const HostMaterial &mat);
	uint add(const aiMaterial *aiMat, const std::string_view &basedir);
	uint add(const Texture &tex);
	void set(uint index, const HostMaterial &mat);

	void generateDeviceMaterials();

	[[nodiscard]] bool isDirty() const { return m_IsDirty; }

	[[nodiscard]] const HostMaterial &get(uint index) const { return m_HostMaterials.at(index); }
	[[nodiscard]] HostMaterial &get(uint index) { return m_HostMaterials.at(index); }

	[[nodiscard]] const std::vector<HostMaterial> &getMaterials() const { return m_HostMaterials; }
	[[nodiscard]] const std::vector<DeviceMaterial> &getDeviceMaterials() const { return m_Materials; }
	[[nodiscard]] const std::vector<MaterialTexIds> &getMaterialTexIds() const { return m_MaterialTexIds; }
	[[nodiscard]] const std::vector<Texture> &getTextures() const { return m_Textures; }
	[[nodiscard]] const std::vector<TextureData> &getTextureDescriptors() const { return m_TextureDescriptors; }
	[[nodiscard]] const std::vector<bool> &getMaterialLightFlags() const { return m_IsEmissive; }

  private:
	bool m_IsDirty = true;
	int getTextureIndex(const std::string_view &file);

	std::mutex m_MatMutex;
	std::mutex m_TexMutex;

	std::vector<bool> m_IsEmissive;
	std::vector<HostMaterial> m_HostMaterials;
	std::vector<MaterialTexIds> m_MaterialTexIds;
	std::vector<DeviceMaterial> m_Materials;
	std::vector<Texture> m_Textures;
	std::vector<TextureData> m_TextureDescriptors;
	std::map<std::string, int> m_TexMapping;
};
} // namespace rfw
