#include "rfw.h"
#include "MaterialList.h"

using namespace rfw;

static float RoughnessToAlpha(float roughness)
{
	roughness = fmaxf(roughness, 1e-4f);
	const float x = logf(roughness);
	return fminf(1.0f, (1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x));
}

MaterialList::MaterialList() { add(HostMaterial()); }

MaterialList::~MaterialList()
{
	for (auto tex : m_Textures)
		tex.cleanup();

	m_Textures.clear();
	m_TextureDescriptors.clear();
}

uint MaterialList::add(const HostMaterial &mat)
{
	auto lock = std::lock_guard(m_MatMutex);
	const auto idx = m_HostMaterials.size();
	m_IsEmissive.push_back(mat.isEmissive());
	m_HostMaterials.push_back(mat);
	return static_cast<unsigned int>(idx);
}

uint MaterialList::add(const aiMaterial *aiMat, const std::string_view &basedir)
{
	HostMaterial mat{};

	aiColor3D ambient = {0, 0, 0}, diffuse = {0, 0, 0}, specular = {0, 0, 0}, emissive = {0, 0, 0}, transparent = {0, 0, 0};
	float opacity = 0, shininess = 0, shininessStrength = 0, eta = 0, reflectivity = 0;

	aiMat->Get(AI_MATKEY_COLOR_AMBIENT, ambient);
	aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
	aiMat->Get(AI_MATKEY_COLOR_SPECULAR, specular);
	aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
	aiMat->Get(AI_MATKEY_COLOR_TRANSPARENT, transparent);

	aiMat->Get(AI_MATKEY_OPACITY, opacity);
	aiMat->Get(AI_MATKEY_SHININESS, shininess);
	aiMat->Get(AI_MATKEY_SHININESS_STRENGTH, shininessStrength);
	aiMat->Get(AI_MATKEY_REFRACTI, eta);
	aiMat->Get(AI_MATKEY_REFLECTIVITY, reflectivity);

	if (any(notEqual(vec3(emissive.r, emissive.g, emissive.b), vec3(0.0f))))
		mat.color = max(vec3(emissive.r, emissive.g, emissive.b), vec3(0.0f));
	else
		mat.color = max(vec3(diffuse.r, diffuse.g, diffuse.b), vec3(0.0f));

	mat.absorption = max(vec3(transparent.r, transparent.g, transparent.b), vec3(0.0f));
	mat.metallic = max(reflectivity, 0.0f);
	mat.subsurface = max(0.0f, 0.0f);
	if (reflectivity > 0)
		mat.metallic = reflectivity;
	if (shininessStrength > 0 && shininessStrength < 1)
		mat.specular = shininessStrength;
	else
		mat.specular = 0.0f;

	if (shininess > 0.0f)
		mat.roughness = max(0.0f, 1.0f - sqrt(min(shininess, 1024.0f) / 1024.0f));
	else
		mat.roughness = 1.0f;

	if (eta > 1.0f)
		mat.eta = eta;

	if (opacity != 0.0f)
		mat.transmission = 1.0f - max(opacity, 0.0f);

	mat.setFlag(MatPropFlags::HasSmoothNormals);

	const std::string base = std::string(basedir) + '/';

	for (uint i = 0; i < aiMat->GetTextureCount(aiTextureType_DIFFUSE) && i < 3; i++)
	{
		if (i == 0 && all(glm::equal(mat.color, vec3(0.0f))))
			mat.color = vec3(1.0f);

		aiString str;
		aiMat->GetTexture(aiTextureType_DIFFUSE, i, &str);
		const std::string file = base + str.C_Str();

		const int idx = getTextureIndex(file);

		switch (i)
		{
		case (0):
			if (idx >= 0 && m_Textures.at(idx).flags & Texture::HAS_ALPHA)
				mat.setFlag(HasAlpha);
			mat.setFlag(HasDiffuseMap);
			mat.map[TEXTURE0].textureID = idx;
			break;
		case (1):
			mat.setFlag(Has2ndDiffuseMap);
			mat.map[TEXTURE1].textureID = idx;
			break;
		case (2):
			mat.setFlag(MatPropFlags::Has3rdDiffuseMap);
			mat.map[TEXTURE2].textureID = idx;
			break;
		default:
			break;
		}
	}

	for (uint i = 0; i < aiMat->GetTextureCount(aiTextureType_NORMALS) && i < 3; i++)
	{
		aiString str;
		aiMat->GetTexture(aiTextureType_NORMALS, i, &str);
		const std::string file = base + str.C_Str();

		const uint idx = getTextureIndex(file);
		switch (i)
		{
		case (0):
			mat.setFlag(MatPropFlags::HasNormalMap);
			mat.map[NORMALMAP0].textureID = idx;
			break;
		case (1):
			mat.setFlag(MatPropFlags::Has2ndNormalMap);
			mat.map[NORMALMAP1].textureID = idx;
			break;
		case (2):
			mat.setFlag(MatPropFlags::Has3rdNormalMap);
			mat.map[NORMALMAP2].textureID = idx;
			break;
		default:
			break;
		}
	}

	if (aiMat->GetTextureCount(aiTextureType_SPECULAR) > 0)
	{
		aiString str;
		aiMat->GetTexture(aiTextureType_SPECULAR, 0, &str);
		const std::string file = base + str.C_Str();

		mat.setFlag(MatPropFlags::HasSpecularityMap);
		mat.map[SPECULARITY].textureID = getTextureIndex(file);
	}

	if (aiMat->GetTextureCount(aiTextureType_SHININESS) > 0)
	{
		aiString str;
		aiMat->GetTexture(aiTextureType_SHININESS, 0, &str);
		const std::string file = base + str.C_Str();

		mat.setFlag(MatPropFlags::HasRoughnessMap);
		mat.map[ROUGHNESS0].textureID = getTextureIndex(file);
	}

	if (aiMat->GetTextureCount(aiTextureType_OPACITY) > 0 && mat.map[TEXTURE0].textureID != -1)
	{
		aiString str;
		aiMat->GetTexture(aiTextureType_OPACITY, 0, &str);
		const std::string file = base + str.C_Str();
		mat.setFlag(MatPropFlags::HasAlphaMap);
		const int textureIndex = getTextureIndex(file);
		mat.map[ALPHAMASK].textureID = textureIndex;
	}

	auto lock = std::lock_guard(m_MatMutex);
	const uint idx = static_cast<uint>(m_HostMaterials.size());
	m_IsEmissive.push_back(mat.isEmissive());
	m_HostMaterials.push_back(mat);
	return idx;
}

uint rfw::MaterialList::add(const Texture &tex)
{
	uint idx = (uint)m_Textures.size();

	m_Textures.push_back(tex);

	TextureData data{};
	if (tex.type == Texture::FLOAT4)
	{
		data.type = TextureData::FLOAT4;
		data.data = tex.fdata;
	}
	else if (tex.type == Texture::UNSIGNED_INT)
	{
		data.type = TextureData::UINT;
		data.data = tex.udata;
	}

	data.width = tex.width;
	data.height = tex.height;
	data.texAddr = idx;
	data.texelCount = tex.texelCount;
	m_TextureDescriptors.emplace_back(data);

	return idx;
}

void rfw::MaterialList::set(uint index, const HostMaterial &mat)
{
	bool valid = true;

	for (int i = 0; i < 11; i++)
	{
		if (mat.map[i].textureID > 0 && mat.map[i].textureID >= m_Textures.size())
			valid = false, WARNING("Material %i: map %i has invalid index %i", index, i, mat.map[i].textureID);
	}

	if (!valid)
	{
		WARNING("Invalid replacement material for %i, not updating", index);
		return;
	}

	m_HostMaterials[index] = mat;
	m_IsEmissive[index] = mat.isEmissive();
	m_IsDirty = true;
}

void MaterialList::generateDeviceMaterials()
{
	m_Materials.resize(m_HostMaterials.size());
	m_MaterialTexIds.resize(m_HostMaterials.size());
	for (size_t i = 0, s = m_HostMaterials.size(); i < s; i++)
	{
		const auto &hostMat = m_HostMaterials.at(i);
		const auto [mat, matTexIds] = hostMat.convertToDeviceMaterial(this);
		memcpy(&m_Materials.at(i), &mat, sizeof(DeviceMaterial));
		m_MaterialTexIds.at(i) = matTexIds;
		const auto m = static_cast<const Material &>(mat);
	}

	m_IsDirty = false;
}

int MaterialList::getTextureIndex(const std::string_view &file)
{
	const auto filename = std::string(file.data());

	if (m_TexMapping.find(filename) != m_TexMapping.end())
		return m_TexMapping.at(filename);

	auto lock = std::lock_guard(m_TexMutex);
	const auto idx = static_cast<uint>(m_Textures.size());
	m_TexMapping[filename] = idx;

	try
	{
		m_Textures.emplace_back(filename);
		m_TexMapping[filename] = idx;
	}
	catch (const std::exception &e)
	{
		WARNING(e.what());
		m_TexMapping[filename] = -1;
		return -1;
	}

	const auto &tex = m_Textures.at(idx);

	TextureData data{};
	if (tex.type == Texture::FLOAT4)
	{
		data.type = TextureData::FLOAT4;
		data.data = tex.fdata;
	}
	else if (tex.type == Texture::UNSIGNED_INT)
	{
		data.type = TextureData::UINT;
		data.data = tex.udata;
	}

	data.width = tex.width;
	data.height = tex.height;
	data.texAddr = idx;
	data.texelCount = tex.texelCount;
	m_TextureDescriptors.emplace_back(data);

	return idx;
}

#define TOCHAR(a) ((uint)((a)*255.0f))
#define TOUINT4(a, b, c, d) (TOCHAR(a) + (TOCHAR(b) << 8u) + (TOCHAR(c) << 16u) + (TOCHAR(d) << 24u))
std::pair<Material, MaterialTexIds> HostMaterial::convertToDeviceMaterial(MaterialList *list) const
{
	Material gpuMat{};
	MaterialTexIds gpuMatEx{};
	for (int i = 0; i < 11; i++)
		gpuMatEx.texture[i] = map[i].textureID;

	const auto &textures = list->getTextures();

	// base properties
	memset(&gpuMat, 0, sizeof(Material));
	gpuMat.diffuse_r = color.x;
	gpuMat.diffuse_g = color.y;
	gpuMat.diffuse_b = color.z;
	gpuMat.transmittance_r = absorption.x;
	gpuMat.transmittance_g = absorption.y;
	gpuMat.transmittance_b = absorption.z;
	gpuMat.parameters.x = TOUINT4(metallic, subsurface, specular, roughness);
	gpuMat.parameters.y = TOUINT4(specularTint, anisotropic, sheen, sheenTint);
	gpuMat.parameters.z = TOUINT4(clearcoat, clearcoatGloss, transmission, eta * 0.5f);
	gpuMat.parameters.w = TOUINT4(custom0, custom1, custom2, custom3);
	const Texture *t0 = map[TEXTURE0].textureID == -1 ? nullptr : &textures[map[TEXTURE0].textureID];

	const Texture *t1 = map[TEXTURE1].textureID == -1 ? nullptr : &textures[map[TEXTURE1].textureID];

	const Texture *t2 = map[TEXTURE2].textureID == -1 ? nullptr : &textures[map[TEXTURE2].textureID];

	const Texture *nm0 = map[NORMALMAP0].textureID == -1 ? nullptr : &textures[map[NORMALMAP0].textureID];

	const Texture *nm1 = map[NORMALMAP1].textureID == -1 ? nullptr : &textures[map[NORMALMAP1].textureID];

	const Texture *nm2 = map[NORMALMAP2].textureID == -1 ? nullptr : &textures[map[NORMALMAP2].textureID];

	const Texture *r = map[ROUGHNESS0].textureID == -1 ? nullptr : &textures[map[ROUGHNESS0].textureID];

	const Texture *s = map[SPECULARITY].textureID == -1 ? nullptr : &textures[map[SPECULARITY].textureID];

	const Texture *cm = map[COLORMASK].textureID == -1 ? nullptr : &textures[map[COLORMASK].textureID];

	const Texture *am = map[ALPHAMASK].textureID == -1 ? nullptr : &textures[map[ALPHAMASK].textureID];

	bool hdr = false;
	if (t0)
	{
		if (t0->type == Texture::Type::FLOAT4)
			hdr = true;
	}
	gpuMat.flags = (eta > 0 ? (1u << IsDielectric) : 0) +				 // is dielectric
				   (hdr ? (1u << DiffuseMapIsHDR) : 0) +				 // diffuse map is hdr
				   (t0 ? (1u << HasDiffuseMap) : 0) +					 // has diffuse map
				   (nm0 ? (1u << HasNormalMap) : 0) +					 // has normal map
				   (s ? (1u << HasSpecularityMap) : 0) +				 // has specularity map
				   (r ? (1u << HasRoughnessMap) : 0) +					 // has roughness map
				   ((flags & ANISOTROPIC) ? (1u << IsAnisotropic) : 0) + // is anisotropic
				   (nm1 ? (1u << Has2ndNormalMap) : 0) +				 // has 2nd normal map
				   (nm2 ? (1u << Has3rdNormalMap) : 0) +				 // has 3rd normal map
				   (t1 ? (1u << Has2ndNormalMap) : 0) +					 // has 2nd diffuse map
				   (t2 ? (1u << Has3rdDiffuseMap) : 0) +				 // has 3rd diffuse map
				   ((flags & SMOOTH) ? (1u << HasSmoothNormals) : 0) +	 // has smooth normals
				   ((flags & HASALPHA) ? (1u << HasAlpha) : 0);			 // has alpha
	// maps
	if (t0) // texture layer 0
	{
		gpuMat.texwidth0 = t0->width;
		gpuMat.texheight0 = t0->height;
		gpuMat.texaddr0 = map[TEXTURE0].textureID;
		gpuMat.uoffs0 = map[TEXTURE0].uvoffset.x;
		gpuMat.voffs0 = map[TEXTURE0].uvoffset.y;
		gpuMat.uscale0 = map[TEXTURE0].uvscale.x;
		gpuMat.vscale0 = (half)map[TEXTURE0].uvscale.y;
	}
	if (t1) // texture layer 1
	{
		gpuMat.texwidth1 = t1->width;
		gpuMat.texheight1 = t1->height;
		gpuMat.texaddr0 = map[TEXTURE1].textureID;
		gpuMat.uoffs1 = map[TEXTURE1].uvoffset.x;
		gpuMat.voffs1 = map[TEXTURE1].uvoffset.y;
		gpuMat.uscale1 = map[TEXTURE1].uvscale.x;
		gpuMat.vscale1 = (half)map[TEXTURE1].uvscale.y;
	}
	if (t2) // texture layer 2
	{
		gpuMat.texwidth2 = t2->width;
		gpuMat.texheight2 = t2->height;
		gpuMat.texaddr0 = map[TEXTURE2].textureID;
		gpuMat.uoffs2 = map[TEXTURE2].uvoffset.x;
		gpuMat.voffs2 = map[TEXTURE2].uvoffset.y;
		gpuMat.uscale2 = map[TEXTURE2].uvscale.x;
		gpuMat.vscale2 = (half)map[TEXTURE2].uvscale.y;
	}
	if (nm0) // normal map layer 0
	{
		gpuMat.nmapwidth0 = nm0->width;
		gpuMat.nmapheight0 = nm0->height;
		gpuMat.texaddr0 = map[NORMALMAP0].textureID;
		gpuMat.nuoffs0 = map[NORMALMAP0].uvoffset.x;
		gpuMat.nvoffs0 = map[NORMALMAP0].uvoffset.y;
		gpuMat.nuscale0 = map[NORMALMAP0].uvscale.x;
		gpuMat.nvscale0 = map[NORMALMAP0].uvscale.y;
	}
	if (nm1) // normal map layer 1
	{
		gpuMat.nmapwidth1 = nm1->width;
		gpuMat.nmapheight1 = nm1->height;
		gpuMat.texaddr0 = map[NORMALMAP1].textureID;
		gpuMat.nuoffs1 = map[NORMALMAP1].uvoffset.x;
		gpuMat.nvoffs1 = map[NORMALMAP1].uvoffset.y;
		gpuMat.nuscale1 = map[NORMALMAP1].uvscale.x;
		gpuMat.nvscale1 = map[NORMALMAP1].uvscale.y;
	}
	if (nm2) // normal map layer 2
	{
		gpuMat.nmapwidth2 = nm2->width;
		gpuMat.nmapheight2 = nm2->height;
		gpuMat.texaddr0 = map[NORMALMAP2].textureID;
		gpuMat.nuoffs2 = map[NORMALMAP2].uvoffset.x;
		gpuMat.nvoffs2 = map[NORMALMAP2].uvoffset.y;
		gpuMat.nuscale2 = map[NORMALMAP2].uvscale.x;
		gpuMat.nvscale2 = map[NORMALMAP2].uvscale.y;
	}
	if (r) // roughness map
	{
		gpuMat.rmapwidth = r->width;
		gpuMat.rmapheight = r->height;
		gpuMat.texaddr0 = map[ROUGHNESS0].textureID;
		gpuMat.ruoffs = map[ROUGHNESS0].uvoffset.x;
		gpuMat.rvoffs = map[ROUGHNESS0].uvoffset.y;
		gpuMat.ruscale = map[ROUGHNESS0].uvscale.x;
		gpuMat.rvscale = map[ROUGHNESS0].uvscale.y;
	}
	if (s) // specularity map
	{
		gpuMat.smapwidth = s->width;
		gpuMat.smapheight = s->height;
		gpuMat.texaddr0 = map[SPECULARITY].textureID;
		gpuMat.suoffs = map[SPECULARITY].uvoffset.x;
		gpuMat.svoffs = map[SPECULARITY].uvoffset.y;
		gpuMat.suscale = map[SPECULARITY].uvscale.x;
		gpuMat.svscale = map[SPECULARITY].uvscale.y;
	}

	if (cm) // color mask map
	{
		gpuMat.cmapwidth = cm->width;
		gpuMat.cmapheight = cm->height;
		gpuMat.cuoffs = map[COLORMASK].uvoffset.x;
		gpuMat.cvoffs = map[COLORMASK].uvoffset.y;
		gpuMat.cuscale = map[COLORMASK].uvscale.x, gpuMat.cvscale = map[COLORMASK].uvscale.y;
	}

	if (am) // alpha mask map
	{
		gpuMat.amapwidth = am->width;
		gpuMat.amapheight = am->height;
		gpuMat.auoffs = map[ALPHAMASK].uvoffset.x;
		gpuMat.avoffs = map[ALPHAMASK].uvoffset.y;
		gpuMat.auscale = map[ALPHAMASK].uvscale.x, gpuMat.avscale = map[ALPHAMASK].uvscale.y;
	}

	return std::make_pair(gpuMat, gpuMatEx);
}
