//
// Created by meir on 10/5/19.
//

#include "RenderPass.h"
uint32_t vkc::RenderPass::addColorAttachment(vk::Format format, vk::AttachmentLoadOp loadOp,
											 vk::AttachmentStoreOp storeOp, vk::AttachmentLoadOp stencilLoadOp,
											 vk::AttachmentStoreOp stencilStoreOp, vk::SampleCountFlagBits samples,
											 vk::ImageLayout initialLayout, vk::ImageLayout finalLayout)
{
	vk::AttachmentDescription description{};
	description.format = format;
	description.loadOp = loadOp;
	description.storeOp = storeOp;
	description.stencilLoadOp = stencilLoadOp;
	description.stencilStoreOp = stencilStoreOp;
	description.samples = samples;
	description.initialLayout = initialLayout;
	description.finalLayout = finalLayout;

	const auto idx = static_cast<uint32_t>(m_AttachmentDescriptions.size());
	m_AttachmentDescriptions.emplace_back(description);
	return idx;
}

void vkc::RenderPass::addSubpass(const vkc::Subpass &pass)
{
	const auto idx = m_Subpasses.size();
	m_Subpasses.push_back(pass);
	vk::SubpassDescription description{};
	description.pipelineBindPoint = pass.bindPoint;
	description.colorAttachmentCount = static_cast<uint32_t>(m_Subpasses.at(idx).attachments.size());
	description.pColorAttachments = m_Subpasses.at(idx).attachments.data();

	m_SubpassDescriptions.push_back(description);
}

void vkc::RenderPass::finalize()
{
	vk::RenderPassCreateInfo createInfo{};
	createInfo.attachmentCount = static_cast<uint32_t>(m_AttachmentDescriptions.size());
	createInfo.pAttachments = m_AttachmentDescriptions.data();
	createInfo.subpassCount = static_cast<uint32_t>(m_Subpasses.size());
	createInfo.pSubpasses = m_SubpassDescriptions.data();

	m_RenderPass = m_Device->createRenderPass(createInfo);
	m_Finalized = true;
}

void vkc::RenderPass::cleanup()
{
	if (m_RenderPass)
		m_Device->destroyRenderPass(m_RenderPass);
	m_RenderPass = nullptr;
	m_Finalized = false;
}
