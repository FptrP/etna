#include "etna/SyncCommandBuffer.hpp"
#include "etna/Etna.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/GlobalContext.hpp"

namespace etna 
{
void SyncCommandBuffer::expectState(const Buffer &buffer, BufferState state)
{
  trackingState.expectState(buffer, state);
}
void SyncCommandBuffer::expectState(const Image &image, uint32_t mip, uint32_t layer, ImageSubresState state)
{
  trackingState.expectState(image, mip, layer, state);
}

void SyncCommandBuffer::expectState(const Image &image, vk::ImageSubresourceRange range, 
  ImageSubresState state)
{
  for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; mip++)
  {
    for (uint32_t layer = range.baseArrayLayer; layer < range.layerCount + range.baseArrayLayer; layer++)
    {
      trackingState.expectState(image, mip, layer, state);
    }
  }
}

void SyncCommandBuffer::expectState(const Image &image, ImageSubresState state)
{
  vk::ImageSubresourceRange range {};
  range.levelCount = image.getInfo().mipLevels;
  range.layerCount = image.getInfo().arrayLayers;
  expectState(image, range, state);
}

vk::Result SyncCommandBuffer::reset()
{
  return cmd->reset();
}

vk::Result SyncCommandBuffer::begin()
{
  etna::get_context().getQueueTrackingState() //maybe not the best place. Add bufferState
    .setExpectedStates(trackingState);
  return cmd->begin(vk::CommandBufferBeginInfo{});
}

vk::Result SyncCommandBuffer::end()
{
  return cmd->end();
}

void SyncCommandBuffer::clearColorImage(const Image &image, vk::ImageLayout layout, 
  vk::ClearColorValue clear_color, vk::ArrayProxy<vk::ImageSubresourceRange> ranges)
{
  ImageSubresState state {
    .activeStages = vk::PipelineStageFlagBits2::eTransfer,
    .activeAccesses = vk::AccessFlagBits2::eTransferWrite,
    .layout = layout
  };

  for (auto &range : ranges)
  {
    trackingState.requestState(image, range.baseMipLevel, range.levelCount, 
      range.baseArrayLayer, range.layerCount, state);
  }

  flushBarrier();
  cmd->clearColorImage(image.get(), layout, clear_color, ranges);
}


void SyncCommandBuffer::copyBufferToImage(const Buffer &src, const Image &dst, vk::ImageLayout dstLayout,
  const vk::ArrayProxy<vk::BufferImageCopy> &regions)
{
  trackingState.requestState(src, BufferState {
    .activeStages = vk::PipelineStageFlagBits2::eTransfer,
    .activeAccesses = vk::AccessFlagBits2::eTransferRead
  });

  for (auto &region : regions)
  {
    vk::ImageSubresourceRange range {
      .baseMipLevel = region.imageSubresource.mipLevel,
      .levelCount = 1,
      .baseArrayLayer = region.imageSubresource.baseArrayLayer,
      .layerCount = region.imageSubresource.layerCount
    };

    trackingState.requestState(dst, range, ImageSubresState {
      .activeStages = vk::PipelineStageFlagBits2::eTransfer,
      .activeAccesses = vk::AccessFlagBits2::eTransferWrite,
      .layout = dstLayout
    });
  }

  flushBarrier();

  cmd->copyBufferToImage(src.get(), dst.get(), dstLayout, regions);
}

void SyncCommandBuffer::transformLayout(const Image &image, vk::ImageLayout layout, 
  vk::ImageSubresourceRange range)
{
  ImageSubresState state {
    .activeStages = vk::PipelineStageFlags2{},
    .activeAccesses = vk::AccessFlags2{},
    .layout = layout
  };

  trackingState.requestState(image, range.baseMipLevel, range.levelCount, 
    range.baseArrayLayer, range.layerCount, state);
  
  flushBarrier();
}

void SyncCommandBuffer::bindDescriptorSet(vk::PipelineBindPoint bind_point, 
    vk::PipelineLayout layout, uint32_t set_index, 
    const DescriptorSet &set, std::span<const uint32_t> dynamic_offsets)
{
  set.requestStates(trackingState);
  cmd->bindDescriptorSets(bind_point, layout, set_index, {set.getVkSet()}, dynamic_offsets);
}

void SyncCommandBuffer::bindPipeline(vk::PipelineBindPoint bind_point, const PipelineBase &pipeline)
{
  cmd->bindPipeline(bind_point, pipeline.getVkPipeline());
}

void SyncCommandBuffer::dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
  flushBarrier();
  cmd->dispatch(groups_x, groups_y, groups_z);
}

void SyncCommandBuffer::pushConstants(ShaderProgramId program, uint32_t offset, uint32_t size, const void *data)
{
  auto info = etna::get_shader_program(program);
  auto constInfo = info.getPushConst();
  
  ETNA_ASSERTF(constInfo.size > 0, "Shader program {} doesn't have push constants", program);
  ETNA_ASSERTF(offset + size < constInfo.size, "pushConstants: out of range");

  cmd->pushConstants(info.getPipelineLayout(), constInfo.stageFlags, offset, size, data);
}

void SyncCommandBuffer::beginRendering(vk::Rect2D area,
    vk::ArrayProxy<const RenderingAttachment> color_attachments,
    std::optional<RenderingAttachment> depth_attachment,
    std::optional<RenderingAttachment> stencil_attachment)
{
  std::vector<vk::RenderingAttachmentInfo> colorInfos;

  for (auto &colorAttachment : color_attachments)
  {
    ETNA_ASSERTF(colorAttachment.resolveMode == vk::ResolveModeFlagBits::eNone, \
      "MSAA resolve not supported");

    auto &image = colorAttachment.view.getOwner();
    auto range = colorAttachment.view.getRange();

    trackingState.requestState(
      image, 
      range,
      ImageSubresState {
        .activeStages = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .activeAccesses = vk::AccessFlagBits2::eColorAttachmentWrite,
        .layout = colorAttachment.layout
      }); 

    vk::RenderingAttachmentInfo info {
      .imageView = vk::ImageView(colorAttachment.view),
      .imageLayout = colorAttachment.layout,
      .resolveMode = vk::ResolveModeFlagBits::eNone,
      .loadOp = colorAttachment.loadOp,
      .storeOp = colorAttachment.storeOp,
      .clearValue = colorAttachment.clearValue
    };

    colorInfos.push_back(info);
  }

  std::optional<vk::RenderingAttachmentInfo> depthAttachment;
  std::optional<vk::RenderingAttachmentInfo> stencilAttachment;

  // tricky part. currently barriers don't support separate aspect mask, so 
  // we need to make layout transitions both for depth and stencil
  // For now depth and stencil should point to the same image if used
  if (depth_attachment && stencil_attachment) 
  {
    ETNA_ASSERT(depth_attachment->view.getOwner().get() == stencil_attachment->view.getOwner().get());
    ETNA_ASSERTF(false, "Stencil not supported yet :(");
  }
  else if (depth_attachment)
  {
    auto layout = depth_attachment->layout;
    bool readOnly = layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    auto &image = depth_attachment->view.getOwner();
    auto range = depth_attachment->view.getRange();
    
    auto stages = vk::PipelineStageFlagBits2::eEarlyFragmentTests
      | vk::PipelineStageFlagBits2::eLateFragmentTests;

    vk::AccessFlags2 access = vk::AccessFlagBits2::eDepthStencilAttachmentRead;
    if (!readOnly)
      access |= vk::AccessFlagBits2::eDepthStencilAttachmentWrite;

    trackingState.requestState(image, range, ImageSubresState{stages, access, layout});

    depthAttachment = vk::RenderingAttachmentInfo {
      .imageView = vk::ImageView(depth_attachment->view),
      .imageLayout = layout,
      .resolveMode = vk::ResolveModeFlagBits::eNone,
      .loadOp = depth_attachment->loadOp,
      .storeOp = depth_attachment->storeOp,
      .clearValue = depth_attachment->clearValue
    };
  }
  else if (stencil_attachment)
  {
    ETNA_ASSERTF(false, "Stencil not supported yet :(");
  }

  vk::RenderingInfo renderInfo {
    .renderArea = area,
    .layerCount = 1,
    .viewMask = 0,
    .pDepthAttachment = depthAttachment.has_value()? &*depthAttachment : nullptr,
    .pStencilAttachment = stencilAttachment.has_value()? &*stencilAttachment : nullptr
  };

  renderInfo.setColorAttachments(colorInfos);

  flushBarrier();
  cmd->beginRendering(renderInfo);
}
  
void SyncCommandBuffer::endRendering()
{
  cmd->endRendering();
}

void SyncCommandBuffer::bindVertexBuffer(uint32_t binding_index, const Buffer &buffer, vk::DeviceSize offset)
{
  trackingState.requestState(buffer, BufferState {
    vk::PipelineStageFlagBits2::eVertexInput,
    vk::AccessFlagBits2::eVertexAttributeRead
  });

  cmd->bindVertexBuffers(binding_index, {buffer.get()}, {offset});
}

void SyncCommandBuffer::bindIndexBuffer(const Buffer &buffer, uint32_t offset, vk::IndexType type)
{
  trackingState.requestState(buffer, BufferState {
    vk::PipelineStageFlagBits2::eIndexInput,
    vk::AccessFlagBits2::eIndexRead
  });
  //TODO: check sizes 
  cmd->bindIndexBuffer(buffer.get(), offset, type);
}

void SyncCommandBuffer::draw(uint32_t vertex_count, uint32_t instance_count, 
  uint32_t first_vertex, uint32_t first_index)
{
  flushBarrier();
  cmd->draw(vertex_count, instance_count, first_vertex, first_index);
}

void SyncCommandBuffer::drawIndexed(uint32_t index_cout, uint32_t instance_count, 
    uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance)
{
  flushBarrier();
  cmd->drawIndexed(index_cout, instance_count, first_index, vertex_offset, first_instance);
}

void SyncCommandBuffer::setViewport(uint32_t first_viewport, vk::ArrayProxy<const vk::Viewport> viewports)
{
  cmd->setViewport(first_viewport, viewports);
}
void SyncCommandBuffer::setScissor(uint32_t first_scissor, vk::ArrayProxy<const vk::Rect2D> scissors)
{
  cmd->setScissor(first_scissor, scissors);
}

vk::Result SyncCommandBuffer::submit(const SubmitInfo *info, vk::Fence signalFence)
{
  etna::get_context().getQueueTrackingState() // handle error
    .onSubmit(trackingState);

  vk::SubmitInfo submitInfo {
    .commandBufferCount = 1,
    .pCommandBuffers = &cmd.get()
  };

  if (info)
  {
    submitInfo.setWaitSemaphores(info->waitSemaphores);
    submitInfo.setWaitDstStageMask(info->waitDstStageMask);
    submitInfo.setSignalSemaphores(info->signalSemaphores);
  } 

  return etna::get_context().getQueue().submit({submitInfo}, signalFence);
}

} // namespace etna