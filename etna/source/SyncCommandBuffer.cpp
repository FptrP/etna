#include "etna/SyncCommandBuffer.hpp"
#include "etna/Etna.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/GlobalContext.hpp"

namespace etna 
{


CommandBufferPool::CommandBufferPool()
{
  auto device = etna::get_context().getDevice();
  vk::CommandPoolCreateInfo info {
    .queueFamilyIndex = etna::get_context().getQueueFamilyIdx()
  };
  
  info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

  primaryCmd = device.createCommandPoolUnique(info).value;

  info.flags = vk::CommandPoolCreateFlags{};

  secondaryCmd = device.createCommandPoolUnique(info).value;
}

CommandBufferPool::~CommandBufferPool() {}


vk::UniqueCommandBuffer CommandBufferPool::allocatePrimary()
{
  vk::CommandBufferAllocateInfo info {
    .commandPool = primaryCmd.get(),
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = 1
  };

  auto cmd = std::move(etna::get_context().getDevice().allocateCommandBuffersUnique(info).value[0]);
  return cmd;
}

vk::UniqueCommandBuffer CommandBufferPool::allocateSecondary()
{
  vk::CommandBufferAllocateInfo info {
    .commandPool = secondaryCmd.get(),
    .level = vk::CommandBufferLevel::eSecondary,
    .commandBufferCount = 1
  };

  auto cmd = std::move(etna::get_context().getDevice().allocateCommandBuffersUnique(info).value[0]);
  return cmd;
}

SyncCommandBuffer CommandBufferPool::allocate()
{
  return {*this};
}

SyncCommandBuffer::SyncCommandBuffer(CommandBufferPool &pool_)
  : pool{pool_}, cmd {pool.allocatePrimary()}
{}

void SyncCommandBuffer::expectState(const Buffer &buffer, BufferState state)
{
  ETNA_ASSERT(currentState == State::Recording || currentState == State::Rendering);
  trackingState.expectState(buffer, state);
}
void SyncCommandBuffer::expectState(const Image &image, uint32_t mip, uint32_t layer, ImageSubresState state)
{
  ETNA_ASSERT(currentState == State::Recording || currentState == State::Rendering);
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
  currentState = State::Initial; 
  usedRenderCmd.clear();
  return cmd->reset();
}

vk::Result SyncCommandBuffer::begin()
{
  ETNA_ASSERT(currentState == State::Initial);
  currentState = State::Recording;
  etna::get_context().getQueueTrackingState() //maybe not the best place. Add bufferState
    .setExpectedStates(trackingState);
  return cmd->begin(vk::CommandBufferBeginInfo{});
}

vk::Result SyncCommandBuffer::end()
{
  ETNA_ASSERT(currentState == State::Recording);
  currentState = State::Executable;
  return cmd->end();
}

void SyncCommandBuffer::copyBuffer(const Buffer &src, const Buffer &dst,
  const vk::ArrayProxy<vk::BufferCopy> &regions)
{
  ETNA_ASSERT(currentState == State::Recording);
  
  trackingState.requestState(src, BufferState {
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferRead
  }); 

  trackingState.requestState(dst, BufferState {
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite
  }); 

  flushBarrier();
  cmd->copyBuffer(src.get(), dst.get(), regions);
}

void SyncCommandBuffer::blitImage(const Image &src,
  vk::ImageLayout srcLayout,
  const Image &dst, vk::ImageLayout dstLayout,
  const vk::ArrayProxy<vk::ImageBlit> regions,
  vk::Filter filter)
{
  ETNA_ASSERT(currentState == State::Recording);

  for (const auto &region : regions)
  {
    vk::ImageSubresourceRange srcRange {
      .baseMipLevel = region.srcSubresource.mipLevel,
      .levelCount = 1,
      .baseArrayLayer = region.srcSubresource.baseArrayLayer,
      .layerCount = region.srcSubresource.layerCount
    };

    trackingState.requestState(src, srcRange, ImageSubresState {
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferRead,
      srcLayout
    });

    vk::ImageSubresourceRange dstRange {
      .baseMipLevel = region.dstSubresource.mipLevel,
      .levelCount = 1,
      .baseArrayLayer = region.dstSubresource.baseArrayLayer,
      .layerCount = region.dstSubresource.layerCount
    };

    trackingState.requestState(dst, dstRange, ImageSubresState {
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferWrite,
      dstLayout
    });
  }

  flushBarrier();
  cmd->blitImage(src.get(), srcLayout, dst.get(), dstLayout, regions, filter);
}

void SyncCommandBuffer::clearColorImage(const Image &image, vk::ImageLayout layout, 
  vk::ClearColorValue clear_color, vk::ArrayProxy<vk::ImageSubresourceRange> ranges)
{
  ETNA_ASSERT(currentState == State::Recording);
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
  ETNA_ASSERT(currentState == State::Recording);
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
  ETNA_ASSERT(currentState == State::Recording);
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

  if (bind_point == vk::PipelineBindPoint::eGraphics)
  {
    ETNA_ASSERT(currentState == State::Rendering);
    renderCmd.value()->bindDescriptorSets(bind_point, layout, set_index, {set.getVkSet()}, dynamic_offsets);
    return;
  }

  ETNA_ASSERT(currentState == State::Recording);
  cmd->bindDescriptorSets(bind_point, layout, set_index, {set.getVkSet()}, dynamic_offsets);
}

void SyncCommandBuffer::bindPipeline(vk::PipelineBindPoint bind_point, const PipelineBase &pipeline)
{
  if (bind_point == vk::PipelineBindPoint::eGraphics){
    ETNA_ASSERT(currentState == State::Rendering);
    renderCmd.value()->bindPipeline(bind_point, pipeline.getVkPipeline());
    return;
  }
  ETNA_ASSERT(currentState == State::Recording);
  cmd->bindPipeline(bind_point, pipeline.getVkPipeline());
}

void SyncCommandBuffer::dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
  ETNA_ASSERT(currentState == State::Recording);
  flushBarrier();
  cmd->dispatch(groups_x, groups_y, groups_z);
}

void SyncCommandBuffer::pushConstants(ShaderProgramId program, uint32_t offset, uint32_t size, const void *data)
{
  auto info = etna::get_shader_program(program);
  auto constInfo = info.getPushConst();
  
  ETNA_ASSERTF(constInfo.size > 0, "Shader program {} doesn't have push constants", program);
  ETNA_ASSERTF(offset + size <= constInfo.size, "pushConstants: out of range");

  if (currentState == State::Rendering)
    renderCmd.value()->pushConstants(info.getPipelineLayout(), constInfo.stageFlags, offset, size, data);
  else
  {
    ETNA_ASSERT(currentState == State::Recording);
    cmd->pushConstants(info.getPipelineLayout(), constInfo.stageFlags, offset, size, data);
  }
}

void SyncCommandBuffer::beginRendering(vk::Rect2D area,
    vk::ArrayProxy<const RenderingAttachment> color_attachments,
    std::optional<RenderingAttachment> depth_attachment,
    std::optional<RenderingAttachment> stencil_attachment)
{
  ETNA_ASSERT(currentState == State::Recording);

  std::vector<vk::RenderingAttachmentInfo> colorInfos;
  std::vector<vk::Format> colorFmt;

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

    colorFmt.push_back(image.getInfo().format);
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

  vk::Format depthFormat {vk::Format::eUndefined};
  vk::Format stencilFormat {vk::Format::eUndefined};
  std::optional<vk::RenderingAttachmentInfo> depthAttachment;
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

    depthFormat = image.getInfo().format;
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

  renderState.emplace(RenderInfo{});
  renderState->renderArea = area;
  renderState->colorAttachments = colorInfos;

  if (depthAttachment.has_value())
    renderState->depthAttachment.emplace(*depthAttachment);

  renderCmd.emplace(pool.allocateSecondary());
  
  vk::CommandBufferInheritanceRenderingInfo secondaryInfo {
    .colorAttachmentCount = colorFmt.size(),
    .pColorAttachmentFormats = colorFmt.data(),
    .depthAttachmentFormat = depthFormat,
    .stencilAttachmentFormat = stencilFormat
  };
  
  vk::CommandBufferInheritanceInfo inheritanceInfo {
    .pNext = &secondaryInfo
  };

  vk::CommandBufferBeginInfo beginInfo {
    .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
    .pInheritanceInfo = &inheritanceInfo
  };

  auto res = renderCmd.value()->begin(beginInfo);
  ETNA_ASSERT(res == vk::Result::eSuccess);
  currentState = State::Rendering;
}
  
void SyncCommandBuffer::endRendering()
{
  ETNA_ASSERT(currentState == State::Rendering);
  ETNA_ASSERT(renderState.has_value() && renderCmd.has_value());

  renderCmd.value()->end();

  flushBarrier();

  vk::RenderingInfo vkRenderInfo {
    .flags = vk::RenderingFlagBits::eContentsSecondaryCommandBuffers,
    .renderArea = renderState->renderArea,
    .layerCount = 1,
    .colorAttachmentCount = renderState->colorAttachments.size(),
    .pColorAttachments = renderState->colorAttachments.data(),
    .pDepthAttachment = renderState->depthAttachment.has_value()? &renderState->depthAttachment.value() : nullptr    
  };

  cmd->beginRendering(vkRenderInfo);
  cmd->executeCommands({renderCmd.value().get()});
  cmd->endRendering();

  currentState = State::Recording;
  usedRenderCmd.emplace_back(std::move(renderCmd).value());
}

void SyncCommandBuffer::bindVertexBuffer(uint32_t binding_index, const Buffer &buffer, vk::DeviceSize offset)
{
  ETNA_ASSERT(currentState == State::Rendering);
  trackingState.requestState(buffer, BufferState {
    vk::PipelineStageFlagBits2::eVertexInput,
    vk::AccessFlagBits2::eVertexAttributeRead
  });

  renderCmd.value()->bindVertexBuffers(binding_index, {buffer.get()}, {offset});
}

void SyncCommandBuffer::bindIndexBuffer(const Buffer &buffer, uint32_t offset, vk::IndexType type)
{
  ETNA_ASSERT(currentState == State::Rendering);
  trackingState.requestState(buffer, BufferState {
    vk::PipelineStageFlagBits2::eIndexInput,
    vk::AccessFlagBits2::eIndexRead
  });

  renderCmd.value()->bindIndexBuffer(buffer.get(), offset, type);
}

void SyncCommandBuffer::draw(uint32_t vertex_count, uint32_t instance_count, 
  uint32_t first_vertex, uint32_t first_index)
{
  ETNA_ASSERT(currentState == State::Rendering);
  renderCmd.value()->draw(vertex_count, instance_count, first_vertex, first_index);
}

void SyncCommandBuffer::drawIndexed(uint32_t index_cout, uint32_t instance_count, 
    uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance)
{
  ETNA_ASSERT(currentState == State::Rendering);
  renderCmd.value()->drawIndexed(index_cout, instance_count, first_index, vertex_offset, first_instance);
}

void SyncCommandBuffer::setViewport(uint32_t first_viewport, vk::ArrayProxy<const vk::Viewport> viewports)
{
  ETNA_ASSERT(currentState == State::Rendering);
  renderCmd.value()->setViewport(first_viewport, viewports);
}
void SyncCommandBuffer::setScissor(uint32_t first_scissor, vk::ArrayProxy<const vk::Rect2D> scissors)
{
  ETNA_ASSERT(currentState == State::Rendering);
  renderCmd.value()->setScissor(first_scissor, scissors);
}

vk::Result SyncCommandBuffer::submit(const SubmitInfo *info, vk::Fence signalFence)
{
  ETNA_ASSERT(currentState = State::Executable);
  currentState = State::Pending;
  
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