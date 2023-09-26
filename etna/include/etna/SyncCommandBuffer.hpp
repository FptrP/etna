#pragma once
#ifndef ETNA_SYNC_COMMAND_BUFFER_INCLUDED
#define ETNA_SYNC_COMMAND_BUFFER_INCLUDED

#include <etna/ResourceTracking.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/ComputePipeline.hpp>

namespace etna
{
struct DescriptorSet;
struct SyncCommandBuffer;

struct CommandBufferPool
{
  CommandBufferPool();
  CommandBufferPool(CommandBufferPool &&) = default;
  ~CommandBufferPool();

  SyncCommandBuffer allocate();

  vk::UniqueCommandBuffer allocatePrimary();
  vk::UniqueCommandBuffer allocateSecondary();

private:
  vk::UniqueCommandPool primaryCmd;
  vk::UniqueCommandPool secondaryCmd;
};


struct SubmitInfo
{
  std::vector<vk::Semaphore> waitSemaphores;
  std::vector<vk::PipelineStageFlags> waitDstStageMask;
  std::vector<vk::Semaphore> signalSemaphores;
};

struct RenderingAttachment
{
  ImageView view;
  vk::ImageLayout layout;
  vk::ResolveModeFlagBits resolveMode {vk::ResolveModeFlagBits::eNone};
  std::optional<ImageView> resolveImageView {std::nullopt};
  vk::ImageLayout resolveLayout {vk::ImageLayout::eUndefined};
  vk::AttachmentLoadOp loadOp {vk::AttachmentLoadOp::eDontCare};
  vk::AttachmentStoreOp storeOp {vk::AttachmentStoreOp::eStore};
  vk::ClearValue clearValue {vk::ClearColorValue{0.f, 0.f, 0.f, 0.f}}; 
};

struct SyncCommandBuffer
{
  SyncCommandBuffer(CommandBufferPool &pool_);

  vk::Result reset();
  vk::Result begin();
  vk::Result end();

  vk::CommandBuffer &get()
  {
    return *cmd;
  }

  vk::CommandBuffer getRenderCmd() const 
  {
    ETNA_ASSERT(currentState == State::Rendering && renderCmd.has_value());
    return renderCmd->get();
  }

  const vk::CommandBuffer &get() const
  {
    return *cmd;
  }

  CmdBufferTrackingState &getTrackingState()
  {
    return trackingState;
  }

  const CmdBufferTrackingState &getTrackingState() const
  {
    return trackingState;
  }

  void copyBuffer(const Buffer &src, const Buffer &dst,
    const vk::ArrayProxy<vk::BufferCopy> &regions);

  void fillBuffer(const Buffer &dst, vk::DeviceSize offset, vk::DeviceSize size, uint32_t data);

  void blitImage(const Image &src,
    vk::ImageLayout srcLayout,
    const Image &dst, vk::ImageLayout dstLayout,
    const vk::ArrayProxy<vk::ImageBlit> regions,
    vk::Filter filter);

  void clearColorImage(const Image &image, vk::ImageLayout layout, 
    vk::ClearColorValue clear_color, vk::ArrayProxy<vk::ImageSubresourceRange> ranges);

  void copyBufferToImage(const Buffer &src, const Image &dst, vk::ImageLayout dstLayout,
    const vk::ArrayProxy<vk::BufferImageCopy> &regions);

  void transformLayout(const Image &image, vk::ImageLayout layout, vk::ImageSubresourceRange range);

  void bindDescriptorSet(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout, 
    uint32_t set_index, const DescriptorSet &set, std::span<const uint32_t> dynamic_offsets = {});

  void bindPipeline(vk::PipelineBindPoint bind_point, const PipelineBase &pipeline);  
  void dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
  void pushConstants(ShaderProgramId program, uint32_t offset, uint32_t size, const void *data);  

  template <typename T>
  void pushConstants(ShaderProgramId program, uint32_t offset, const T &data)
  {
    pushConstants(program, offset, sizeof(T), &data);
  }

  void beginRendering(vk::Rect2D area,
    vk::ArrayProxy<const RenderingAttachment> color_attachments,
    std::optional<RenderingAttachment> depth_attachment = {},
    std::optional<RenderingAttachment> stencil_attachment = {});
  
  void endRendering();

  void bindVertexBuffer(uint32_t binding_index, const Buffer &buffer, vk::DeviceSize offset);
  void bindIndexBuffer(const Buffer &buffer, uint32_t offset, vk::IndexType type);
  void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_index);
  void drawIndexed(uint32_t index_cout, uint32_t instance_count, 
    uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance);

  void setViewport(uint32_t first_viewport, vk::ArrayProxy<const vk::Viewport> viewports);
  void setScissor(uint32_t first_scissor, vk::ArrayProxy<const vk::Rect2D> scissors);

  void bindPipeline(const ComputePipeline &pipeline)
  {
    bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  }
  
  void bindPipeline(const GraphicsPipeline &pipeline)
  {
    bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
  }

  void expectState(const Buffer &buffer, BufferState state);
  void expectState(const Image &image, uint32_t mip, uint32_t layer, ImageSubresState state);
  void expectState(const Image &image, vk::ImageSubresourceRange range, ImageSubresState state);
  void expectState(const Image &image, ImageSubresState state);

  vk::Result submit(vk::Fence signalFence = {})
  { 
    return submit(nullptr, signalFence);
  }
  vk::Result submit(const SubmitInfo &info, vk::Fence signalFence = {})
  { 
    return submit(&info, signalFence);
  }

private:

  vk::Result submit(const SubmitInfo *info, vk::Fence signalFence);

  void flushBarrier()
  {
    trackingState.flushBarrier(barrier);
    barrier.flush(*cmd);
  }

  // TODO - add state validation
  enum class State // https://registry.khronos.org/vulkan/site/spec/latest/chapters/cmdbuffers.html
  {
    Initial, // -> begin or free
    Recording, // -> acquire resources, record commands 
    Executable, // end()
    Rendering, // recording draws
    Pending // after submit
    // Invalid - TODO
  };

  struct RenderInfo
  {
    vk::Rect2D renderArea{};
    std::vector<vk::RenderingAttachmentInfo> colorAttachments;
    std::optional<vk::RenderingAttachmentInfo> depthAttachment;
    std::optional<vk::RenderingAttachmentInfo> stencilAttachment;

    RenderInfo(){}
    RenderInfo(RenderInfo &&) = default;
    RenderInfo &operator=(RenderInfo &&) = default;
  };
  CommandBufferPool &pool;
  CmdBufferTrackingState trackingState;
  tracking::CmdBarrier barrier;
  vk::UniqueCommandBuffer cmd;

  State currentState = State::Initial;
  
  std::optional<RenderInfo> renderState {};
  std::optional<vk::UniqueCommandBuffer> renderCmd {};
  
  std::vector<vk::UniqueCommandBuffer> usedRenderCmd;
};


}

#endif