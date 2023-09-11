#pragma once
#ifndef ETNA_SYNC_COMMAND_BUFFER_INCLUDED
#define ETNA_SYNC_COMMAND_BUFFER_INCLUDED

#include <etna/Image.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/ComputePipeline.hpp>

namespace etna
{

struct CmdBarrier
{
  std::optional<vk::MemoryBarrier2> memoryBarrier;
  std::vector<vk::ImageMemoryBarrier2> imageBarriers;

  void flush(vk::CommandBuffer cmd);
  void clear()
  {
    memoryBarrier = std::nullopt;
    imageBarriers.clear();
  }
};

struct CmdBufferTrackingState
{
  struct ImageState
  {
    struct SubresourceState
    {
      vk::PipelineStageFlags2 activeStages {};
      vk::AccessFlags2 activeAccesses {};
      vk::ImageLayout layout {vk::ImageLayout::eUndefined};

      bool operator==(const SubresourceState &) const = default;
    };

    ImageState(const Image &image)
      : resource {image.get()}, aspect {image.getAspectMaskByFormat()}, 
        mipLevels {image.getInfo().mipLevels},
        arrayLayers{image.getInfo().arrayLayers}
    {
      states.resize(mipLevels * arrayLayers, {});
    }

    ImageState(vk::Image img_, vk::ImageAspectFlags aspect_, uint32_t mips_, uint32_t layers_)
      : resource {img_}, aspect{aspect_}, mipLevels{mips_}, arrayLayers{layers_}
    {
      states.resize(mipLevels * arrayLayers, {});
    }

    std::optional<SubresourceState> &getSubresource(uint32_t mip, uint32_t layer)
    {
      uint32_t index = layer * mipLevels + mip; 
      ETNA_ASSERT(index < mipLevels * arrayLayers);
      return states.at(index);
    }

    vk::Image resource {};
    vk::ImageAspectFlags aspect{};
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    std::vector<std::optional<SubresourceState>> states; //mips x layers
  };

  struct BufferState // generates only memory barriers
  {
    vk::PipelineStageFlags2 activeStages {};
    vk::AccessFlags2 activeAccesses {};

    bool operator==(const BufferState &) const = default;
  };

  using HandleT = uint64_t; // TODO: we definetly get a situation, when resource is deleted
  // but it's Handle is still in TrackingState. Any new resource can get the same handle.
  // The best way is to add 64-bit  id to each resource that are always increasing 
  using ResContainer = std::unordered_map<HandleT, std::variant<ImageState, BufferState>>;

  CmdBufferTrackingState() {}



  //Sets resource state. 
  void expectState(const Image &image, uint32_t mip, uint32_t layer, ImageState::SubresourceState state);
  void expectState(const Buffer &buffer, BufferState state);
  
  void initResourceStates(const ResContainer &states);
  void initResourceStates(ResContainer &&states);

  //requests transition to new state
  void requestState(const Image &image, uint32_t mip, uint32_t layer, ImageState::SubresourceState state);
  void requestState(const Image &image, uint32_t firstMip, uint32_t mipCount, 
    uint32_t firstLayer, uint32_t layerCount, ImageState::SubresourceState state);
  // for now range.aspectMask is ignored
  void requestState(const Image &image, vk::ImageSubresourceRange range, ImageState::SubresourceState state);

  void requestState(const Buffer &buffer, BufferState state);

  void flushBarrier(CmdBarrier &barrier);

  void onSync(); //sets all activeStages and accesses to zero, saves image layouts
  void removeUnusedResources();

  ResContainer takeStates()
  {
    ResContainer out {};
    std::swap(resources, out);
    return out;
  }

  const ResContainer &getStates() const
  {
    return resources;
  }

  const ResContainer &getExpectedStates() const
  {
    return expectedResources;
  }

  void clearExpectedStates()
  {
    expectedResources.clear();
  }

  void clearAll()
  {
    expectedResources.clear();
    resources.clear();
    requests.clear();
  }

private:
  BufferState &acquireResource(HandleT handle);
  ImageState::SubresourceState &acquireResource(HandleT handle, 
    const ImageState &request_state, uint32_t mip, uint32_t layer);


  static std::optional<vk::ImageMemoryBarrier2> genBarrier(vk::Image img,
    vk::ImageAspectFlags aspect,
    uint32_t mip,
    uint32_t layer,
    ImageState::SubresourceState &src,
    const ImageState::SubresourceState &dst);

  static void genBarrier(std::optional<vk::MemoryBarrier2> &barrier,
    BufferState &src,
    const BufferState &dst);

  //ResContainer expectedState;
  //<Something> acquiredState
  ResContainer expectedResources; //for validation on submit
  ResContainer resources; //TODO: store expected states in separate ResContainer
  ResContainer requests;
};

struct QueueTrackingState
{
  void onWait(); //clears all activeStages/activeAccesses
  void onSubmit(CmdBufferTrackingState &state); //validates expected resources, updates currentState
  
  //TODO: 
  bool isResourceUsed(const Buffer &buffer) const;
  bool isResourceUsed(const Image &image, uint32_t mip, uint32_t layer) const;
  
  void setExpectedStates(CmdBufferTrackingState &state) const //reads current states from queue
  {
    state.initResourceStates(currentStates);
  }

private:
  CmdBufferTrackingState::ResContainer currentStates;
};

using ImageSubresState = CmdBufferTrackingState::ImageState::SubresourceState;

struct DescriptorSet;

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
  SyncCommandBuffer(vk::UniqueCommandBuffer &&cmd_) : cmd{std::move(cmd_)} {}

  vk::Result reset();
  vk::Result begin();
  vk::Result end();

  vk::CommandBuffer &get()
  {
    return *cmd;
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


  void clearColorImage(const Image &image, vk::ImageLayout layout, 
    vk::ClearColorValue clear_color, vk::ArrayProxy<vk::ImageSubresourceRange> ranges);

  void transformLayout(const Image &image, vk::ImageLayout layout, vk::ImageSubresourceRange range);

  void bindDescriptorSet(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout, 
    uint32_t set_index, const DescriptorSet &set, std::span<const uint32_t> dynamic_offsets = {});

  void bindPipeline(vk::PipelineBindPoint bind_point, const PipelineBase &pipeline);  
  void dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
  void pushConstants(ShaderProgramId program, uint32_t offset, uint32_t size, const void *data);  

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

  CmdBufferTrackingState::ResContainer takeResourceStates()
  {
    return trackingState.takeStates();
  }

  const CmdBufferTrackingState::ResContainer &getResourceStates() const
  {
    return trackingState.getStates();
  }

  void initResourceStates(const CmdBufferTrackingState::ResContainer &resources)
  {
    trackingState.initResourceStates(resources);
  }

  void initResourceStates(CmdBufferTrackingState::ResContainer &&resources)
  {
    trackingState.initResourceStates(std::move(resources));
  }

  void onSync()
  {
    trackingState.onSync();
  }

  void expectState(const Buffer &buffer, CmdBufferTrackingState::BufferState state);
  void expectState(const Image &image, uint32_t mip, uint32_t layer, ImageSubresState state);
  void expectState(const Image &image, vk::ImageSubresourceRange range, ImageSubresState state);
  void expectState(const Image &image, ImageSubresState state);
  
private:

  void flushBarrier()
  {
    trackingState.flushBarrier(barrier);
    barrier.flush(*cmd);
  }

  CmdBufferTrackingState trackingState;
  CmdBarrier barrier;
  vk::UniqueCommandBuffer cmd;

};


}

#endif