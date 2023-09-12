#pragma once
#ifndef ETNA_RESOURCE_TRACKING_HPP_INCLUDED
#define ETNA_RESOURCE_TRACKING_HPP_INCLUDED

#include <etna/Image.hpp>
#include <etna/Buffer.hpp>

namespace etna::tracking
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

using ImageSubresState = ImageState::SubresourceState;

struct BufferState // generates only memory barriers
{
  vk::PipelineStageFlags2 activeStages {};
  vk::AccessFlags2 activeAccesses {};

  bool operator==(const BufferState &) const = default;
};

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

using HandleT = uint64_t; 

inline HandleT to_handle(const Image &image)
{
  return reinterpret_cast<HandleT>(VkImage(image.get()));
}

inline HandleT to_handle(const Buffer &buffer)
{
  return reinterpret_cast<HandleT>(VkBuffer(buffer.get()));
}


// TODO: we definetly get a situation, when resource is deleted
// but it's Handle is still in TrackingState. Any new resource can get the same handle.
// The best way is to add 64-bit  id to each resource that are always increasing 

using ResContainer = std::unordered_map<HandleT, std::variant<ImageState, BufferState>>;

struct CmdBufferTrackingState
{
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
  void removeUnusedResources(); // removes expectedResources that were not used

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
  ImageSubresState &acquireResource(HandleT handle, 
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

  ResContainer expectedResources; //for validation on submit
  ResContainer resources;
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

  void onResourceDeletion(HandleT handle)
  {
    currentStates.erase(handle);
  }

private:
  ResContainer currentStates;
};

}

namespace etna
{
  using BufferState = tracking::BufferState;
  using ImageSubresState = tracking::ImageSubresState;
  using CmdBufferTrackingState = tracking::CmdBufferTrackingState;
  using QueueTrackingState = tracking::QueueTrackingState;
}

#endif