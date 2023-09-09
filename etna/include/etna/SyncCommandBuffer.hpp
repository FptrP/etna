#pragma once
#ifndef ETNA_SYNC_COMMAND_BUFFER_INCLUDED
#define ETNA_SYNC_COMMAND_BUFFER_INCLUDED

#include <etna/Image.hpp>
#include <etna/Buffer.hpp>

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

  //Sets resource state. 
  void expectState(const Image &image, uint32_t mip, uint32_t layer, ImageState::SubresourceState state);
  void expectState(const Buffer &buffer, BufferState state);

  //requests transition to new state
  void requestState(const Image &image, uint32_t mip, uint32_t layer, ImageState::SubresourceState state);
  void requestState(const Buffer &buffer, BufferState state);

  void flushBarrier(CmdBarrier &barrier);

  void onSync(); //sets all activeStages and accesses to zero, saves image layouts

  ResContainer takeStates()
  {
    ResContainer out {};
    std::swap(resources, out);
    return out;
  }

private:

  static std::optional<vk::ImageMemoryBarrier2> genBarrier(vk::Image img,
    vk::ImageAspectFlags aspect,
    uint32_t mip,
    uint32_t layer,
    ImageState::SubresourceState &src,
    const ImageState::SubresourceState &dst);

  static void genBarrier(std::optional<vk::MemoryBarrier2> &barrier,
    BufferState &src,
    const BufferState &dst);

  ResContainer resources; //TODO: store expected states in separate ResContainer
  ResContainer requests;
};

struct SyncCommandBuffer
{
  void begin();
  void end();
  
  void clearColorImage();
  void copyBuffers();
  void copyBufferToImage();
  //e.t.c.


private:
  vk::CommandBuffer cmd;
  CmdBufferTrackingState states;
};

using ImageSubresState = CmdBufferTrackingState::ImageState::SubresourceState;

}

#endif