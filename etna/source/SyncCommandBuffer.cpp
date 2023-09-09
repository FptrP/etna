#include "etna/SyncCommandBuffer.hpp"

namespace etna 
{

CmdBufferTrackingState::ImageState &find_or_add(
  CmdBufferTrackingState::ResContainer &resources,
  const Image &image)
{
  auto handle = reinterpret_cast<CmdBufferTrackingState::HandleT>(VkImage(image.get()));
  auto it = resources.find(handle);
  if (it == resources.end())
    it = resources.emplace(handle, CmdBufferTrackingState::ImageState{image}).first;
  auto state = std::get_if<CmdBufferTrackingState::ImageState>(&it->second);
  ETNA_ASSERT(state);
  return *state;
}

// acquire logic should be here in request
CmdBufferTrackingState::BufferState &find_or_add(
  CmdBufferTrackingState::ResContainer &resources,
  const Buffer &buffer)
{
  auto handle = reinterpret_cast<CmdBufferTrackingState::HandleT>(VkBuffer(buffer.get()));
  auto it = resources.find(handle);
  if (it == resources.end())
    it = resources.emplace(handle, CmdBufferTrackingState::BufferState{}).first;
  auto state = std::get_if<CmdBufferTrackingState::BufferState>(&it->second);
  ETNA_ASSERT(state);
  return *state;
}

void CmdBufferTrackingState::expectState(
  const Image &image, 
  uint32_t mip, 
  uint32_t layer, 
  ImageState::SubresourceState state)
{
  auto &srcState = find_or_add(resources, image).getSubresource(mip, layer);
  if (srcState.has_value())
  {
    ETNA_ASSERTF(*srcState == state, "Incompatible state");
    return;
  }
  *srcState = state;
}

void CmdBufferTrackingState::requestState(
  const Image &image, 
  uint32_t mip, 
  uint32_t layer, 
  ImageState::SubresourceState state)
{
  auto &dstState = find_or_add(requests, image).getSubresource(mip, layer);
  if (!dstState.has_value())
  {
    *dstState = state; //acquire logic should be here
    return;
  }

  ETNA_ASSERTF(dstState->layout == state.layout, "Different layouts requested for image");
  dstState->activeAccesses |= state.activeAccesses; // TODO: check if accesses are compatible
  dstState->activeStages |= state.activeStages;
}

constexpr vk::AccessFlagBits2 READ_ACCESS[] {
  vk::AccessFlagBits2::eAccelerationStructureReadKHR,
  vk::AccessFlagBits2::eIndexRead,
  vk::AccessFlagBits2::eIndirectCommandRead,
  vk::AccessFlagBits2::eVertexAttributeRead,
  vk::AccessFlagBits2::eUniformRead,
  vk::AccessFlagBits2::eInputAttachmentRead,
  vk::AccessFlagBits2::eShaderRead,
  vk::AccessFlagBits2::eColorAttachmentRead,
  vk::AccessFlagBits2::eDepthStencilAttachmentRead,
  vk::AccessFlagBits2::eTransferRead,
  vk::AccessFlagBits2::eMemoryRead,
  vk::AccessFlagBits2::eShaderSampledRead,
  vk::AccessFlagBits2::eShaderStorageRead,
};

constexpr vk::AccessFlags2 READ_ACCESS_MASK = 
  vk::AccessFlagBits2::eAccelerationStructureReadKHR
  | vk::AccessFlagBits2::eIndexRead
  | vk::AccessFlagBits2::eIndirectCommandRead
  | vk::AccessFlagBits2::eVertexAttributeRead
  | vk::AccessFlagBits2::eUniformRead
  | vk::AccessFlagBits2::eInputAttachmentRead
  | vk::AccessFlagBits2::eShaderRead
  | vk::AccessFlagBits2::eColorAttachmentRead
  | vk::AccessFlagBits2::eDepthStencilAttachmentRead
  | vk::AccessFlagBits2::eTransferRead
  | vk::AccessFlagBits2::eMemoryRead
  | vk::AccessFlagBits2::eShaderSampledRead
  | vk::AccessFlagBits2::eShaderStorageRead;

constexpr vk::AccessFlagBits2 WRITE_ACCESS[] {
  vk::AccessFlagBits2::eShaderWrite,
  vk::AccessFlagBits2::eColorAttachmentWrite,
  vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
  vk::AccessFlagBits2::eTransferWrite,
  vk::AccessFlagBits2::eMemoryWrite,
  vk::AccessFlagBits2::eShaderStorageWrite
};

constexpr vk::AccessFlags2 WRITE_ACCESS_MASK = 
  vk::AccessFlagBits2::eShaderWrite
  | vk::AccessFlagBits2::eColorAttachmentWrite
  | vk::AccessFlagBits2::eDepthStencilAttachmentWrite
  | vk::AccessFlagBits2::eTransferWrite
  | vk::AccessFlagBits2::eMemoryWrite
  | vk::AccessFlagBits2::eShaderStorageWrite;

static constexpr bool is_read_access(vk::AccessFlags2 flags)
{
  return (flags & READ_ACCESS_MASK) != vk::AccessFlags2{};
}

static constexpr bool is_write_access(vk::AccessFlags2 flags)
{
  return (flags & WRITE_ACCESS_MASK) != vk::AccessFlags2{};
}


std::optional<vk::ImageMemoryBarrier2> CmdBufferTrackingState::genBarrier(
  vk::Image img,
  vk::ImageAspectFlags aspect,
  uint32_t mip,
  uint32_t layer,
  ImageState::SubresourceState &src,
  const ImageState::SubresourceState &dst)
{
  // write -> write  - barrier, dstStage = only write command, dstAccess=only writes
  // write -> read   - barrier, dstStage = all commands, dstAccess = memoryRead|memoryWrite
  // layout change   - barrier, dstStage = all commands, dstAccess = memoryRead|memoryWrite
  // read -> write   - execution dependency only (but again, with dstStage = all commands :( 
  // read -> read    - no barrier (thats why dstStages = all commands required)

  vk::ImageSubresourceRange range {
    .aspectMask = aspect,
    .baseMipLevel = mip,
    .levelCount = 1,
    .baseArrayLayer = layer,
    .layerCount = 1
  };

  //layout change
  if (src.layout != dst.layout)
  {
    auto barrier = vk::ImageMemoryBarrier2 {
      .srcStageMask = src.activeStages| vk::PipelineStageFlagBits2::eTopOfPipe, // incorrect mask can be here
      .srcAccessMask = src.activeAccesses & WRITE_ACCESS_MASK, // incorrect mask can be here
      .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
      .dstAccessMask = vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite,
      .oldLayout = src.layout,
      .newLayout = dst.layout,
      .image = img,
      .subresourceRange = range
    };
    src = dst;
    return barrier;
  }

  bool isSrcWrite = is_write_access(src.activeAccesses);
  bool isSrcRead = is_read_access(src.activeAccesses);
  bool isDstWrite = is_write_access(dst.activeAccesses);
  bool isDstRead = is_read_access(dst.activeAccesses);

  if (isSrcWrite) //barrier required
  {
    vk::ImageMemoryBarrier2 barrier {
      .srcStageMask = src.activeStages,
      .srcAccessMask = src.activeAccesses & WRITE_ACCESS_MASK, // & writeMask? 
      .image = img,
      .subresourceRange = range
    };

    if (isDstWrite) //write (and maybe read); make available for next command only
    {
      barrier.dstStageMask = dst.activeStages;
      barrier.dstAccessMask = dst.activeAccesses;
    }
    else if (isDstRead) // read only. make available for all accesses to not insert barrier between reads
    {
      barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
      barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite;
    }
    src = dst;
    return barrier;
  }

  if (isSrcRead && isDstWrite) // and !isSrcWrite
  {
    vk::ImageMemoryBarrier2 barrier {
      .srcStageMask = src.activeStages,
      .srcAccessMask = vk::AccessFlags2{}, // no accesses to make visible  
      .dstStageMask = dst.activeStages,
      .dstAccessMask = vk::AccessFlags2{}, // if image was in read state, than it is visible for all accesses
      .image = img,
      .subresourceRange = range
    };
    src = dst;
    return barrier;
  }
  else if (isSrcRead && isDstRead) // and !isDstWrite
  {
    src.activeAccesses |= dst.activeAccesses;
    src.activeStages |= dst.activeStages;
    return {};
  }

  //!isSrcWrite && !isSrcRead -> resource was not used 
  ETNA_ASSERTF(src.activeAccesses == vk::AccessFlagBits2::eNone, "Unknown resource access");
  ETNA_ASSERTF(src.activeStages == vk::PipelineStageFlags2{}, "Unknown pipeline stages");
  // 1) resource was not used yet, but is memory may be used
  // (it's previous owner was destroyed and it is allocated again)
  // this should somehow been managed with fences e.t.c
  // 2) or explicit barrier was called before
  src = dst;
  return {};
}

static void merge(std::optional<vk::MemoryBarrier2> &dst, const vk::MemoryBarrier2 &src)
{
  if (dst.has_value())
  {
    dst->srcStageMask |= src.srcStageMask;
    dst->srcAccessMask |= src.srcAccessMask;
    dst->dstStageMask |= src.dstStageMask;
    dst->dstAccessMask |= src.dstAccessMask;
    return;
  }
  *dst = src;
}

void CmdBufferTrackingState::genBarrier(std::optional<vk::MemoryBarrier2> &barrier,
    BufferState &src,
    const BufferState &dst)
{
  bool isSrcWrite = is_write_access(src.activeAccesses);
  bool isSrcRead = is_read_access(src.activeAccesses);
  bool isDstWrite = is_write_access(dst.activeAccesses);
  bool isDstRead = is_read_access(dst.activeAccesses);

  if (isSrcWrite) //flush writes
  {
    vk::MemoryBarrier2 nextBarrier {
      .srcStageMask = src.activeStages,
      .srcAccessMask = src.activeAccesses & WRITE_ACCESS_MASK
    };

    if (isDstWrite)
    {
      nextBarrier.dstStageMask = dst.activeStages;
      nextBarrier.dstAccessMask = dst.activeAccesses;
    }
    else if (isDstRead)
    {
      nextBarrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
      nextBarrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite;
    }
    src = dst;
    merge(barrier, nextBarrier);
    return; 
  }
  
  if (isSrcRead && isDstWrite)
  {
    vk::MemoryBarrier2 nextBarrier {
      .srcStageMask = src.activeStages,
      .srcAccessMask = vk::AccessFlags2 {},
      .dstStageMask = dst.activeStages,
      .dstAccessMask = vk::AccessFlags2 {} //if buffer was in read state, than it was visible for all accesses
    };
    src = dst;
    merge(barrier, nextBarrier);
    return;
  }
  else if (isSrcRead && isDstRead) //no barriers
  {
    src.activeAccesses |= dst.activeAccesses;
    src.activeStages |= dst.activeStages;
    return;
  }

  ETNA_ASSERTF(src.activeAccesses == vk::AccessFlagBits2::eNone, "Unknown resource access");
  ETNA_ASSERTF(src.activeStages == vk::PipelineStageFlags2{}, "Unknown pipeline stages");
  // first use or after manualy placed barrier? 
  src = dst;
}

void CmdBufferTrackingState::flushBarrier(CmdBarrier &barrier)
{
  for (auto &[handle, state] : requests)
  {
    if (auto imageState = std::get_if<ImageState>(&state))
    {
      auto it = resources.find(handle);
      if (it == resources.end()) // create state, assuming that resource is not used
      {
        it = resources.emplace( //acquire logic should be added here
          handle, 
          ImageState{
            imageState->resource, 
            imageState->aspect, 
            imageState->mipLevels, 
            imageState->arrayLayers
        }).first;
      }

      auto &srcState = std::get<ImageState>(it->second);
      auto apiImage = srcState.resource;
      ETNA_ASSERT(apiImage == imageState->resource);

      for (uint32_t layer = 0; layer < imageState->arrayLayers; layer++)
      {
        for (uint32_t mip = 0; mip < imageState->mipLevels; mip++)
        {
          auto &srcSubres = srcState.getSubresource(mip, layer);
          auto &dstSubres = imageState->getSubresource(mip, layer);

          if (!srcSubres.has_value())
          {
            srcSubres = ImageState::SubresourceState {}; //create default. Maybe assert? 
            // in future acquire logic here
          }
          if (auto img_barrier = genBarrier(apiImage, srcState.aspect, mip, layer, *srcSubres, *dstSubres))
            barrier.imageBarriers.push_back(*img_barrier);
        }
      }
    }
    else if (auto bufferState = std::get_if<BufferState>(&state))
    {
      auto it = resources.find(handle);
      if (it == resources.end())
      { //acquire logic should be added here
        it = resources.emplace(handle, BufferState{}).first;
      } 
      auto &srcState = std::get<BufferState>(it->second); 
      genBarrier(barrier.memoryBarrier, srcState, *bufferState);
    }
  }

  requests.clear();
}


void CmdBufferTrackingState::expectState(const Buffer &buffer, BufferState state)
{
  auto handle = reinterpret_cast<HandleT>(VkBuffer(buffer.get()));
  auto it = resources.find(handle);
  ETNA_ASSERTF(it == resources.end(), "incompatible initial state for buffer");
  
  resources.emplace(handle, state);
}

void CmdBufferTrackingState::requestState(const Buffer &buffer, BufferState state)
{
  auto &dstState = find_or_add(requests, buffer);
  dstState.activeAccesses |= state.activeAccesses;
  dstState.activeStages |= state.activeStages;
}

void CmdBufferTrackingState::onSync()
{
  ETNA_ASSERT(requests.size() == 0);
  
  for (auto &[_, res] : resources)
  {
    if (auto imageState = std::get_if<ImageState>(&res))
    {
      for (auto &substate : imageState->states)
      {
        if (substate.has_value())
        {
          substate->activeAccesses = vk::AccessFlags2{};
          substate->activeStages = vk::PipelineStageFlags2{};
        }
      }
    }
    else if (auto bufferState = std::get_if<BufferState>(&res))
    {
      *bufferState = BufferState{};
    }
  }
}

void CmdBarrier::flush(vk::CommandBuffer cmd)
{
  vk::DependencyInfo info {
    .memoryBarrierCount = memoryBarrier.has_value()? 1 : 0,
    .pMemoryBarriers = &*memoryBarrier,
    .imageMemoryBarrierCount = imageBarriers.size(),
    .pImageMemoryBarriers = imageBarriers.data()
  };
  
  cmd.pipelineBarrier2(info);
  clear();
}


} // namespace etna