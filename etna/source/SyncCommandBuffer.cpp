#include "etna/SyncCommandBuffer.hpp"
#include "etna/Etna.hpp"
#include "etna/DescriptorSet.hpp"
#include "etna/GlobalContext.hpp"

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
  auto &srcState = find_or_add(expectedResources, image).getSubresource(mip, layer);
  srcState = state;
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
    dstState = state; //acquire logic should be here
    return;
  }

  ETNA_ASSERTF(dstState->layout == state.layout, "Different layouts requested for image");
  dstState->activeAccesses |= state.activeAccesses; // TODO: check if accesses are compatible
  dstState->activeStages |= state.activeStages;
}

void CmdBufferTrackingState::requestState(const Image &image, uint32_t firstMip, uint32_t mipCount, 
    uint32_t firstLayer, uint32_t layerCount, ImageState::SubresourceState state)
{
  for (uint32_t mip = firstMip; mip < firstMip + mipCount; mip++)
  {
    for (uint32_t layer = firstLayer; layer < firstLayer + layerCount; layer++)
    {
      requestState(image, mip, layer, state);
    }
  }
}

void CmdBufferTrackingState::requestState(const Image &image, vk::ImageSubresourceRange range, 
  ImageState::SubresourceState state)
{
  requestState(image, range.baseMipLevel, range.levelCount, range.baseArrayLayer, range.layerCount, state);
}

void CmdBufferTrackingState::initResourceStates(const ResContainer &states)
{
  if (!expectedResources.size())
  {  
    expectedResources = states;
    return;
  }

  ETNA_ASSERT(false); //unimplemented

}

void CmdBufferTrackingState::initResourceStates(ResContainer &&states)
{
  if (!expectedResources.size())
  {
    expectedResources = std::move(states);
    return;
  }

  ETNA_ASSERT(false); //unimplemented
}


void QueueTrackingState::onWait() //clears all activeStages/activeAccesses
{
  for (auto &[_, res] : currentStates)
  {
    if (auto imageState = std::get_if<CmdBufferTrackingState::ImageState>(&res))
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
    else if (auto bufferState = std::get_if<CmdBufferTrackingState::BufferState>(&res))
    {
      *bufferState = CmdBufferTrackingState::BufferState{};
    }
  }
}

static bool is_compatible(const CmdBufferTrackingState::BufferState &state,
                          const CmdBufferTrackingState::BufferState &expected)
{
  //state.activeStages >= state.activeStages;  
  bool stagesCompatible = bool(expected.activeStages & vk::PipelineStageFlagBits2::eAllCommands);
  bool accessesCompatible = bool(expected.activeAccesses & 
    (vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite));
  //TODO: more universal compatible cases: read only
  if ((state.activeStages & expected.activeStages) == state.activeStages)
    stagesCompatible |= true;
  if ((state.activeAccesses & expected.activeAccesses) == state.activeAccesses)
    accessesCompatible |= true;
  
  return stagesCompatible && accessesCompatible;
}

static bool is_compatible(const ImageSubresState &state, const ImageSubresState &expected)
{
  if (state.layout != expected.layout && expected.layout != vk::ImageLayout::eUndefined)
    return false;
  
  bool stagesCompatible = bool(expected.activeStages & vk::PipelineStageFlagBits2::eAllCommands);
  bool accessesCompatible = bool(expected.activeAccesses & 
    (vk::AccessFlagBits2::eMemoryRead|vk::AccessFlagBits2::eMemoryWrite));
  //TODO: more universal compatible cases: read only

  if ((state.activeStages & expected.activeStages) == state.activeStages)
    stagesCompatible |= true;
  if ((state.activeAccesses & expected.activeAccesses) == state.activeAccesses)
    accessesCompatible |= true;
  
  return stagesCompatible && accessesCompatible;
}

void QueueTrackingState::onSubmit(CmdBufferTrackingState &state)
{
  state.removeUnusedResources();
  const auto &expectedStates = state.getExpectedStates();
  
  for (auto &[handle, state] : expectedStates)
  {
    auto it = currentStates.find(handle);
    if (it == currentStates.end()) //resource was not used yet
      continue;

    const auto &currentState = it->second;

    if (auto imageState = std::get_if<CmdBufferTrackingState::ImageState>(&state))
    {
      auto srcState = std::get_if<CmdBufferTrackingState::ImageState>(&currentState);
      ETNA_ASSERT(srcState);
      for (uint32_t i = 0; i < imageState->states.size(); i++)
      {
        if (srcState->states[i].has_value() && imageState->states[i].has_value())
        {
          ETNA_ASSERTF(is_compatible(*srcState->states[i], *imageState->states[i]), \
            "Expected resource state is incompatible with actual resource state");
        }
      }
    }
    else if (auto bufferState = std::get_if<CmdBufferTrackingState::BufferState>(&state))
    {
      auto srcState = std::get_if<CmdBufferTrackingState::BufferState>(&currentState);
      ETNA_ASSERT(srcState);
      ETNA_ASSERTF(is_compatible(*srcState, *bufferState), \
        "Expected resource state is incompatible with actual resource state");
    }
  }

  //update current states

  const auto &resources = state.getStates();
  for (auto &[handle, state] : resources)
  {
    auto it = currentStates.find(handle);
    if (it == currentStates.end())
    {
      currentStates.emplace(handle, state);
      continue;
    }
    
    if (auto imageState = std::get_if<CmdBufferTrackingState::ImageState>(&state))
    {
      auto dstState = std::get_if<CmdBufferTrackingState::ImageState>(&it->second);
      ETNA_ASSERT(dstState);

      for (uint32_t i = 0; i < imageState->states.size(); i++)
      {
        if (imageState->states[i].has_value())
          dstState->states[i] = imageState->states[i];
      }
    }
    else 
    {
      //buffers
      ETNA_ASSERT(it->second.index() == state.index());
      it->second = state;
    }
    //it->second = state;
  }

  state.clearAll();
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

CmdBufferTrackingState::BufferState &CmdBufferTrackingState::acquireResource(
  CmdBufferTrackingState::HandleT handle)
{
  //check resources
  auto it = resources.find(handle);
  if (it != resources.end())
  {
    auto state = std::get_if<BufferState>(&it->second);
    ETNA_ASSERT(state);
    return *state;
  }
  //check expected states, import
  it = expectedResources.find(handle);
  if (it != expectedResources.end())
  {
    auto state = std::get_if<BufferState>(&it->second);
    ETNA_ASSERT(state);
    it = resources.emplace(handle, *state).first;
    return std::get<BufferState>(it->second);
  }

  //TODO: request from queue
  //For now assume that resource is not used
  expectedResources.emplace(handle, BufferState{});
  it = resources.emplace(handle, BufferState{}).first;
  return std::get<BufferState>(it->second);
}

ImageSubresState &CmdBufferTrackingState::acquireResource(
  CmdBufferTrackingState::HandleT handle, 
  const CmdBufferTrackingState::ImageState &request_state, uint32_t mip, uint32_t layer)
{
  //check resources
  bool imageInResources = false;
  bool imageInExpected = false;
  auto it = resources.find(handle);
  
  if (it != resources.end())
  {
    auto imageState = std::get_if<ImageState>(&it->second);
    ETNA_ASSERT(imageState);
    auto &subState = imageState->getSubresource(mip, layer);
    if (subState.has_value())
    {
      return *subState;
    } // else subState is not in expectedStates too
    imageInResources = true;
  }

  it = expectedResources.find(handle);
  if (it != expectedResources.end())
  {
    auto imageState = std::get_if<ImageState>(&it->second);
    ETNA_ASSERT(imageState);

    if (imageState->getSubresource(mip, layer).has_value())
    {
      //expectedState contains info, but it is not added to resources yet
      auto [res, unique] = resources.emplace(handle, ImageState{*imageState});
      ETNA_ASSERT(unique);
      auto &subres = std::get<ImageState>(res->second).getSubresource(mip, layer);
      ETNA_ASSERT(subres.has_value());
      return *subres;
    }
    imageInExpected = true;
  }
  
  //1) expectedResources and resources both not contain handle
  //2) expectedResources and resources both contain handle, but both do not contain subresource 
  ETNA_ASSERT(imageInExpected == imageInResources);
  ImageState *imageExpected = nullptr;
  ImageState *imageResources = nullptr;

  if (!imageInExpected && !imageInResources)
  {
    auto it = expectedResources.emplace(handle, ImageState{
      request_state.resource, 
      request_state.aspect,
      request_state.mipLevels,
      request_state.arrayLayers}
    ).first;

    imageExpected = std::get_if<ImageState>(&it->second);

    it = resources.emplace(handle, ImageState{
      request_state.resource, 
      request_state.aspect,
      request_state.mipLevels,
      request_state.arrayLayers}
    ).first;

    imageResources = std::get_if<ImageState>(&it->second);
  }
  else
  {
    imageExpected = std::get_if<ImageState>(&expectedResources.at(handle));
    imageResources = std::get_if<ImageState>(&resources.at(handle));
  }

  ETNA_ASSERT(imageExpected && imageResources);

  imageExpected->getSubresource(mip, layer) = ImageSubresState{};
  auto &dstSubres = imageResources->getSubresource(mip, layer);
  dstSubres = ImageSubresState{};
  return *dstSubres;
}

void CmdBufferTrackingState::flushBarrier(CmdBarrier &barrier)
{
  for (auto &[handle, state] : requests)
  {
    if (auto imageState = std::get_if<ImageState>(&state))
    {
      for (uint32_t layer = 0; layer < imageState->arrayLayers; layer++)
      {
        for (uint32_t mip = 0; mip < imageState->mipLevels; mip++)
        {
          auto &srcSubres = acquireResource(handle, *imageState, mip, layer);
          auto &dstSubres = imageState->getSubresource(mip, layer);
          
          auto imgBarrier = genBarrier(imageState->resource, imageState->aspect, 
            mip, layer, srcSubres, *dstSubres);
          
          if (imgBarrier.has_value())
            barrier.imageBarriers.push_back(*imgBarrier);
        }
      }
    }
    else if (auto bufferState = std::get_if<BufferState>(&state))
    {
      auto &srcState = acquireResource(handle);
      genBarrier(barrier.memoryBarrier, srcState, *bufferState);
    }
  }

  requests.clear();
}

void CmdBufferTrackingState::removeUnusedResources()
{
  ETNA_ASSERT(requests.size() == 0);

  for (auto &[handle, state] : resources)
  {
    if (auto imageState = std::get_if<ImageState>(&state))
    {
      auto it = expectedResources.find(handle);
      ETNA_ASSERT(it != expectedResources.end());
      auto expectedState = std::get_if<ImageState>(&it->second);
      ETNA_ASSERT(expectedState);

      for (uint32_t i = 0; i < imageState->states.size(); i++)
      {
        if (!imageState->states[i].has_value())
        {
          expectedState->states.at(i) = std::nullopt;
        }
      }
    }
    else if (std::holds_alternative<BufferState>(state))
    {
      //assert expected resources also
      auto it = expectedResources.find(handle);
      ETNA_ASSERT(it != expectedResources.end());
      ETNA_ASSERT(std::holds_alternative<BufferState>(it->second));
      expectedResources.erase(it);
    }
  }
}

void CmdBufferTrackingState::expectState(const Buffer &buffer, BufferState state)
{
  auto handle = reinterpret_cast<HandleT>(VkBuffer(buffer.get()));
  auto it = expectedResources.find(handle);
  if (it == expectedResources.end())
    expectedResources.emplace(handle, state);
  else
    it->second = state;
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
  // What's with expectedResources?????
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
  if (!memoryBarrier.has_value() && !imageBarriers.size())
    return;

  vk::DependencyInfo info {
    .memoryBarrierCount = memoryBarrier.has_value()? 1 : 0,
    .pMemoryBarriers = &*memoryBarrier,
    .imageMemoryBarrierCount = imageBarriers.size(),
    .pImageMemoryBarriers = imageBarriers.data()
  };
  
  cmd.pipelineBarrier2(info);
  clear();
}

void SyncCommandBuffer::expectState(const Buffer &buffer, CmdBufferTrackingState::BufferState state)
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

  trackingState.flushBarrier(barrier);
  barrier.flush(*cmd);
  cmd->clearColorImage(image.get(), layout, clear_color, ranges);
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
  trackingState.requestState(buffer, CmdBufferTrackingState::BufferState {
    vk::PipelineStageFlagBits2::eVertexInput,
    vk::AccessFlagBits2::eVertexAttributeRead
  });

  cmd->bindVertexBuffers(binding_index, {buffer.get()}, {offset});
}

void SyncCommandBuffer::bindIndexBuffer(const Buffer &buffer, uint32_t offset, vk::IndexType type)
{
  trackingState.requestState(buffer, CmdBufferTrackingState::BufferState {
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

} // namespace etna