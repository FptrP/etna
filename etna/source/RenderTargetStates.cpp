#include "etna/RenderTargetStates.hpp"

#include "etna/GlobalContext.hpp"

#include <unordered_map>
#include <variant>

namespace etna
{

bool RenderTargetState::inScope = false;

RenderTargetState::RenderTargetState(
    SyncCommandBuffer &cmd_,
    vk::Extent2D extent,
    const vk::ArrayProxy<RenderingAttachment> &color_attachments, 
    std::optional<RenderingAttachment> depth_attachment)
  : cmd {cmd_}
{
  ETNA_ASSERTF(!inScope, "RenderTargetState scopes shouldn't overlap.");
  inScope = true;
  
  vk::Viewport viewport
  {
    .x = 0.0f,
    .y = 0.0f,
    .width  = static_cast<float>(extent.width),
    .height = static_cast<float>(extent.height),
    .minDepth = 0.0f,
    .maxDepth = 1.0f
  };
  vk::Rect2D scissor
  {
    .offset = {0, 0},
    .extent = extent
  };

  cmd.beginRendering(scissor, color_attachments, depth_attachment);
  cmd.setViewport(0, {viewport});
  cmd.setScissor(0, {scissor});
}

RenderTargetState::~RenderTargetState()
{
  cmd.endRendering();
  inScope = false;
}
}
