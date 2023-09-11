#pragma once
#ifndef ETNA_STATES_HPP_INCLUDED
#define ETNA_STATES_HPP_INCLUDED

#include <etna/Image.hpp>
#include <etna/SyncCommandBuffer.hpp>
#include <vulkan/vulkan.hpp>

#include <vector>

namespace etna
{

class RenderTargetState
{
  SyncCommandBuffer &cmd;
  static bool inScope;
public:  
  RenderTargetState(
    SyncCommandBuffer &cmd_,
    vk::Extent2D extend,
    const vk::ArrayProxy<RenderingAttachment> &color_attachments, 
    std::optional<RenderingAttachment> depth_attachment);
  
  ~RenderTargetState();
};

}

#endif // ETNA_STATES_HPP_INCLUDED
