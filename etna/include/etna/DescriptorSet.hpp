#pragma once
#ifndef ETNA_DESCRIPTOR_SET_HPP_INCLUDED
#define ETNA_DESCRIPTOR_SET_HPP_INCLUDED

#include <variant>
#include <etna/Vulkan.hpp>
#include "DescriptorSetLayout.hpp"
#include "BindingItems.hpp"

namespace etna::tracking
{
  struct CmdBufferTrackingState;
}

namespace etna
{
  struct Binding
  {
    /*Todo: add resource wrappers*/
    Binding(uint32_t rbinding, const ImageBinding &image_info, uint32_t array_index = 0)
      : binding {rbinding}, arrayElem {array_index}, resources {image_info} {}
    Binding(uint32_t rbinding, const BufferBinding &buffer_info, uint32_t array_index = 0)
      : binding {rbinding}, arrayElem {array_index}, resources {buffer_info} {}
  
    uint32_t binding;
    uint32_t arrayElem;
    std::variant<ImageBinding, BufferBinding> resources;
  };

  /*Maybe we need a hierarchy of descriptor sets*/
  struct DescriptorSet
  {
    DescriptorSet() {}
    DescriptorSet(uint64_t gen, DescriptorLayoutId id, vk::DescriptorSet vk_set, std::vector<Binding> bindings)
      : generation {gen}, layoutId {id}, set {vk_set}, bindings{std::move(bindings)}
    {}

    bool isValid() const;
    
    vk::DescriptorSet getVkSet() const
    {
      return set;
    }

    DescriptorLayoutId getLayoutId() const
    {
      return layoutId;
    }

    uint64_t getGen() const
    {
      return generation;
    }

    const std::vector<Binding>& getBindings() const
    {
      return bindings;
    }

    void requestStates(tracking::CmdBufferTrackingState &state) const;
  
  private:
    uint64_t generation {};
    DescriptorLayoutId layoutId {};
    vk::DescriptorSet set{};
    std::vector<Binding> bindings{};
  };

  /*Base version. Allocate and use descriptor sets while writing command buffer, they will be destroyed
  automaticaly. Resource allocation tracking shoud be added. 
  For long-living descriptor sets (e.g bindless resource sets) separate allocator shoud be added, with ManagedDescriptorSet with destructor*/
  struct DynamicDescriptorPool
  {
    DynamicDescriptorPool(vk::Device dev, uint32_t framesInFlight);
    ~DynamicDescriptorPool();
    
    void flip();
    void destroyAllocatedSets();
    void reset(uint32_t frames_in_flight);

    DescriptorSet allocateSet(DescriptorLayoutId layoutId, std::vector<Binding> bindings);

    vk::DescriptorPool getCurrentPool() const
    {
      return pools[frameIndex];
    }
    
    uint64_t getNumFlips() const
    {
      return flipsCount;
    }

    bool isSetValid(const DescriptorSet &set) const
    {
      return set.getVkSet() && set.getGen() + numFrames > flipsCount;
    }

  private:
    vk::Device vkDevice;

    uint32_t numFrames = 0;
    uint32_t frameIndex = 0;
    uint64_t flipsCount = 0; /*for tracking invalid sets*/
    std::vector<vk::DescriptorPool> pools;
  };

  void write_set(const DescriptorSet &dst);
}

#endif // ETNA_DESCRIPTOR_SET_HPP_INCLUDED
