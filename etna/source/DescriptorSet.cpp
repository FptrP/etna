#include <etna/GlobalContext.hpp>
#include <etna/DescriptorSet.hpp>
#include <etna/Etna.hpp>
#include <etna/SyncCommandBuffer.hpp>

#include <array>
#include <vector>

namespace etna
{

  bool DescriptorSet::isValid() const
  {
    return get_context().getDescriptorPool().isSetValid(*this);
  }

  /*Todo: Add struct with parameters*/
  static constexpr uint32_t NUM_DESCRIPORS = 2048;

  static constexpr uint32_t NUM_TEXTURES = 2048; 
  static constexpr uint32_t NUM_RW_TEXTURES = 512;
  static constexpr uint32_t NUM_BUFFERS = 2048; 
  static constexpr uint32_t NUM_RW_BUFFERS = 512;
  static constexpr uint32_t NUM_SAMPLERS = 128;

  static constexpr std::array<vk::DescriptorPoolSize, 6> g_default_pool_size {
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, NUM_BUFFERS},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, NUM_RW_BUFFERS},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, NUM_SAMPLERS},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, NUM_RW_TEXTURES},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, NUM_RW_TEXTURES},
    vk::DescriptorPoolSize {vk::DescriptorType::eCombinedImageSampler, NUM_TEXTURES}
  };

  DynamicDescriptorPool::DynamicDescriptorPool(vk::Device dev, uint32_t frames_in_flight)
    : vkDevice{dev}
    , numFrames{frames_in_flight}
  {
    frameIndex = 0;
    flipsCount = 0;

    vk::DescriptorPoolCreateInfo info {};
    info.setMaxSets(NUM_DESCRIPORS);
    info.setPoolSizes(g_default_pool_size);

    for (uint32_t i = 0; i < numFrames; i++)
    {
      pools.push_back(vkDevice.createDescriptorPool(info).value);
    }
  }

  DynamicDescriptorPool::~DynamicDescriptorPool()
  {
    for (auto pool : pools)
      vkDevice.destroyDescriptorPool(pool);
  }

  void DynamicDescriptorPool::flip()
  {
    frameIndex = (frameIndex + 1) % numFrames;
    flipsCount++;
    vkDevice.resetDescriptorPool(pools[frameIndex]); /*All allocated sets are destroyed*/
  }

  void DynamicDescriptorPool::destroyAllocatedSets()
  {
    for (uint32_t i = 0; i < numFrames; i++)
      flip();
  }
    
  DescriptorSet DynamicDescriptorPool::allocateSet(DescriptorLayoutId layoutId, std::vector<Binding> bindings)
  {
    auto &dslCache = get_context().getDescriptorSetLayouts();
    auto setLayouts = {dslCache.getVkLayout(layoutId)};

    vk::DescriptorSetAllocateInfo info {};
    info.setDescriptorPool(pools[frameIndex]);
    info.setSetLayouts(setLayouts);

    vk::DescriptorSet vkSet {};
    ETNA_ASSERT(vkDevice.allocateDescriptorSets(&info, &vkSet) == vk::Result::eSuccess);
    return DescriptorSet {flipsCount, layoutId, vkSet, std::move(bindings)};
  }

  static bool is_image_resource(vk::DescriptorType dsType)
  {
    switch (dsType)
    {
    case vk::DescriptorType::eUniformBuffer:
    case vk::DescriptorType::eStorageBuffer:
    case vk::DescriptorType::eUniformBufferDynamic:
    case vk::DescriptorType::eStorageBufferDynamic:
      return false;
    case vk::DescriptorType::eCombinedImageSampler:
    case vk::DescriptorType::eSampledImage:
    case vk::DescriptorType::eStorageImage:
    case vk::DescriptorType::eSampler:
      return true;
    default:
      break;
    }

    ETNA_PANIC("Descriptor write error : unsupported resource ", vk::to_string(dsType));
    return false;
  }

  static void validate_descriptor_write(const DescriptorSet &dst)
  {
    auto &layoutInfo = get_context().getDescriptorSetLayouts().getLayoutInfo(dst.getLayoutId());
    const auto &bindings = dst.getBindings();

    std::array<uint32_t, MAX_DESCRIPTOR_BINDINGS> unboundResources {};

    for (uint32_t binding = 0; binding < MAX_DESCRIPTOR_BINDINGS; binding++)
    {
      unboundResources[binding] = 
        layoutInfo.isBindingUsed(binding)? layoutInfo.getBinding(binding).descriptorCount : 0u;
    }

    for (auto &binding : bindings)
    {
      if (!layoutInfo.isBindingUsed(binding.binding))
        ETNA_PANIC("Descriptor write error: descriptor set doesn't have {} slot", binding.binding);

      auto &bindingInfo = layoutInfo.getBinding(binding.binding);
      bool isImageRequired = is_image_resource(bindingInfo.descriptorType); 
      bool isImageBinding = std::get_if<ImageBinding>(&binding.resources) != nullptr;
      if (isImageRequired != isImageBinding)
      {
        ETNA_PANIC("Descriptor write error: slot {} {} required but {} bound", binding.binding,
            (isImageRequired ? "image" : "buffer"),
            (isImageBinding  ? "imaged" : "buffer"));
      }

      unboundResources[binding.binding] -= 1;
    }

    for (uint32_t binding = 0; binding < MAX_DESCRIPTOR_BINDINGS; binding++)
    {
      if (unboundResources[binding])
        ETNA_PANIC("Descriptor write error: slot ", binding, " has ", unboundResources[binding], " unbound resources");
    }
  }

  void write_set(const DescriptorSet &dst)
  {
    ETNA_ASSERT(dst.isValid());
    validate_descriptor_write(dst);

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(dst.getBindings().size());

    uint32_t numBufferInfo = 0;
    uint32_t numImageInfo = 0;

    auto &layoutInfo = get_context().getDescriptorSetLayouts().getLayoutInfo(dst.getLayoutId());

    for (auto &binding : dst.getBindings())
    {
      auto &bindingInfo = layoutInfo.getBinding(binding.binding);
      if (is_image_resource(bindingInfo.descriptorType))
        numImageInfo++;
      else
        numBufferInfo++;
    }

    std::vector<vk::DescriptorImageInfo> imageInfos;
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    imageInfos.resize(numImageInfo);
    bufferInfos.resize(numBufferInfo);
    numImageInfo = 0;
    numBufferInfo = 0;

    for (auto &binding : dst.getBindings())
    {
      auto &bindingInfo = layoutInfo.getBinding(binding.binding);
      vk::WriteDescriptorSet write {};
      write
        .setDstSet(dst.getVkSet())
        .setDescriptorCount(1)
        .setDstBinding(binding.binding)
        .setDstArrayElement(binding.arrayElem)
        .setDescriptorType(bindingInfo.descriptorType);

      if (is_image_resource(bindingInfo.descriptorType))
      {
        auto img = std::get<ImageBinding>(binding.resources).descriptor_info;
        imageInfos[numImageInfo] = img;
        write.setPImageInfo(imageInfos.data() + numImageInfo);
        numImageInfo++;
      }
      else
      {
        auto buf = std::get<BufferBinding>(binding.resources).descriptor_info;
        bufferInfos[numBufferInfo] = buf;
        write.setPBufferInfo(bufferInfos.data() + numBufferInfo);
        numBufferInfo++;
      }

      writes.push_back(write);
    }

    get_context().getDevice().updateDescriptorSets(writes, {});
  }

  static vk::PipelineStageFlags2 shader_stage_to_pipeline_stage(vk::ShaderStageFlags shader_stages)
  {
    constexpr std::tuple<vk::ShaderStageFlagBits, vk::PipelineStageFlagBits2> mapping[]
    {
      {vk::ShaderStageFlagBits::eVertex, vk::PipelineStageFlagBits2::eVertexShader},
      {vk::ShaderStageFlagBits::eFragment, vk::PipelineStageFlagBits2::eFragmentShader},
      {vk::ShaderStageFlagBits::eCompute, vk::PipelineStageFlagBits2::eComputeShader},
      {vk::ShaderStageFlagBits::eGeometry, vk::PipelineStageFlagBits2::eGeometryShader},
      {vk::ShaderStageFlagBits::eTessellationControl, vk::PipelineStageFlagBits2::eTessellationControlShader},
      {vk::ShaderStageFlagBits::eTessellationEvaluation, vk::PipelineStageFlagBits2::eTessellationEvaluationShader},
      {vk::ShaderStageFlagBits::eMeshEXT, vk::PipelineStageFlagBits2::eMeshShaderEXT}
    };

    vk::PipelineStageFlags2 flags{};

    for (const auto &[shaderFlag, pipelineFlag] : mapping)
    {
      if (shader_stages & shaderFlag)
        flags |= pipelineFlag;
    }
    return flags;
  }

  static vk::AccessFlags2 descriptor_type_to_access_flag(vk::DescriptorType descriptor_type)
  {
    constexpr std::tuple<vk::DescriptorType, vk::AccessFlags2> mapping[]
    {
      {vk::DescriptorType::eSampler, {}},
      {vk::DescriptorType::eSampledImage, vk::AccessFlagBits2::eShaderSampledRead},
      {vk::DescriptorType::eCombinedImageSampler, vk::AccessFlagBits2::eShaderSampledRead},
      {vk::DescriptorType::eUniformTexelBuffer, vk::AccessFlagBits2::eShaderSampledRead},

      {vk::DescriptorType::eUniformBuffer, vk::AccessFlagBits2::eUniformRead},
      {vk::DescriptorType::eUniformBufferDynamic, vk::AccessFlagBits2::eUniformRead},

      //one flag may be ok here. TODO : more info about bindings
      {vk::DescriptorType::eStorageBuffer, vk::AccessFlagBits2::eShaderStorageRead|vk::AccessFlagBits2::eShaderStorageWrite},
      {vk::DescriptorType::eStorageBufferDynamic, vk::AccessFlagBits2::eShaderStorageRead|vk::AccessFlagBits2::eShaderStorageWrite},
      {vk::DescriptorType::eStorageImage, vk::AccessFlagBits2::eShaderStorageRead|vk::AccessFlagBits2::eShaderStorageWrite},
      {vk::DescriptorType::eStorageTexelBuffer, vk::AccessFlagBits2::eShaderStorageRead|vk::AccessFlagBits2::eShaderStorageWrite},

      {vk::DescriptorType::eInputAttachment, vk::AccessFlagBits2::eInputAttachmentRead}
      
      //TODO: acceleration structures
    };

    for (auto [desc, access] : mapping)
    {
      if (desc == descriptor_type)
        return access;
    }
    return {};
  }

  void DescriptorSet::requestStates(CmdBufferTrackingState &state) const
  {
    auto &layoutInfo = get_context().getDescriptorSetLayouts().getLayoutInfo(layoutId);
    for (auto &binding : bindings)
    {
      auto &bindingInfo = layoutInfo.getBinding(binding.binding);

      if (auto image = std::get_if<ImageBinding>(&binding.resources))
      {
        state.requestState(
          image->image, 
          image->range.baseMipLevel, 
          image->range.levelCount,
          image->range.baseArrayLayer,
          image->range.layerCount,
          ImageSubresState {
            .activeStages = shader_stage_to_pipeline_stage(bindingInfo.stageFlags),
            .activeAccesses = descriptor_type_to_access_flag(bindingInfo.descriptorType),
            .layout = image->descriptor_info.imageLayout
          });
      }
      else if (auto buffer = std::get_if<BufferBinding>(&binding.resources))
      {
        state.requestState(
          buffer->buffer,
          CmdBufferTrackingState::BufferState {
            .activeStages = shader_stage_to_pipeline_stage(bindingInfo.stageFlags),
            .activeAccesses = descriptor_type_to_access_flag(bindingInfo.descriptorType)
          });
      }
    }

  }
}
