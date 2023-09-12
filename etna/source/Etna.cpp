#include <memory>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <vulkan/vulkan_format_traits.hpp>


namespace etna
{
  static std::unique_ptr<GlobalContext> g_context {};

  GlobalContext &get_context()
  {
    return *g_context;
  }
  
  bool is_initilized()
  {
    return static_cast<bool>(g_context);
  }
  
  void initialize(const InitParams &params)
  {
    g_context.reset(new GlobalContext(params));
  }
  
  void shutdown()
  {
    g_context->getDescriptorSetLayouts().clear(g_context->getDevice());
    g_context.reset(nullptr);
  }

  ShaderProgramId create_program(const std::string &name, const std::vector<std::string> &shaders_path)
  {
    return g_context->getShaderManager().loadProgram(name, shaders_path);
  }

  void reload_shaders()
  {
    g_context->getDescriptorSetLayouts().clear(g_context->getDevice());
    g_context->getShaderManager().reloadPrograms();
    g_context->getPipelineManager().recreate();
    g_context->getDescriptorPool().destroyAllocatedSets();
  }

  ShaderProgramInfo get_shader_program(ShaderProgramId id)
  {
    return g_context->getShaderManager().getProgramInfo(id);
  }
  
  ShaderProgramInfo get_shader_program(const std::string &name)
  {
    return g_context->getShaderManager().getProgramInfo(name);
  }

  DescriptorSet create_descriptor_set(DescriptorLayoutId layout, std::vector<Binding> bindings)
  {
    auto set = g_context->getDescriptorPool().allocateSet(layout, bindings);
    write_set(set);
    return set;
  }

  Image create_image_from_bytes(ImageCreateInfo info, SyncCommandBuffer &command_buffer, const void *data)
  {
    const auto block_size = vk::blockSize(info.format);
    const auto image_size = block_size * info.extent.width * info.extent.height * info.extent.depth;
    etna::Buffer staging_buf = g_context->createBuffer(etna::Buffer::CreateInfo
    {
      .size = image_size, 
      .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc,
      .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
      .name = "tmp_staging_buf"
    });

    auto *mapped_mem = staging_buf.map();
    memcpy(mapped_mem, data, image_size);
    staging_buf.unmap();

    info.imageUsage |= vk::ImageUsageFlagBits::eTransferDst;
    auto image = g_context->createImage(ImageCreateInfo{info});

    command_buffer.begin();
    command_buffer.copyBufferToImage(staging_buf, image, vk::ImageLayout::eTransferDstOptimal, {
      vk::BufferImageCopy {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource {image.getAspectMaskByFormat(), 0, 0, 1},
        .imageOffset {0, 0},
        .imageExtent = info.extent
      }
    });

    command_buffer.end();
    command_buffer.submit();

    etna::get_context().getQueue().waitIdle();
    staging_buf.reset();
    return image;
  }

  void flip_descriptor_pool()
  {
    g_context->getDescriptorPool().flip();
  }
}
