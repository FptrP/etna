#include "etna/SubmitContext.hpp"
#include "etna/GlobalContext.hpp"
#include "etna/Etna.hpp"

namespace etna
{
  struct SwapchainParams
  {
    vk::Format imageFmt;
    vk::ColorSpaceKHR colorSpace;
    vk::SurfaceCapabilitiesKHR params;
  };

  static std::optional<SwapchainParams> query_swapchain_support(vk::SurfaceKHR surface)
  {
    auto physicalDevice = etna::get_context().getPhysicalDevice();
    auto [status, surfaceSupport] = physicalDevice.getSurfaceSupportKHR(
      etna::get_context().getQueueFamilyIdx(), surface);

    if (status != vk::Result::eSuccess || surfaceSupport != VK_TRUE)
      return {};

    auto caps = physicalDevice.getSurfaceCapabilitiesKHR(surface).value;
    auto supportedFormats = physicalDevice.getSurfaceFormatsKHR(surface).value;

    ETNA_ASSERT(supportedFormats.size() > 0);

    return SwapchainParams {
      .imageFmt = supportedFormats.at(0).format,
      .colorSpace = supportedFormats.at(0).colorSpace,
      .params = caps
    };
  }

  static auto create_swapchain(vk::SurfaceKHR surface, const SwapchainParams &params)
  {
    auto device = etna::get_context().getDevice();
    
    vk::SwapchainCreateInfoKHR info {
      .surface = surface,
      .minImageCount = params.params.minImageCount,
      .imageFormat = params.imageFmt,
      .imageColorSpace = params.colorSpace,
      .imageExtent = params.params.currentExtent,
      .imageArrayLayers = 1,
      .imageUsage = params.params.supportedUsageFlags,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = params.params.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = vk::PresentModeKHR::eFifo // always supported. TODO: make option for vsync=false
    };

    return device.createSwapchainKHRUnique(info);
  }

  std::vector<Image> get_swapchain_images(vk::SwapchainKHR swapchain, const SwapchainParams &sparams)
  {
    
    auto device = etna::get_context().getDevice();
    auto [result, apiImages] = device.getSwapchainImagesKHR(swapchain);
    
    ETNA_ASSERT(result == vk::Result::eSuccess);

    std::vector<etna::Image> etnaImages;
    etnaImages.reserve(apiImages.size());

    for (auto image : apiImages) {
      auto imageInfo = ImageCreateInfo::colorRT(
        sparams.params.currentExtent.width, 
        sparams.params.currentExtent.height,
        sparams.imageFmt,
        "swapchain_image");

      imageInfo.imageUsage = sparams.params.supportedUsageFlags;
      etnaImages.push_back(Image {image, std::move(imageInfo)}); // proxy texture
    }
    return etnaImages;
  }

  static vk::UniqueSemaphore create_binary_semaphore()
  {
    vk::SemaphoreCreateInfo info {};
    auto device = etna::get_context().getDevice();
    return device.createSemaphoreUnique(info).value;
  }
  
  static vk::UniqueFence create_fence(bool signaled)
  {
    vk::FenceCreateInfo info {};
    if (signaled)
      info.flags = vk::FenceCreateFlagBits::eSignaled;
    auto device = etna::get_context().getDevice();
    return device.createFenceUnique(info).value;
  }

  static vk::UniqueCommandPool create_command_pool()
  {
    vk::CommandPoolCreateInfo info {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = etna::get_context().getQueueFamilyIdx()
    };

    return etna::get_context().getDevice().createCommandPoolUnique(info).value;
  }

  static std::vector<vk::UniqueCommandBuffer> allocate_main_cmd(vk::CommandPool pool, uint32_t count)
  {
    vk::CommandBufferAllocateInfo info {
      .commandPool = pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = count
    };

    auto device = etna::get_context().getDevice();
    return device.allocateCommandBuffersUnique(info).value;
  }

  static constexpr SwapchainState from_result(vk::Result result)
  {
    switch (result)
    {
      case vk::Result::eSuboptimalKHR:
        return SwapchainState::Suboptimal;
      case vk::Result::eErrorOutOfDateKHR:
        return SwapchainState::OutOfDate;
      default:
        return SwapchainState::Ok;
    }
    return SwapchainState::Ok;
  }

  SimpleSubmitContext::~SimpleSubmitContext()
  {
    etna::get_context().getDevice().waitIdle();
  }

  std::unique_ptr<SimpleSubmitContext> create_submit_context(vk::SurfaceKHR surface, vk::Extent2D windowSize)
  {
    auto swapchainInfo = query_swapchain_support(surface);
    ETNA_ASSERTF(swapchainInfo, "Vulkan device does not support swapchain");
    
    // with wayland window is not displayed until first draw and currentExtent is zero
    if (!swapchainInfo->params.currentExtent.width || swapchainInfo->params.currentExtent.height)
      swapchainInfo->params.currentExtent = windowSize;

    auto [status, swapchain] = create_swapchain(surface, *swapchainInfo);
    ETNA_ASSERTF(status == vk::Result::eSuccess, "Swapchain create error");

    auto swapchainImages = get_swapchain_images(*swapchain, *swapchainInfo); 
    
    std::vector<vk::UniqueSemaphore> acquireSem;    
    std::vector<vk::UniqueSemaphore> submitSem;

    acquireSem.reserve(swapchainImages.size());
    submitSem.reserve(swapchainImages.size());
    
    for (uint32_t i = 0; i < swapchainImages.size(); i++)
    {
      acquireSem.emplace_back(create_binary_semaphore());
      submitSem.emplace_back(create_binary_semaphore());
    }

    const uint32_t framesInFlight = etna::get_context().getNumFramesInFlight();
    
    //auto cmdPool = create_command_pool();
    //std::vector<vk::UniqueCommandBuffer> cmdBuffers = allocate_main_cmd(*cmdPool, framesInFlight); 
    
    //std::vector<SyncCommandBuffer> syncCmd;
    //syncCmd.reserve(cmdBuffers.size());
    //for (auto &elem : cmdBuffers)
    //  syncCmd.emplace_back(std::move(elem));

    std::vector<vk::UniqueFence> cmdFence;
    cmdFence.reserve(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; i++)
      cmdFence.emplace_back(create_fence(true));

    auto ctx = SimpleSubmitContext::createEmpty();
    ctx->surface = vk::UniqueSurfaceKHR{surface, etna::get_context().getInstance()};
    ctx->swapchain = std::move(swapchain);
    ctx->swapchainFormat = swapchainInfo->imageFmt;
    ctx->swapchainImages = std::move(swapchainImages);
    ctx->imageAcquireSemaphores = std::move(acquireSem);
    ctx->renderFinishedSemaphores = std::move(submitSem);
    //ctx->commandPool = std::move(cmdPool);
    //ctx->commandBuffers = std::move(syncCmd);
    ctx->cmdReadyFences = std::move(cmdFence);

    std::vector<SyncCommandBuffer> syncCmd;
    syncCmd.reserve(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; i++)
      syncCmd.emplace_back(ctx->commandPool);

    ctx->commandBuffers = std::move(syncCmd);

    return ctx; 
  }

  
  SyncCommandBuffer &SimpleSubmitContext::acquireNextCmd()
  {
    ETNA_ASSERTF(!cmdAcquired, \
      "command buffer is already acquired. Submit it before acquiring next");
    
    auto device = etna::get_context().getDevice();
    device.waitForFences({*cmdReadyFences[cmdIndex]}, VK_TRUE, ~0ull);
    
    etna::flip_descriptor_pool();

    auto &cmdBuffer = commandBuffers[cmdIndex];  

    cmdBuffer.reset();
    
    device.resetFences({*cmdReadyFences[cmdIndex]});
    cmdAcquired = true;
    
    return cmdBuffer;
  }   
    
  SwapchainState SimpleSubmitContext::submitCmd(SyncCommandBuffer &cmd, bool present)
  {
    ETNA_ASSERTF(!present || currentBackbuffer.has_value(), \
      "Presentation is requested, but backbuffer is not acquired");
    ETNA_ASSERT(cmd.get() == commandBuffers[cmdIndex].get());
    
    SubmitInfo submitInfo {
      .waitSemaphores = {*imageAcquireSemaphores[semaphoreIndex]},
      .waitDstStageMask = {vk::PipelineStageFlagBits::eAllCommands},
      .signalSemaphores = {*renderFinishedSemaphores[semaphoreIndex]}
    };

    if (present)
    {
      auto res = commandBuffers[cmdIndex].submit(submitInfo, *cmdReadyFences[cmdIndex]);
      ETNA_ASSERT(res == vk::Result::eSuccess);
    } else 
    {
      auto res = commandBuffers[cmdIndex].submit(*cmdReadyFences[cmdIndex]);
      ETNA_ASSERT(res == vk::Result::eSuccess);
    }
    
    cmdAcquired = false;
    cmdIndex = (cmdIndex + 1) % getFramesInFlight();

    if (present)
    {
      uint32_t index = *currentBackbuffer; 

      vk::PresentInfoKHR presentInfo {
        .swapchainCount = 1,
        .pSwapchains = &swapchain.get(),
        .pImageIndices = &index      
      };

      presentInfo.setWaitSemaphores(submitInfo.signalSemaphores);
      VkPresentInfoKHR cInfo = presentInfo;

      auto queue = etna::get_context().getQueue();
      
      /*queue.presentKHR asserts on OutOfDate swapchain*/
      auto result = vk::Result(VULKAN_HPP_DEFAULT_DISPATCHER.vkQueuePresentKHR(queue, &cInfo));
      currentBackbuffer = {};
      semaphoreIndex = (semaphoreIndex + 1) % getBackbuffersCount();
      return from_result(result); //if we are OutOfDate, then state might be completely broken
    }

    return SwapchainState::Ok;
  }

  std::tuple<Image*, SwapchainState> SimpleSubmitContext::acquireBackbuffer()
  {
    ETNA_ASSERTF(!currentBackbuffer.has_value(), \
      "Backbuffer is already acquired");
    
    auto device = etna::get_context().getDevice();
    auto [status, imageIndex] = device.acquireNextImageKHR(
      *swapchain, ~0ull, *imageAcquireSemaphores[semaphoreIndex]);

    SwapchainState state = from_result(status);
    
    if (state != SwapchainState::OutOfDate)
    {
      currentBackbuffer = imageIndex; 
      return {&swapchainImages[*currentBackbuffer], state};
    }

    return {nullptr, state};
  }
    
  void SimpleSubmitContext::recreateSwapchain(vk::Extent2D resolution)
  {
    swapchain.reset();
    swapchainImages.clear();

    if (currentBackbuffer) //Suboptimal on acquire. Maybe add warning if cmdAcquired?
      currentBackbuffer = {};

    auto swapchainInfo = query_swapchain_support(*surface);
    ETNA_ASSERT(swapchainInfo.has_value());
    
    if (!swapchainInfo->params.currentExtent.width && !swapchainInfo->params.currentExtent.height)
      swapchainInfo->params.currentExtent = resolution;

    auto [status, newSwapchain] = create_swapchain(*surface, *swapchainInfo);
    ETNA_ASSERTF(status == vk::Result::eSuccess, "Swapchain create error");

    swapchain = std::move(newSwapchain);
    swapchainImages = get_swapchain_images(*swapchain, *swapchainInfo); 
    swapchainFormat = swapchainInfo->imageFmt;
    ETNA_ASSERT(swapchainImages.size() == imageAcquireSemaphores.size()); //Recreate semaphores?
  }

}