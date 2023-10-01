#pragma once
#ifndef ETNA_SUBMIT_CONTEXT_HPP_INCLUDED
#define ETNA_SUBMIT_CONTEXT_HPP_INCLUDED

#include <etna/Image.hpp>
#include <etna/SyncCommandBuffer.hpp>

#include <memory>
#include <variant>

namespace etna
{

  enum class SwapchainState {
    Ok,
    Suboptimal,
    OutOfDate
  }; 

  struct SimpleSubmitContext
  {
    SimpleSubmitContext(const SimpleSubmitContext &) = delete;
    SimpleSubmitContext &operator=(const SimpleSubmitContext &) = delete;
    SimpleSubmitContext(SimpleSubmitContext &&) = delete;
    SimpleSubmitContext &operator=(SimpleSubmitContext &&) = delete;
    
    ~SimpleSubmitContext();

    SyncCommandBuffer &acquireNextCmd();    

    SwapchainState submitCmd(SyncCommandBuffer &cmd, bool present);
    std::tuple<Image*, SwapchainState> acquireBackbuffer(); // image is nullptr if SwapchainState is OutOfDate
    
    vk::Extent2D recreateSwapchain(vk::Extent2D resolution); // 1) synchoronization required
                                                     // 2) resources that depend on swapchain images
                                                     // (framebuffers, imageViews) must be destroyed


    uint32_t getBackbuffersCount() const { return swapchainImages.size(); }
    uint32_t getFramesInFlight() const { return commandBuffers.size(); }
    vk::Format getSwapchainFmt() const { return swapchainFormat; }
    
    CommandBufferPool &getCommandPool() { return commandPool; }

  private:
    vk::UniqueSurfaceKHR surface;
    vk::UniqueSwapchainKHR swapchain;
    vk::Format swapchainFormat;

    std::vector<Image> swapchainImages;
    std::vector<vk::UniqueSemaphore> imageAcquireSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;

    std::optional<uint32_t> currentBackbuffer {};
    uint32_t semaphoreIndex = 0u;

    //vk::UniqueCommandPool commandPool;
    CommandBufferPool commandPool;
    std::vector<SyncCommandBuffer> commandBuffers;
    std::vector<vk::UniqueFence> cmdReadyFences;

    uint32_t cmdIndex = 0;    
    bool cmdAcquired = false;

    SimpleSubmitContext() {}

    static std::unique_ptr<SimpleSubmitContext> createEmpty()
    {
      return std::unique_ptr<SimpleSubmitContext>{new SimpleSubmitContext{}};
    }

    friend std::unique_ptr<SimpleSubmitContext> create_submit_context(
      vk::SurfaceKHR surface, 
      vk::Extent2D windowSize,
      bool force_srgb
    );
  };

  std::unique_ptr<SimpleSubmitContext> create_submit_context(
    vk::SurfaceKHR surface, vk::Extent2D windowSize, bool force_srgb);

}


#endif