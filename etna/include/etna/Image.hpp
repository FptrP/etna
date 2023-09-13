#pragma once
#ifndef ETNA_IMAGE_HPP_INCLUDED
#define ETNA_IMAGE_HPP_INCLUDED

#include <etna/Vulkan.hpp>
#include <vk_mem_alloc.h>


namespace etna
{

struct ImageBinding;

struct ImageCreateInfo
{
  std::string name;
  vk::ImageType imageType {vk::ImageType::e2D};
  vk::ImageCreateFlags imageFlags{};
  vk::Format format {vk::Format::eR8G8B8A8Unorm};
  vk::Extent3D extent {1, 1, 1};
  uint32_t mipLevels = 1;
  uint32_t arrayLayers = 1;
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
  vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
  vk::ImageUsageFlags imageUsage{};
  VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;

public:
  vk::ImageCreateInfo toVkInfo() const;

  static vk::ImageUsageFlags imageUsageFromFmt(vk::Format format, bool linear_layout);

  static ImageCreateInfo colorRT(uint32_t w, uint32_t h, vk::Format fmt, std::string_view name = "");
  static ImageCreateInfo depthRT(uint32_t w, uint32_t h, vk::Format fmt, std::string_view name = "");
  static ImageCreateInfo image2D(uint32_t w, uint32_t h, vk::Format fmt, std::string_view name = ""); // color image + mips
  static ImageCreateInfo imageCube(uint32_t size, vk::Format fmt, std::string_view name = "");
  // TODO: 
  static ImageCreateInfo imageArray(uint32_t w, uint32_t h, vk::Format fmt, 
    uint32_t layers, uint32_t levels, std::string_view name = "");
  
  static ImageCreateInfo image3D(uint32_t w, uint32_t h, uint32_t d, vk::Format fmt, std::string_view name = "");

  static ImageCreateInfo colorRT_MSAA(uint32_t w, uint32_t h, vk::Format fmt, 
    vk::SampleCountFlagBits samples, std::string_view name = "");

  static ImageCreateInfo depthRT_MSAA(uint32_t w, uint32_t h, vk::Format fmt,
    vk::SampleCountFlagBits samples, std::string_view name = "");
};

struct Image;

struct ImageView
{
  ImageView(const Image &img, vk::ImageSubresourceRange range_, vk::ImageView view_)
    : owner {img}, range{range_}, view{view_} {}

  explicit operator vk::ImageView() const
  {
    return view;
  }
  
  vk::ImageSubresourceRange getRange() const
  {
    return range;
  }

  const Image &getOwner() const
  {
    return owner;
  }

private:
  const Image &owner; // shared_ptr<> when :(
  vk::ImageSubresourceRange range;
  vk::ImageView view;
};

class Image
{
public:

  Image(VmaAllocator alloc, ImageCreateInfo &&info);
  Image(vk::Image apiImage, ImageCreateInfo &&info); // for swapchain images

  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;

  void swap(Image& other);
  Image(Image&&) noexcept;
  Image& operator=(Image&&) noexcept;

  [[nodiscard]] vk::Image get() const { return image; }

  ~Image();
  void reset();

  // Missed view features: format reintepretation  
  struct ViewParams 
  {
    ViewParams() {}

    vk::ImageViewType type {vk::ImageViewType::e2D};
    vk::ImageAspectFlags aspect{}; // if empty then select default for image
    uint32_t baseMip = 0;
    uint32_t levelCount = 1; 
    uint32_t baseArrayLayer = 0;
    uint32_t layerCount = 1;

    bool operator==(const ViewParams& b) const = default;
    vk::ImageViewCreateInfo toVkInfo() const;
  };

  vk::UniqueImageView createView(vk::ImageViewCreateInfo &&info) const; // For missed features
  ImageView getView(ViewParams params) const;

  ImageBinding genBinding(vk::Sampler sampler, vk::ImageLayout layout, 
    ViewParams params = ViewParams{}) const;

  vk::ImageAspectFlags getAspectMaskByFormat() const;

  const ImageCreateInfo &getInfo() const { return imageInfo; }

private:
  struct ViewParamsHasher
  {
    size_t operator()(ViewParams params) const
    {
      uint32_t hash = 0;
      hashPack(hash, uint32_t(params.type), uint32_t(params.aspect), 
        params.baseMip, params.levelCount, params.baseArrayLayer, params.layerCount);
      return hash;
    }
  private:
    template<typename HashT, typename... HashTs>
    inline void hashPack(uint32_t& hash, const HashT& first, HashTs&&... other) const
    {
      auto hasher = std::hash<uint32_t>();
      hash ^= hasher(first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      (hashPack(hash, std::forward<HashTs>(other)), ...);
    }
  };

  mutable std::unordered_map<
    ViewParams, 
    std::tuple<vk::ImageSubresourceRange, vk::UniqueImageView>,
    ViewParamsHasher>  views;
  
  VmaAllocator allocator{};

  VmaAllocation allocation{};
  vk::Image image{}; //todo: add extent, layers, mips
  ImageCreateInfo imageInfo {};
};

}

#endif // ETNA_IMAGE_HPP_INCLUDED
