#include <etna/Image.hpp>
#include <etna/GlobalContext.hpp>
#include "DebugUtils.hpp"

#include <cmath>

namespace etna
{

//static vk::ImageUsageFlags 

static uint32_t mips_from_extent(uint32_t w, uint32_t h)
{
  return uint32_t(std::log2(std::max(w, h)) + 1);
}

vk::ImageUsageFlags ImageCreateInfo::imageUsageFromFmt(vk::Format format, bool linear_layout)
{
  auto physicalDevice = etna::get_context().getPhysicalDevice();
  auto properties = physicalDevice.getFormatProperties(format);
  auto features = linear_layout? properties.linearTilingFeatures : properties.optimalTilingFeatures;

  vk::ImageUsageFlags flags{};

  constexpr std::tuple<vk::FormatFeatureFlagBits, vk::ImageUsageFlagBits> usageMap[] {
    {vk::FormatFeatureFlagBits::eColorAttachment, vk::ImageUsageFlagBits::eColorAttachment},
    {vk::FormatFeatureFlagBits::eSampledImage, vk::ImageUsageFlagBits::eSampled},
    {vk::FormatFeatureFlagBits::eStorageImage, vk::ImageUsageFlagBits::eStorage},
    {vk::FormatFeatureFlagBits::eTransferSrc, vk::ImageUsageFlagBits::eTransferSrc},
    {vk::FormatFeatureFlagBits::eTransferDst, vk::ImageUsageFlagBits::eTransferDst},
    {vk::FormatFeatureFlagBits::eDepthStencilAttachment, vk::ImageUsageFlagBits::eDepthStencilAttachment}
  };

  for (auto [feature, flag] : usageMap)
    if (features & feature)
      flags |= flag;
  
  return flags;
}

vk::ImageCreateInfo ImageCreateInfo::toVkInfo() const
{
  return vk::ImageCreateInfo {
    .flags = imageFlags,
    .imageType = imageType,
    .format = format,
    .extent = extent,
    .mipLevels = mipLevels,
    .arrayLayers = arrayLayers,
    .samples = samples,
    .tiling = tiling,
    .usage = imageUsage,
    .sharingMode = vk::SharingMode::eExclusive,
    .initialLayout = vk::ImageLayout::eUndefined
  };
}

ImageCreateInfo ImageCreateInfo::colorRT(uint32_t w, uint32_t h, vk::Format fmt, std::string_view name)
{
  ImageCreateInfo info {};
  info.name = name;
  info.extent = vk::Extent3D{.width = w, .height = h, .depth = 1};
  info.format = fmt;
  info.imageUsage = imageUsageFromFmt(fmt, false);
  ETNA_ASSERT(info.imageUsage & vk::ImageUsageFlagBits::eColorAttachment);
  return info;
}

ImageCreateInfo ImageCreateInfo::depthRT(uint32_t w, uint32_t h, vk::Format fmt, std::string_view name)
{
  ImageCreateInfo info {};
  info.name = name;
  info.extent = vk::Extent3D{.width = w, .height = h, .depth = 1};
  info.format = fmt;
  info.imageUsage = imageUsageFromFmt(fmt, false);
  ETNA_ASSERT(info.imageUsage & vk::ImageUsageFlagBits::eDepthStencilAttachment);
  return info;
}

ImageCreateInfo ImageCreateInfo::image2D(uint32_t w, uint32_t h, vk::Format fmt, std::string_view name)
{
  ImageCreateInfo info {};
  info.name = name;
  info.extent = vk::Extent3D{.width = w, .height = h, .depth = 1};
  info.format = fmt;
  info.mipLevels = mips_from_extent(w, h);
  info.imageUsage = imageUsageFromFmt(fmt, false);
  return info;
}


Image::Image(VmaAllocator alloc, ImageCreateInfo &&info)
  : allocator{alloc}, imageInfo{std::move(info)}
{
  vk::ImageCreateInfo image_info = imageInfo.toVkInfo();

  VmaAllocationCreateInfo alloc_info{
    .flags = 0,
    .usage = imageInfo.memoryUsage,
    .requiredFlags = 0,
    .preferredFlags = 0,
    .memoryTypeBits = 0,
    .pool = nullptr,
    .pUserData = nullptr,
    .priority = 0.f
  };
  
  VkImage img;

  auto retcode = vmaCreateImage(allocator, &static_cast<const VkImageCreateInfo&>(image_info), &alloc_info,
      &img, &allocation, nullptr);
  // Note that usually vulkan.hpp handles doing the assertion
  // and a pretty message, but VMA cannot do that.
  ETNA_ASSERTF(retcode == VK_SUCCESS,
    "Error %s occurred while trying to allocate an etna::Image!",
    vk::to_string(static_cast<vk::Result>(retcode)));
  image = vk::Image(img);
  etna::set_debug_name(image, info.name.data());
}

Image::Image(vk::Image apiImage, ImageCreateInfo &&info)
  : allocator{nullptr}, allocation {nullptr}, image{apiImage}, imageInfo{std::move(info)}
{}

void Image::swap(Image& other)
{
  std::swap(allocator, other.allocator);
  std::swap(allocation, other.allocation);
  std::swap(image, other.image);
  std::swap(imageInfo, other.imageInfo);
}

Image::Image(Image&& other) noexcept
{
  swap(other);
}

Image& Image::operator =(Image&& other) noexcept
{
  if (this == &other)
    return *this;

  reset();
  swap(other);

  return *this;
}

void Image::reset()
{
  if (!image)
    return;
  
  views.clear();
  if (allocator && allocation)
    vmaDestroyImage(allocator, VkImage(image), allocation);
  allocator = {};
  allocation = {};
  image = vk::Image{};
  imageInfo = {};
}

Image::~Image()
{
  reset();
}

static vk::ImageAspectFlags get_aspeck_mask(vk::Format format)
{
  switch (format)
  {
  case vk::Format::eD16Unorm:
  case vk::Format::eD32Sfloat:
    return vk::ImageAspectFlagBits::eDepth;
  case vk::Format::eD16UnormS8Uint:
  case vk::Format::eD24UnormS8Uint:
  case vk::Format::eD32SfloatS8Uint:
    return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
  default:
    return vk::ImageAspectFlagBits::eColor;
  }
}

vk::ImageAspectFlags Image::getAspectMaskByFormat() const
{
  return get_aspeck_mask(imageInfo.format);
}

vk::ImageViewCreateInfo  Image::ViewParams::toVkInfo() const
{
  return vk::ImageViewCreateInfo {
    .viewType = type,
    .format = vk::Format::eUndefined,
    .subresourceRange = vk::ImageSubresourceRange {
      .aspectMask = aspect,
      .baseMipLevel = baseMip,
      .levelCount = levelCount,
      .baseArrayLayer = baseArrayLayer,
      .layerCount = layerCount
    }
  };
}

vk::UniqueImageView Image::createView(vk::ImageViewCreateInfo &&info) const
{
  info.image = image;
  if (info.format == vk::Format::eUndefined) // for format reinterpretation 
    info.format = imageInfo.format;
  if (info.subresourceRange.aspectMask == vk::ImageAspectFlags{})
    info.subresourceRange.aspectMask = get_aspeck_mask(info.format);
  auto device = etna::get_context().getDevice();
  return device.createImageViewUnique(info).value;
}

vk::ImageView Image::getView(Image::ViewParams params) const
{
  if (params.aspect == vk::ImageAspectFlags{})
    params.aspect = getAspectMaskByFormat();

  auto it = views.find(params);
  
  if (it == views.end())
  {
    auto apiView = createView(params.toVkInfo());
    it = views.emplace(params, std::move(apiView)).first;
  }

  return views[params].get();
}

ImageBinding Image::genBinding(vk::Sampler sampler, vk::ImageLayout layout, ViewParams params) const
{
  return ImageBinding{*this, vk::DescriptorImageInfo {sampler, getView(params), layout}};
}

}
