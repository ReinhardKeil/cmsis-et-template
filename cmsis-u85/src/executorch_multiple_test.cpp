// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "TinyCNNMultiple_pte.h"

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

namespace {

// Both methods stay loaded; their planned buffers have independent lifetimes.
constexpr size_t kMethodPoolSize = 256 * 1024;
constexpr size_t kTempPoolSize = 64 * 1024;

// The scratch pool must be in memory accessible to the NPU.
#ifdef APP_POOL_SECTION
#define MULTIPLE_POOL_ATTRIBUTES __attribute__((section(APP_POOL_SECTION)))
#else
#define MULTIPLE_POOL_ATTRIBUTES
#endif
alignas(64) uint8_t g_method_pool[kMethodPoolSize] MULTIPLE_POOL_ATTRIBUTES;
alignas(64) uint8_t g_temp_pool[kTempPoolSize] MULTIPLE_POOL_ATTRIBUTES;

// quantize_multiple_models.py, seed 0, uniform input 0.25. Allow a few INT8
// rounding steps across CMSIS-NN and Ethos-U implementations.
constexpr std::array<float, 10> kReference = {
  -0.0255408380f, 0.0706129074f, 0.0540864803f, -0.0525840782f,
   0.0390624590f, 0.0270432401f, -0.0360576548f, -0.0555888824f,
  -0.1397234052f, 0.0015024022f,
};
constexpr float kTolerance = 0.006f;

bool fail(const char *operation)
{
  printf("ExecuTorch multiple: %s failed\nTest_result: FAIL\n", operation);
  return false;
}

bool read_integer(EmbeddedModule &module, const char *name, SizesType &value)
{
  auto values = module.execute(name, std::vector<EValue>{});
  if (!values.ok() || values->size() != 1 || !values->front().isInt() ||
      values->front().toInt() <= 0 ||
      values->front().toInt() > std::numeric_limits<SizesType>::max()) {
    return fail(name);
  }
  value = static_cast<SizesType>(values->front().toInt());
  printf("%s: %ld\n", name, static_cast<long>(value));
  return true;
}

template <class Sizes, class Strides>
size_t physical_offset(size_t logical, const Sizes &sizes, const Strides &strides)
{
  size_t offset = 0;
  for (size_t dimension = sizes.size(); dimension-- > 0;) {
    offset += (logical % sizes[dimension]) * strides[dimension];
    logical /= sizes[dimension];
  }
  return offset;
}

template <size_t OutputRank>
bool run_method(EmbeddedModule &module, const char *method,
                const std::array<SizesType, 4> &image_shape, const std::vector<float> &input,
                const std::array<SizesType, OutputRank> &output_shape, std::vector<float> &output)
{
  auto metadata = module.method_meta(method);
  if (!metadata.ok() || metadata->num_inputs() != 1 || metadata->num_outputs() != 1) {
    return fail("method metadata");
  }
  auto tensor_metadata = metadata->input_tensor_meta(0);
  auto output_metadata = metadata->output_tensor_meta(0);
  if (!tensor_metadata.ok() || !output_metadata.ok() ||
      tensor_metadata->scalar_type() != ScalarType::Float ||
      output_metadata->scalar_type() != ScalarType::Float ||
      tensor_metadata->sizes().size() != image_shape.size() ||
      tensor_metadata->dim_order().size() != image_shape.size() ||
      output_metadata->sizes().size() != output_shape.size() ||
      tensor_metadata->nbytes() != input.size() * sizeof(float) ||
      output_metadata->nbytes() != output.size() * sizeof(float)) {
    return fail("tensor metadata");
  }
  auto sizes = image_shape;
  std::array<DimOrderType, 4> order{};
  std::array<size_t, 4> strides{};
  for (size_t i = 0; i < image_shape.size(); ++i) {
    if (tensor_metadata->sizes()[i] != image_shape[i]) {
      return fail("input shape");
    }
    order[i] = tensor_metadata->dim_order()[i];
  }
  for (size_t i = 0; i < output_shape.size(); ++i) {
    if (output_metadata->sizes()[i] != output_shape[i]) {
      return fail("output shape");
    }
  }
  size_t stride = 1;
  for (size_t i = image_shape.size(); i-- > 0;) {
    strides[order[i]] = stride;
    stride *= image_shape[order[i]];
  }
  // Pipeline buffers use logical contiguous order. Import the layout expected
  // by the method, which can differ between CPU and NPU lowering.
  std::vector<float> physical_input(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    physical_input[physical_offset(i, image_shape, strides)] = input[i];
  }
  TensorImpl implementation(ScalarType::Float, sizes.size(), sizes.data(),
                            physical_input.data(), order.data());
  auto results = module.execute(method, EValue(Tensor(&implementation)));
  if (!results.ok() || results->size() != 1 || !results->front().isTensor()) {
    return fail(method);
  }
  const auto tensor = results->front().toTensor();
  if (tensor.scalar_type() != ScalarType::Float ||
      static_cast<size_t>(tensor.numel()) != output.size()) {
    return fail("output tensor");
  }
  // Copy before executing another method: output storage belongs to the method.
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = tensor.const_data_ptr<float>()[physical_offset(i, output_shape, tensor.strides())];
  }
  return true;
}

} // namespace

extern "C" int executorch_run_test(void)
{
  executorch::runtime::runtime_init();
  printf("ExecuTorch filter/classify application: %lu byte program\n", TinyCNNMultiple_pte_size);
  // NOLOAD scratch sections are not zeroed at boot.
  std::memset(g_temp_pool, 0, sizeof(g_temp_pool));
  EmbeddedModule module(
      TinyCNNMultiple_pte, TinyCNNMultiple_pte_size,
      std::make_unique<BufferDataLoader>(TinyCNNMultiple_pte, TinyCNNMultiple_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));
  if (module.load() != executorch::runtime::Error::Ok) {
    fail("load");
    return 1;
  }

  SizesType image_width, image_height, input_channels, num_classes;
  if (!read_integer(module, "IMAGE_WIDTH", image_width) ||
      !read_integer(module, "IMAGE_HEIGHT", image_height) ||
      !read_integer(module, "INPUT_CHANNELS", input_channels) ||
      !read_integer(module, "NUM_CLASSES", num_classes)) {
    return 1;
  }
  // One NHWC image for inference; filtering preserves its dimensions.
  const std::array<SizesType, 4> image_shape = {1, image_height, image_width, input_channels};
  const std::array<SizesType, 2> output_shape = {1, num_classes};
  size_t image_elements = 1;
  for (auto dimension : image_shape) {
    if (static_cast<size_t>(dimension) >
        std::numeric_limits<size_t>::max() / sizeof(float) / image_elements) {
      fail("image size");
      return 1;
    }
    image_elements *= dimension;
  }
  if (static_cast<size_t>(num_classes) != kReference.size()) {
    fail("reference size");
    return 1;
  }

  std::vector<float> image(image_elements, 0.25f);
  std::vector<float> filtered_image(image_elements);
  std::vector<float> logits(num_classes);
  if (!run_method(module, "filter", image_shape, image, image_shape, filtered_image) ||
      !run_method(module, "classify", image_shape, filtered_image, output_shape, logits)) {
    return 1;
  }

  int result = 0;
  printf("Output:");
  for (size_t i = 0; i < logits.size(); ++i) {
    printf(" %.6f", static_cast<double>(logits[i]));
    if (!std::isfinite(logits[i]) || std::fabs(logits[i] - kReference[i]) > kTolerance) {
      result = 1;
    }
  }
  printf("\nReference comparison (absolute tolerance %.3f): %s\n",
         static_cast<double>(kTolerance), result ? "FAIL" : "PASS");
  printf("Test_result: %s\n", result ? "FAIL" : "PASS");
  fflush(stdout);
  return result;
}
