// Vulkan MatMul (activation x activation), elempack 1, batched over c. Not in upstream ncnn 20261020.
#ifndef LAYER_MATMUL_VULKAN_H
#define LAYER_MATMUL_VULKAN_H

#include "matmul.h"

namespace ncnn {

class MatMul_vulkan : public MatMul
{
public:
    MatMul_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using MatMul::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_matmul;
};

} // namespace ncnn

#endif // LAYER_MATMUL_VULKAN_H
