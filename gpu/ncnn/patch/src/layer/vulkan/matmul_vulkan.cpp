// Vulkan MatMul (activation x activation), elempack 1, batched over c.
// Supports A [c][M][K] (or 2D), B [c][K][N] or transB [c][N][K] (or 2D, broadcast). Needs K % 4 == 0 and N % 4 == 0.
#include "matmul_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

MatMul_vulkan::MatMul_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;

    pipeline_matmul = 0;
}

int MatMul_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = transB;

    pipeline_matmul = new Pipeline(vkdev);
    pipeline_matmul->set_local_size_xyz(8, 8, 1);
    pipeline_matmul->create(LayerShaderType::matmul, opt, specializations);

    return 0;
}

int MatMul_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_matmul;
    pipeline_matmul = 0;

    return 0;
}

int MatMul_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& A = bottom_blobs[0];
    const VkMat& B = bottom_blobs[1];
    VkMat& top_blob = top_blobs[0];

    if (A.elempack != 1 || B.elempack != 1 || A.dims < 2 || A.dims > 3 || B.dims < 2 || B.dims > 3)
    {
        NCNN_LOGE("MatMul_vulkan unsupported dims/elempack A %d/%d B %d/%d", A.dims, A.elempack, B.dims, B.elempack);
        return -1;
    }

    const int M = A.h;
    const int K = A.w;
    const int N = transB ? B.h : B.w;
    const int KB = transB ? B.w : B.h;
    const int batchA = A.dims == 3 ? A.c : 1;
    const int batchB = B.dims == 3 ? B.c : 1;
    const int batch = std::max(batchA, batchB);

    if (K != KB || K % 4 != 0 || N % 4 != 0 || (batchA != batch && batchA != 1) || (batchB != batch && batchB != 1))
    {
        NCNN_LOGE("MatMul_vulkan unsupported shape M=%d N=%d K=%d KB=%d batch %d/%d", M, N, K, KB, batchA, batchB);
        return -1;
    }

    if (A.dims == 3 || B.dims == 3)
        top_blob.create(N, M, batch, A.elemsize, opt.blob_vkallocator);
    else
        top_blob.create(N, M, A.elemsize, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = A;
    bindings[1] = B;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(6);
    constants[0].i = M;
    constants[1].i = N;
    constants[2].i = K;
    constants[3].i = batchA > 1 ? (int)A.cstep : 0;
    constants[4].i = batchB > 1 ? (int)B.cstep : 0;
    constants[5].i = top_blob.dims == 3 ? (int)top_blob.cstep : 0;

    VkMat dispatcher;
    dispatcher.w = N / 4;
    dispatcher.h = (M + 3) / 4;
    dispatcher.c = batch;

    cmd.record_pipeline(pipeline_matmul, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
