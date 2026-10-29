/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
#include "boundary_bucket_sort.h"
#include "packed_column_reindex.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <cstring>
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

inline bool is_npu(const at::Tensor& tensor)
{
#ifdef COMPILE_WITH_XLA
    return tensor.device().type() == at::kXLA;
#else
    return tensor.device().type() == at::kPrivateUse1;
#endif
}
#define TORCH_CHECK_NPU(tensor) TORCH_CHECK(is_npu(tensor), #tensor " must be NPU tensor")

namespace {
constexpr int64_t kSpmmHalfMaxDenseWidth = 1024;
constexpr int64_t kSpmmFloatMaxDenseWidth = 512;

void check_dense_contract(const at::Tensor& dense, int64_t alignment, const char* op_name)
{
    TORCH_CHECK(dense.scalar_type() == at::kHalf || dense.scalar_type() == at::kFloat,
                op_name, " supports only FP16 and FP32 dense tensors");
    TORCH_CHECK(dense.is_contiguous(), op_name, " requires a contiguous dense tensor");
    const int64_t width = dense.size(1);
    const bool is_half = dense.scalar_type() == at::kHalf;
    const int64_t max_width = is_half
        ? kSpmmHalfMaxDenseWidth
        : kSpmmFloatMaxDenseWidth;
    const char* dtype_name = is_half ? "FP16" : "FP32";
    TORCH_CHECK(width > 0 && width <= max_width,
                op_name, " requires 0 < feature width <= ", max_width,
                " for ", dtype_name, ", but got ", width);
    TORCH_CHECK(width % alignment == 0,
                op_name, " requires feature width divisible by ", alignment,
                ", but got ", width,
                ". Pad and crop the dense tensor or use a safe PyTorch fallback.");
}

void check_value_contract(const at::Tensor& value, const at::Tensor& dense, const char* name)
{
    TORCH_CHECK(value.scalar_type() == dense.scalar_type(), name,
                " must have the same dtype as dense");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}
}

at::Tensor spmm_csr_impl_npu(const at::Tensor& rowptr, const at::Tensor& col, const at::Tensor& value, const at::Tensor& dense, const at::Tensor& start) {
    TORCH_CHECK_NPU(rowptr);
    TORCH_CHECK_NPU(col);
    TORCH_CHECK_NPU(value);
    TORCH_CHECK_NPU(dense);
    TORCH_CHECK_NPU(start);
    TORCH_CHECK(rowptr.dim() == 1, "row must be a 1D Tensor, but got: ", rowptr.dim());
    TORCH_CHECK(col.dim() == 1, "col must be a 1D Tensor, but got: ", col.dim());
    TORCH_CHECK(value.dim() == 1, "value must be a 1D Tensor, but got: ", value.dim());
    TORCH_CHECK(dense.dim() == 2, "dense must be a 2D Tensor, but got: ", dense.dim());
    TORCH_CHECK(start.dim() == 1, "start must be a 1D Tensor, but got: ", start.dim());
    check_dense_contract(dense, dense.scalar_type() == at::kHalf ? 16 : 8, "spmm_csr");
    check_value_contract(value, dense, "value");

    size_t N = rowptr.size(0) - 1, M = dense.size(0), D = dense.size(1), nnz1 = col.size(0), nnz2 = value.size(0);
    TORCH_CHECK(nnz1 == nnz2, "col and value must be same size, but got different: ", nnz1 - nnz2);

    auto result = at::zeros({N, D}, dense.options().dtype(dense.dtype()));

    EXEC_NPU_CMD(aclnnSpmmCsr, rowptr, col, value, dense, start, result);
    return result;
}


at::Tensor spmm_cube_impl_npu(const at::Tensor& rowptr, const at::Tensor& col, const at::Tensor& value, const at::Tensor& dense, const at::Tensor& start) {
    TORCH_CHECK_NPU(rowptr);
    TORCH_CHECK_NPU(col);
    TORCH_CHECK_NPU(value);
    TORCH_CHECK_NPU(dense);
    TORCH_CHECK_NPU(start);
    TORCH_CHECK(rowptr.dim() == 1, "row must be a 1D Tensor, but got: ", rowptr.dim());
    TORCH_CHECK(col.dim() == 1, "col must be a 1D Tensor, but got: ", col.dim());
    TORCH_CHECK(value.dim() == 1, "value must be a 1D Tensor, but got: ", value.dim());
    TORCH_CHECK(dense.dim() == 2, "dense must be a 2D Tensor, but got: ", dense.dim());
    TORCH_CHECK(start.dim() == 1, "start must be a 1D Tensor, but got: ", start.dim());
    check_dense_contract(dense, 16, "spmm_cube");
    check_value_contract(value, dense, "value");

    size_t N = rowptr.size(0) - 1, M = dense.size(0), D = dense.size(1), nnz1 = col.size(0), nnz2 = value.size(0);
    TORCH_CHECK(nnz1 == nnz2, "col and value must be same size, but got different: ", nnz1 - nnz2);

    auto result = at::zeros({N * 16, D}, dense.options().dtype(dense.dtype()));
    auto tempspace = at::zeros({512 * 20}, dense.options().dtype(dense.dtype()));

    EXEC_NPU_CMD(aclnnSpmmCube, rowptr, col, value, dense, start, tempspace, result);
    return result;
}

at::Tensor spmm_ascend_cube_impl_npu(const at::Tensor& Crowptr,
                                    const at::Tensor& Ccol_ids,
                                    const at::Tensor& Cvalue,
                                    const at::Tensor& Vrowptr,
                                    const at::Tensor& Vcol_ids,
                                    const at::Tensor& Vvalue,
                                    const at::Tensor& dense,
                                    const at::Tensor& Cstart,
                                    const at::Tensor& Vstart) {
    TORCH_CHECK_NPU(Crowptr);
    TORCH_CHECK_NPU(Ccol_ids);
    TORCH_CHECK_NPU(Cvalue);
    TORCH_CHECK_NPU(Vrowptr);
    TORCH_CHECK_NPU(Vcol_ids);
    TORCH_CHECK_NPU(Vvalue);
    TORCH_CHECK_NPU(dense);
    TORCH_CHECK_NPU(Cstart);
    TORCH_CHECK_NPU(Vstart);
    TORCH_CHECK(Crowptr.dim() == 1, "Crow must be a 1D Tensor, but got: ", Crowptr.dim());
    TORCH_CHECK(Ccol_ids.dim() == 1, "Ccol_ids must be a 1D Tensor, but got: ", Ccol_ids.dim());
    TORCH_CHECK(Cvalue.dim() == 1, "Cvalue must be a 1D Tensor, but got: ", Cvalue.dim());
    TORCH_CHECK(Vrowptr.dim() == 1, "Vrow must be a 1D Tensor, but got: ", Vrowptr.dim());
    TORCH_CHECK(Vcol_ids.dim() == 1, "Vcol_ids must be a 1D Tensor, but got: ", Vcol_ids.dim());
    TORCH_CHECK(Vvalue.dim() == 1, "Vvalue must be a 1D Tensor, but got: ", Vvalue.dim());
    TORCH_CHECK(dense.dim() == 2, "dense must be a 2D Tensor, but got: ", dense.dim());
    TORCH_CHECK(Cstart.dim() == 1, "Cstart must be a 1D Tensor, but got: ", Cstart.dim());
    TORCH_CHECK(Vstart.dim() == 1, "Vstart must be a 1D Tensor, but got: ", Vstart.dim());
    check_dense_contract(dense, 16, "spmm_ascend_cube");
    check_value_contract(Cvalue, dense, "Cvalue");
    check_value_contract(Vvalue, dense, "Vvalue");

    size_t N = std::max((Crowptr.size(0) - 1) * 16, Vrowptr.size(0) - 1), M = dense.size(0), D = dense.size(1);

    auto result = at::zeros({N, D}, dense.options().dtype(dense.dtype()));

    EXEC_NPU_CMD(aclnnSpmmAscendCube, Crowptr, Ccol_ids, Cvalue, Vrowptr, Vcol_ids, Vvalue, dense, Cstart, Vstart, result);
    return result;
}

at::Tensor prepare_spmm_csr_ablation_result(
        const at::Tensor& rowptr,
        const at::Tensor& col,
        const at::Tensor& value,
        const at::Tensor& dense,
        const at::Tensor& start,
        const char* op_name) {
    TORCH_CHECK_NPU(rowptr);
    TORCH_CHECK_NPU(col);
    TORCH_CHECK_NPU(value);
    TORCH_CHECK_NPU(dense);
    TORCH_CHECK_NPU(start);
    TORCH_CHECK(rowptr.dim() == 1, op_name, ": rowptr must be one-dimensional");
    TORCH_CHECK(col.dim() == 1, op_name, ": col must be one-dimensional");
    TORCH_CHECK(value.dim() == 1, op_name, ": value must be one-dimensional");
    TORCH_CHECK(dense.dim() == 2, op_name, ": dense must be two-dimensional");
    TORCH_CHECK(start.dim() == 1, op_name, ": start must be one-dimensional");
    check_dense_contract(dense, dense.scalar_type() == at::kHalf ? 16 : 8, op_name);
    check_value_contract(value, dense, "value");
    TORCH_CHECK(col.size(0) == value.size(0), op_name, ": col and value must have equal length");
    return at::zeros({rowptr.size(0) - 1, dense.size(1)}, dense.options().dtype(dense.dtype()));
}

at::Tensor spmm_csr_aiv_mb_batch_impl_npu(
        const at::Tensor& rowptr,
        const at::Tensor& col,
        const at::Tensor& value,
        const at::Tensor& dense,
        const at::Tensor& start) {
    auto result = prepare_spmm_csr_ablation_result(
        rowptr, col, value, dense, start, "spmm_csr_aiv_mb_batch");
    EXEC_NPU_CMD(aclnnSpmmCsrAivMbBatch, rowptr, col, value, dense, start, result);
    return result;
}

at::Tensor prepare_spmm_mixed_ablation_result(
        const at::Tensor& Crowptr,
        const at::Tensor& Ccol_ids,
        const at::Tensor& Cvalue,
        const at::Tensor& Vrowptr,
        const at::Tensor& Vcol_ids,
        const at::Tensor& Vvalue,
        const at::Tensor& dense,
        const at::Tensor& Cstart,
        const at::Tensor& Vstart,
        const char* op_name) {
    TORCH_CHECK_NPU(Crowptr);
    TORCH_CHECK_NPU(Ccol_ids);
    TORCH_CHECK_NPU(Cvalue);
    TORCH_CHECK_NPU(Vrowptr);
    TORCH_CHECK_NPU(Vcol_ids);
    TORCH_CHECK_NPU(Vvalue);
    TORCH_CHECK_NPU(dense);
    TORCH_CHECK_NPU(Cstart);
    TORCH_CHECK_NPU(Vstart);
    TORCH_CHECK(Crowptr.dim() == 1, op_name, ": Crowptr must be one-dimensional");
    TORCH_CHECK(Ccol_ids.dim() == 1, op_name, ": Ccol_ids must be one-dimensional");
    TORCH_CHECK(Cvalue.dim() == 1, op_name, ": Cvalue must be one-dimensional");
    TORCH_CHECK(Vrowptr.dim() == 1, op_name, ": Vrowptr must be one-dimensional");
    TORCH_CHECK(Vcol_ids.dim() == 1, op_name, ": Vcol_ids must be one-dimensional");
    TORCH_CHECK(Vvalue.dim() == 1, op_name, ": Vvalue must be one-dimensional");
    TORCH_CHECK(dense.dim() == 2, op_name, ": dense must be two-dimensional");
    TORCH_CHECK(Cstart.dim() == 1, op_name, ": Cstart must be one-dimensional");
    TORCH_CHECK(Vstart.dim() == 1, op_name, ": Vstart must be one-dimensional");
    check_dense_contract(dense, 16, op_name);
    check_value_contract(Cvalue, dense, "Cvalue");
    check_value_contract(Vvalue, dense, "Vvalue");
    const auto rows = std::max(
        (Crowptr.size(0) - 1) * 16,
        Vrowptr.size(0) - 1
    );
    return at::zeros({rows, dense.size(1)}, dense.options().dtype(dense.dtype()));
}

at::Tensor spmm_ascend_cube_aic_without_pipeline_impl_npu(
        const at::Tensor& Crowptr,
        const at::Tensor& Ccol_ids,
        const at::Tensor& Cvalue,
        const at::Tensor& Vrowptr,
        const at::Tensor& Vcol_ids,
        const at::Tensor& Vvalue,
        const at::Tensor& dense,
        const at::Tensor& Cstart,
        const at::Tensor& Vstart) {
    auto result = prepare_spmm_mixed_ablation_result(
        Crowptr, Ccol_ids, Cvalue, Vrowptr, Vcol_ids, Vvalue,
        dense, Cstart, Vstart, "spmm_ascend_cube_aic_without_pipeline");
    EXEC_NPU_CMD(aclnnSpmmAscendCubeAicWithoutPipeline,
                 Crowptr, Ccol_ids, Cvalue, Vrowptr, Vcol_ids, Vvalue,
                 dense, Cstart, Vstart, result);
    return result;
}

at::Tensor spmm_ascend_cube_aiv_mb_impl_npu(
        const at::Tensor& Crowptr,
        const at::Tensor& Ccol_ids,
        const at::Tensor& Cvalue,
        const at::Tensor& Vrowptr,
        const at::Tensor& Vcol_ids,
        const at::Tensor& Vvalue,
        const at::Tensor& dense,
        const at::Tensor& Cstart,
        const at::Tensor& Vstart) {
    auto result = prepare_spmm_mixed_ablation_result(
        Crowptr, Ccol_ids, Cvalue, Vrowptr, Vcol_ids, Vvalue,
        dense, Cstart, Vstart, "spmm_ascend_cube_aiv_mb");
    EXEC_NPU_CMD(aclnnSpmmAscendCubeAivMb,
                 Crowptr, Ccol_ids, Cvalue, Vrowptr, Vcol_ids, Vvalue,
                 dense, Cstart, Vstart, result);
    return result;
}
// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("spmm_csr(Tensor rowptr, Tensor col, Tensor value, Tensor dense, Tensor start) -> Tensor");
		m.def("spmm_cube(Tensor rowptr, Tensor col, Tensor value, Tensor dense, Tensor start) -> Tensor");
		m.def("spmm_ascend_cube(Tensor Crowptr, Tensor Ccol_ids, Tensor Cvalue, Tensor Vrowptr, Tensor Vcol_ids, Tensor Vvalue, Tensor dense, Tensor Cstart, Tensor Vstart) -> Tensor");
        m.def("spmm_csr_aiv_mb_batch(Tensor rowptr, Tensor col, Tensor value, Tensor dense, Tensor start) -> Tensor");
        m.def("spmm_ascend_cube_aic_without_pipeline(Tensor Crowptr, Tensor Ccol_ids, Tensor Cvalue, Tensor Vrowptr, Tensor Vcol_ids, Tensor Vvalue, Tensor dense, Tensor Cstart, Tensor Vstart) -> Tensor");
        m.def("spmm_ascend_cube_aiv_mb(Tensor Crowptr, Tensor Ccol_ids, Tensor Cvalue, Tensor Vrowptr, Tensor Vcol_ids, Tensor Vvalue, Tensor dense, Tensor Cstart, Tensor Vstart) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("spmm_csr", &spmm_csr_impl_npu);
		m.impl("spmm_cube", &spmm_cube_impl_npu);
		m.impl("spmm_ascend_cube", &spmm_ascend_cube_impl_npu);
        m.impl("spmm_csr_aiv_mb_batch", &spmm_csr_aiv_mb_batch_impl_npu);
        m.impl("spmm_ascend_cube_aic_without_pipeline", &spmm_ascend_cube_aic_without_pipeline_impl_npu);
        m.impl("spmm_ascend_cube_aiv_mb", &spmm_ascend_cube_aiv_mb_impl_npu);
}

// template <typename index_t, typename val_t>
// std::map<std::string, at::Tensor> your_logic_kernel(
//     at::TensorAccessor<index_t, 1> crow, 
//     at::TensorAccessor<index_t, 1> col,
//     at::TensorAccessor<val_t, 1> val,
//     int64_t nrows,
//     int64_t ncols,
//     int64_t nnz,
//     int64_t nwindows,
//     int64_t window_size)
// {
//     using namespace std;
//     vector<index_t> crow_window_ptr(nwindows + 1, 0),
//                     unique_col_ids(16 * nnz + 16, 0),
//                     block_ptr(nnz + 1, 0);
//     vector<int16_t> block_local_ids(nnz, 0);
//     vector<val_t>   value(nnz, 0);

//     int64_t nnz_ptr = 0, col_ptr = 0, blk_ptr = 0;
//     block_ptr[blk_ptr++] = nnz_ptr;
//     for (int64_t w = 0; w < nwindows; ++w)
//     {
//         crow_window_ptr[w] = col_ptr / 16;

//         int64_t offset = w * window_size, tag = window_size, cnt = -1, last_col_id = -1;
//         vector<index_t> col_local_id(window_size, 0), start(window_size, 0);
//         if (w + 1 < nwindows) for (int64_t i = 0; i < window_size && i + offset < nrows; i++)
//         {
//             start[i] = crow[i + offset];
//             col_local_id[i] = col[start[i]];
//             if (start[i] == crow[i + offset + 1])
//             {
//                 tag--;
//                 col_local_id[i] = ncols;
//             }
//         }
//         else for (int64_t i = 0; i < window_size; i++)
//         {
//             if (i + offset < nrows)
//             {
//                 start[i] = crow[i + offset];
//                 col_local_id[i] = col[start[i]];
//                 if (start[i] == crow[i + offset + 1])
//                 {
//                     tag--;
//                     col_local_id[i] = ncols;
//                 }
//             }
//             else
//             {
//                 tag--;
//                 col_local_id[i] = ncols;
//             }
//         }
//         while (tag)
//         {
//             auto res = min_element(col_local_id.begin(), col_local_id.end());
//             auto col_id = *res, row_id = std::distance(col_local_id.begin(), res), v = val[start[row_id]];
//             // std::cout<<w<<' '<<col_id<<' '<<row_id<<' '<<v<<' '<<col_local_id[row_id]<<std::endl;
//             // for (int i = 0; i < window_size; i++) cout<<col_local_id[i]<<' ';cout<<endl;
//             // for (int i = 0; i < window_size; i++) cout<<start[i]<<' ';cout<<endl;

//             if (last_col_id != col_id)
//             {
//                 unique_col_ids[col_ptr++] = col_id;

//                 cnt ++;
//                 if (cnt == 16)
//                 {
//                     cnt = 0;
//                     block_ptr[blk_ptr++] = nnz_ptr;
//                 }
//                 last_col_id = col_id;
//             }

//             block_local_ids[nnz_ptr] = 16 * row_id + cnt; //row_id + cnt * window_size
//             value[nnz_ptr] = v;
//             nnz_ptr ++;

//             start[row_id] += 1;
//             col_local_id[row_id] = col[start[row_id]];
//             if (start[row_id] == crow[row_id + offset + 1])
//             {
//                 tag--;
//                 col_local_id[row_id] = ncols;
//             }
//         }
        
//         if (cnt != -1)
//         {
//             for (;cnt < 15; cnt++) unique_col_ids[col_ptr++] = 0;
//             block_ptr[blk_ptr++] = nnz_ptr;
//         }
//     }
//     blk_ptr --;
//     crow_window_ptr[nwindows] = blk_ptr;

//     TORCH_CHECK(col_ptr == blk_ptr * 16, "Error: ", col_ptr, blk_ptr);
//     TORCH_CHECK(nnz_ptr == nnz, "Error: ", nnz_ptr, nnz);


//     std::map<std::string, at::Tensor> out;
//     unique_col_ids.resize(16 * blk_ptr);
//     block_ptr.resize(blk_ptr + 1);
//     auto safe_copy_to_tensor = [](const void* src_data, std::vector<int64_t> shape, at::ScalarType dtype) {
//         // 1. 创建全新的 Tensor（在 CPU 上）
//         at::Tensor t = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
        
//         // 2. 计算需要拷贝的总字节数
//         size_t element_size = at::elementSize(dtype);
//         size_t num_elements = 1;
//         for (auto s : shape) num_elements *= s;
        
//         // 3. 执行内存拷贝 (memcpy)
//         if (num_elements > 0) {
//             std::memcpy(t.data_ptr(), src_data, num_elements * element_size);
//         }
//         return t;
//     };

//     // --- 开始转换并填充字典 ---
//     out["crow_window_ptr"] = safe_copy_to_tensor(
//         crow_window_ptr.data(), {(int64_t)nwindows + 1}, c10::CppTypeToScalarType<index_t>::value);

//     out["unique_col_ids"]  = safe_copy_to_tensor(
//         unique_col_ids.data(), {(int64_t)16 * blk_ptr}, c10::CppTypeToScalarType<index_t>::value);

//     out["block_ptr"]       = safe_copy_to_tensor(
//         block_ptr.data(), {(int64_t)blk_ptr + 1}, c10::CppTypeToScalarType<index_t>::value);

//     out["block_local_ids"] = safe_copy_to_tensor(
//         block_local_ids.data(), {(int64_t)nnz}, c10::CppTypeToScalarType<int16_t>::value); // int16_t -> kShort

//     out["value"]           = safe_copy_to_tensor(
//         value.data(), {(int64_t)nnz}, c10::CppTypeToScalarType<val_t>::value);           // val_t -> kHalf
//     // out["crow_window_ptr"]  = at::from_blob(crow_window_ptr.data(), {nwindows + 1}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
//     // out["unique_col_ids"]   = at::from_blob(unique_col_ids.data(), {16 * blk_ptr}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
//     // out["block_ptr"]        = at::from_blob(block_ptr.data(), {blk_ptr + 1}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
//     // out["block_local_ids"]  = at::from_blob(block_local_ids.data(), {nnz}, at::TensorOptions().dtype(c10::CppTypeToScalarType<int16_t>::value)).clone();
//     // out["value"]            = at::from_blob(value.data(), {nnz}, at::TensorOptions().dtype(c10::CppTypeToScalarType<val_t>::value)).clone();
//     // cout<<sizeof(index_t)<<' '<<sizeof(val_t)<<endl;
//     // cout<<c10::CppTypeToScalarType<index_t>::value<<' '<<c10::CppTypeToScalarType<int16_t>::value<<' '<<c10::CppTypeToScalarType<val_t>::value<<endl;
//     return out;
// }
template <typename index_t, typename val_t>
std::map<std::string, at::Tensor> your_logic_kernel(
    at::TensorAccessor<index_t, 1> crow, 
    at::TensorAccessor<index_t, 1> col,
    at::TensorAccessor<val_t, 1> val,
    int64_t nrows,
    int64_t ncols,
    int64_t nnz,
    int64_t nwindows,
    int64_t window_size)
{
    using namespace std;
    vector<index_t> crow_window_ptr(nwindows + 1, 0),
                    unique_col_ids(16 * nnz + 16 * nrows, 0);
    vector<val_t>   value(16 * nnz + 256 * nrows, 0);

    int64_t col_ptr = 0, blk_ptr = 0;
    for (int64_t w = 0; w < nwindows; ++w)
    {
        crow_window_ptr[w] = blk_ptr;

        int64_t offset = w * window_size, tag = window_size, cnt = -1, last_col_id = -1;
        vector<index_t> col_local_id(window_size, 0), start(window_size, 0);
        if (w + 1 < nwindows) for (int64_t i = 0; i < window_size && i + offset < nrows; i++)
        {
            start[i] = crow[i + offset];
            col_local_id[i] = col[start[i]];
            if (start[i] == crow[i + offset + 1])
            {
                tag--;
                col_local_id[i] = ncols;
            }
        }
        else for (int64_t i = 0; i < window_size; i++)
        {
            if (i + offset < nrows)
            {
                start[i] = crow[i + offset];
                col_local_id[i] = col[start[i]];
                if (start[i] == crow[i + offset + 1])
                {
                    tag--;
                    col_local_id[i] = ncols;
                }
            }
            else
            {
                tag--;
                col_local_id[i] = ncols;
            }
        }
        while (tag)
        {
            auto res = min_element(col_local_id.begin(), col_local_id.end());
            auto col_id = *res, row_id = std::distance(col_local_id.begin(), res), v = val[start[row_id]];
            // std::cout<<w<<' '<<col_id<<' '<<row_id<<' '<<v<<' '<<col_local_id[row_id]<<std::endl;
            // for (int i = 0; i < window_size; i++) cout<<col_local_id[i]<<' ';cout<<endl;
            // for (int i = 0; i < window_size; i++) cout<<start[i]<<' ';cout<<endl;

            if (last_col_id != col_id)
            {
                unique_col_ids[col_ptr++] = col_id;

                cnt ++;
                if (cnt == 16)
                {
                    cnt = 0;
                    blk_ptr++;
                }
                last_col_id = col_id;
            }
            //row_id + cnt * window_size
            value[256 * blk_ptr + 16 * row_id + cnt] = v;

            start[row_id] += 1;
            col_local_id[row_id] = col[start[row_id]];
            if (start[row_id] == crow[row_id + offset + 1])
            {
                tag--;
                col_local_id[row_id] = ncols;
            }
        }
        
        if (cnt != -1)
        {
            for (;cnt < 15; cnt++) unique_col_ids[col_ptr++] = 0;
            blk_ptr++;
        }
    }
    crow_window_ptr[nwindows] = blk_ptr;
    printf("%d\n",blk_ptr);

    TORCH_CHECK(col_ptr == blk_ptr * 16, "Error: ", col_ptr, blk_ptr);

    std::map<std::string, at::Tensor> out;
    unique_col_ids.resize(16 * blk_ptr);
    value.resize(16 * col_ptr);
    auto safe_copy_to_tensor = [](const void* src_data, std::vector<int64_t> shape, at::ScalarType dtype) {
        // 1. 创建全新的 Tensor（在 CPU 上）
        at::Tensor t = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
        
        // 2. 计算需要拷贝的总字节数
        size_t element_size = at::elementSize(dtype);
        size_t num_elements = 1;
        for (auto s : shape) num_elements *= s;
        
        // 3. 执行内存拷贝 (memcpy)
        if (num_elements > 0) {
            std::memcpy(t.data_ptr(), src_data, num_elements * element_size);
        }
        return t;
    };

    // --- 开始转换并填充字典 ---
    out["crow_window_ptr"] = safe_copy_to_tensor(
        crow_window_ptr.data(), {(int64_t)nwindows + 1}, c10::CppTypeToScalarType<index_t>::value);

    out["unique_col_ids"]  = safe_copy_to_tensor(
        unique_col_ids.data(), {(int64_t)16 * blk_ptr}, c10::CppTypeToScalarType<index_t>::value);


    out["value"]           = safe_copy_to_tensor(
        value.data(), {(int64_t)16 * col_ptr}, c10::CppTypeToScalarType<val_t>::value);           // val_t -> kHalf
    // out["crow_window_ptr"]  = at::from_blob(crow_window_ptr.data(), {nwindows + 1}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
    // out["unique_col_ids"]   = at::from_blob(unique_col_ids.data(), {16 * blk_ptr}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
    // out["block_ptr"]        = at::from_blob(block_ptr.data(), {blk_ptr + 1}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
    // out["block_local_ids"]  = at::from_blob(block_local_ids.data(), {nnz}, at::TensorOptions().dtype(c10::CppTypeToScalarType<int16_t>::value)).clone();
    // out["value"]            = at::from_blob(value.data(), {nnz}, at::TensorOptions().dtype(c10::CppTypeToScalarType<val_t>::value)).clone();
    // cout<<sizeof(index_t)<<' '<<sizeof(val_t)<<endl;
    // cout<<c10::CppTypeToScalarType<index_t>::value<<' '<<c10::CppTypeToScalarType<int16_t>::value<<' '<<c10::CppTypeToScalarType<val_t>::value<<endl;
    return out;
}
std::map<std::string, at::Tensor> csr_to_ascend_cube(
    at::Tensor crow_ptr,
    at::Tensor col_indices,
    at::Tensor values,
    int64_t nrows,
    int64_t ncols,
    int64_t window_size = 16)
{

    crow_ptr    = crow_ptr.to(torch::kCPU).contiguous();
    col_indices = col_indices.to(torch::kCPU).contiguous();
    values      = values.to(torch::kCPU).contiguous();

    const int64_t nnz       = col_indices.size(0);
    const int64_t nwindows  = (nrows + window_size - 1) / window_size;

    std::map<std::string, at::Tensor> res;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "csr_logic1", ([&] {
        using val_t = scalar_t;
        AT_DISPATCH_ALL_TYPES(crow_ptr.scalar_type(), "csr_logic2", ([&] {
            // 这里的 scalar_t 会根据运行时自动变为 int32_t 或 int64_t
            auto crow_a = crow_ptr.accessor<scalar_t, 1>();
            auto col_a = col_indices.accessor<scalar_t, 1>();
            auto val_a = values.accessor<val_t, 1>();
            
            // 调用上面定义的逻辑
            res = your_logic_kernel<scalar_t, val_t>(crow_a, col_a, val_a, nrows, ncols, nnz, nwindows, window_size);
        }));
    }));
    return res;
}

template <typename index_t, typename val_t>
std::map<std::string, at::Tensor> ToBCSR(
    at::TensorAccessor<index_t, 1> crow, 
    at::TensorAccessor<index_t, 1> col,
    at::TensorAccessor<val_t, 1> val,
    int64_t nrows,
    int64_t ncols,
    int64_t nnz,
    int64_t nwindows,
    int64_t window_size)
{
    using namespace std;
    vector<index_t> crow_window_ptr(nwindows + 1, 0),
                    unique_col_ids(nnz, 0);
    vector<val_t>   value(16 * nnz, 0);

    int64_t col_ptr = 0;
    for (int64_t w = 0; w < nwindows; ++w)
    {
        crow_window_ptr[w] = col_ptr;

        int64_t offset = w * window_size, tag = window_size, cnt = -1, last_col_id = -1;
        vector<index_t> col_local_id(window_size, 0), start(window_size, 0);
        if (w + 1 < nwindows) for (int64_t i = 0; i < window_size && i + offset < nrows; i++)
        {
            start[i] = crow[i + offset];
            col_local_id[i] = col[start[i]];
            if (start[i] == crow[i + offset + 1])
            {
                tag--;
                col_local_id[i] = ncols;
            }
        }
        else for (int64_t i = 0; i < window_size; i++)
        {
            if (i + offset < nrows)
            {
                start[i] = crow[i + offset];
                col_local_id[i] = col[start[i]];
                if (start[i] == crow[i + offset + 1])
                {
                    tag--;
                    col_local_id[i] = ncols;
                }
            }
            else
            {
                tag--;
                col_local_id[i] = ncols;
            }
        }
        while (tag)
        {
            auto res = min_element(col_local_id.begin(), col_local_id.end());
            auto col_id = *res, row_id = std::distance(col_local_id.begin(), res), v = val[start[row_id]];
            // std::cout<<w<<' '<<col_id<<' '<<row_id<<' '<<v<<' '<<col_local_id[row_id]<<std::endl;
            // for (int i = 0; i < window_size; i++) cout<<col_local_id[i]<<' ';cout<<endl;
            // for (int i = 0; i < window_size; i++) cout<<start[i]<<' ';cout<<endl;

            if (last_col_id != col_id)
            {
                unique_col_ids[col_ptr++] = col_id;

                cnt = 1;
                last_col_id = col_id;
            }
            //row_id + cnt * window_size
            value[16 * col_ptr - 16 + row_id] = v;

            start[row_id] += 1;
            col_local_id[row_id] = col[start[row_id]];
            if (start[row_id] == crow[row_id + offset + 1])
            {
                tag--;
                col_local_id[row_id] = ncols;
            }
        }
    }
    crow_window_ptr[nwindows] = col_ptr;
    printf("%d\n",col_ptr);


    std::map<std::string, at::Tensor> out;
    unique_col_ids.resize(col_ptr);
    value.resize(16 * col_ptr);
    auto safe_copy_to_tensor = [](const void* src_data, std::vector<int64_t> shape, at::ScalarType dtype) {
        // 1. 创建全新的 Tensor（在 CPU 上）
        at::Tensor t = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
        
        // 2. 计算需要拷贝的总字节数
        size_t element_size = at::elementSize(dtype);
        size_t num_elements = 1;
        for (auto s : shape) num_elements *= s;
        
        // 3. 执行内存拷贝 (memcpy)
        if (num_elements > 0) {
            std::memcpy(t.data_ptr(), src_data, num_elements * element_size);
        }
        return t;
    };

    // --- 开始转换并填充字典 ---
    out["crow_window_ptr"] = safe_copy_to_tensor(
        crow_window_ptr.data(), {(int64_t)nwindows + 1}, c10::CppTypeToScalarType<index_t>::value);

    out["unique_col_ids"]  = safe_copy_to_tensor(
        unique_col_ids.data(), {(int64_t)col_ptr}, c10::CppTypeToScalarType<index_t>::value);


    out["value"]           = safe_copy_to_tensor(
        value.data(), {(int64_t)16 * col_ptr}, c10::CppTypeToScalarType<val_t>::value);           // val_t -> kHalf
    // out["crow_window_ptr"]  = at::from_blob(crow_window_ptr.data(), {nwindows + 1}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
    // out["unique_col_ids"]   = at::from_blob(unique_col_ids.data(), {16 * blk_ptr}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
    // out["block_ptr"]        = at::from_blob(block_ptr.data(), {blk_ptr + 1}, at::TensorOptions().dtype(c10::CppTypeToScalarType<index_t>::value)).clone();
    // out["block_local_ids"]  = at::from_blob(block_local_ids.data(), {nnz}, at::TensorOptions().dtype(c10::CppTypeToScalarType<int16_t>::value)).clone();
    // out["value"]            = at::from_blob(value.data(), {nnz}, at::TensorOptions().dtype(c10::CppTypeToScalarType<val_t>::value)).clone();
    // cout<<sizeof(index_t)<<' '<<sizeof(val_t)<<endl;
    // cout<<c10::CppTypeToScalarType<index_t>::value<<' '<<c10::CppTypeToScalarType<int16_t>::value<<' '<<c10::CppTypeToScalarType<val_t>::value<<endl;
    return out;
}
std::map<std::string, at::Tensor> csr_to_bcsr(
    at::Tensor crow_ptr,
    at::Tensor col_indices,
    at::Tensor values,
    int64_t nrows,
    int64_t ncols,
    int64_t window_size = 16)
{

    crow_ptr    = crow_ptr.to(torch::kCPU).contiguous();
    col_indices = col_indices.to(torch::kCPU).contiguous();
    values      = values.to(torch::kCPU).contiguous();

    const int64_t nnz       = col_indices.size(0);
    printf("%d\n", nnz);
    const int64_t nwindows  = (nrows + window_size - 1) / window_size;

    std::map<std::string, at::Tensor> res;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "csr_logic1", ([&] {
        using val_t = scalar_t;
        AT_DISPATCH_ALL_TYPES(crow_ptr.scalar_type(), "csr_logic2", ([&] {
            // 这里的 scalar_t 会根据运行时自动变为 int32_t 或 int64_t
            auto crow_a = crow_ptr.accessor<scalar_t, 1>();
            auto col_a = col_indices.accessor<scalar_t, 1>();
            auto val_a = values.accessor<val_t, 1>();
            
            // 调用上面定义的逻辑
            res = ToBCSR<scalar_t, val_t>(crow_a, col_a, val_a, nrows, ncols, nnz, nwindows, window_size);
        }));
    }));
    return res;
}


struct WindowStat {
    int64_t window_id;
    int64_t nnz_win;
    int64_t n_blocks;
    double density;
};

template <typename index_t, typename val_t>
std::map<std::string, at::Tensor> pack_fixed_layout(
    const std::vector<index_t>& crow,
    const std::vector<index_t>& col,
    const std::vector<val_t>& val,
    int64_t nrows,
    int64_t ncols,
    int64_t window_size);

template <typename index_t, typename val_t>
std::map<std::string, at::Tensor> taskmapping(
    at::TensorAccessor<index_t, 1> crow,
    at::TensorAccessor<index_t, 1> col,
    at::TensorAccessor<val_t, 1> val,
    int64_t nrows,
    int64_t ncols,
    int64_t nnz,
    int64_t window_size = 16,
    bool graph_layout = false)
{
    TORCH_CHECK(window_size == 16, "Ascend-SpMM requires a 16-row window");

    constexpr double kAicBandwidthGbps = 295.0;
    constexpr double kAivBandwidthGbps = 505.0;
    const int64_t num_windows = (nrows + window_size - 1) / window_size;
    std::vector<WindowStat> windows(num_windows);
    std::vector<int64_t> col_epoch(ncols, -1);

    for (int64_t w = 0; w < num_windows; ++w) {
        const int64_t r0 = w * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);
        int64_t window_nnz = 0;
        int64_t window_blocks = 0;
        for (int64_t r = r0; r < r1; ++r) {
            window_nnz += crow[r + 1] - crow[r];
            for (int64_t p = crow[r]; p < crow[r + 1]; ++p) {
                const int64_t c = static_cast<int64_t>(col[p]);
                TORCH_CHECK(c >= 0 && c < ncols, "column index out of range");
                if (col_epoch[c] != w) {
                    col_epoch[c] = w;
                    ++window_blocks;
                }
            }
        }
        windows[w] = {
            w,
            window_nnz,
            window_blocks,
            window_blocks ? static_cast<double>(window_nnz) / window_blocks : 0.0,
        };
    }

    std::stable_sort(windows.begin(), windows.end(), [](const WindowStat& lhs, const WindowStat& rhs) {
        if (lhs.density != rhs.density) return lhs.density > rhs.density;
        return lhs.window_id < rhs.window_id;
    });

    int64_t boundary_pos = -1;
    int64_t prefix_blocks = 0;
    int64_t remaining_nnz = nnz;
    for (int64_t pos = 0; pos < num_windows; ++pos) {
        const auto& current = windows[pos];
        const double aic_after = static_cast<double>(prefix_blocks + current.n_blocks) / kAicBandwidthGbps;
        const double aiv_after = static_cast<double>(remaining_nnz - current.nnz_win) / kAivBandwidthGbps;
        if (aic_after >= aiv_after) {
            boundary_pos = pos;
            break;
        }
        prefix_blocks += current.n_blocks;
        remaining_nnz -= current.nnz_win;
    }

    int64_t boundary_window = -1;
    int64_t boundary_blocks = 0;
    int64_t boundary_aic_blocks = 0;
    double estimated_aic_cost = 0.0;
    double estimated_aiv_cost = 0.0;
    double estimated_max_cost = 0.0;
    std::vector<char> boundary_to_aic(ncols, 0);

    if (boundary_pos >= 0) {
        const auto& boundary = windows[boundary_pos];
        boundary_window = boundary.window_id;
        boundary_blocks = boundary.n_blocks;

        std::vector<int64_t> occupancy(ncols, 0);
        std::vector<index_t> boundary_columns;
        boundary_columns.reserve(boundary_blocks);
        const int64_t r0 = boundary_window * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);
        for (int64_t r = r0; r < r1; ++r) {
            for (int64_t p = crow[r]; p < crow[r + 1]; ++p) {
                const index_t c = col[p];
                if (occupancy[static_cast<int64_t>(c)]++ == 0) boundary_columns.push_back(c);
            }
        }
        TORCH_CHECK(static_cast<int64_t>(boundary_columns.size()) == boundary_blocks,
                    "internal boundary-block count mismatch");
        // Occupancies are integers in [1, window_size].  Stable counting
        // buckets therefore order all c_k boundary blocks in O(c_k + W)
        // time and space, preserving first-seen order within equal buckets.
        ascend_spmm::stable_bucket_sort_by_occupancy_desc(
            boundary_columns, occupancy, window_size);

        const int64_t suffix_nnz = remaining_nnz - boundary.nnz_win;
        int64_t assigned_nnz = 0;
        double best_max_cost = std::numeric_limits<double>::infinity();
        int64_t best_q = 0;
        // Enumerate every block prefix and minimize the slower processor's
        // calibrated cost.  The first q is retained when the costs are equal.
        for (int64_t q = 0; q <= boundary_blocks; ++q) {
            const double aic_cost = static_cast<double>(prefix_blocks + q) / kAicBandwidthGbps;
            const double aiv_cost = static_cast<double>(suffix_nnz + boundary.nnz_win - assigned_nnz) / kAivBandwidthGbps;
            const double max_cost = std::max(aic_cost, aiv_cost);
            if (max_cost < best_max_cost) {
                best_max_cost = max_cost;
                best_q = q;
                estimated_aic_cost = aic_cost;
                estimated_aiv_cost = aiv_cost;
                estimated_max_cost = max_cost;
            }
            if (q < boundary_blocks) {
                assigned_nnz += occupancy[static_cast<int64_t>(boundary_columns[q])];
            }
        }
        boundary_aic_blocks = best_q;
        for (int64_t q = 0; q < best_q; ++q) {
            boundary_to_aic[static_cast<int64_t>(boundary_columns[q])] = 1;
        }
    }

    std::vector<int64_t> aic_windows;
    std::vector<int64_t> aiv_windows;
    if (boundary_pos < 0) {
        for (const auto& window : windows) aiv_windows.push_back(window.window_id);
    } else {
        for (int64_t pos = 0; pos < boundary_pos; ++pos) {
            aic_windows.push_back(windows[pos].window_id);
        }
        if (boundary_aic_blocks > 0) {
            aic_windows.push_back(boundary_window);
        } else {
            aiv_windows.push_back(boundary_window);
        }
        for (int64_t pos = boundary_pos + 1; pos < num_windows; ++pos) {
            aiv_windows.push_back(windows[pos].window_id);
        }
    }

    const bool shared_boundary = boundary_aic_blocks > 0 && boundary_aic_blocks < boundary_blocks;
    const int64_t aic_padded_rows = static_cast<int64_t>(aic_windows.size()) * window_size;
    int64_t aic_real_rows = 0;
    for (const int64_t w : aic_windows) {
        aic_real_rows += std::min((w + 1) * window_size, nrows) - w * window_size;
    }
    const int64_t packed_real_extent = aic_padded_rows + (nrows - aic_real_rows);
    const int64_t packed_layout_supported = graph_layout && nrows == ncols ? 1 : 0;
    const int64_t output_rows = packed_layout_supported
        ? num_windows * window_size
        : packed_real_extent;

    std::vector<int64_t> permutation(output_rows, -1);
    std::vector<int64_t> inv_permutation(nrows, -1);
    for (int64_t slot = 0; slot < static_cast<int64_t>(aic_windows.size()); ++slot) {
        const int64_t w = aic_windows[slot];
        const int64_t r0 = w * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t out_row = slot * window_size + (r - r0);
            permutation[out_row] = r;
            inv_permutation[r] = out_row;
        }
    }
    int64_t vector_row = aic_padded_rows;
    for (const int64_t w : aiv_windows) {
        const int64_t r0 = w * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);
        for (int64_t r = r0; r < r1; ++r) {
            permutation[vector_row] = r;
            inv_permutation[r] = vector_row;
            ++vector_row;
        }
    }
    TORCH_CHECK(vector_row == packed_real_extent && vector_row <= output_rows,
                "internal row-packing error");

    std::vector<index_t> crow_window_ptr(aic_windows.size() + 1, 0);
    std::vector<index_t> unique_col_ids;
    std::vector<val_t> cube_values;
    std::vector<int64_t> block_epoch(ncols, -1);
    std::vector<int64_t> block_slot(ncols, -1);

    for (int64_t slot = 0; slot < static_cast<int64_t>(aic_windows.size()); ++slot) {
        const int64_t w = aic_windows[slot];
        const int64_t r0 = w * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);
        std::vector<index_t> block_columns;
        for (int64_t r = r0; r < r1; ++r) {
            for (int64_t p = crow[r]; p < crow[r + 1]; ++p) {
                const index_t c = col[p];
                if (shared_boundary && w == boundary_window && !boundary_to_aic[static_cast<int64_t>(c)]) continue;
                if (block_epoch[static_cast<int64_t>(c)] != slot) {
                    block_epoch[static_cast<int64_t>(c)] = slot;
                    block_columns.push_back(c);
                }
            }
        }
        std::sort(block_columns.begin(), block_columns.end());
        const int64_t block_base = unique_col_ids.size();
        unique_col_ids.insert(unique_col_ids.end(), block_columns.begin(), block_columns.end());
        cube_values.resize(unique_col_ids.size() * window_size, val_t(0));
        for (int64_t local = 0; local < static_cast<int64_t>(block_columns.size()); ++local) {
            const int64_t c = static_cast<int64_t>(block_columns[local]);
            block_slot[c] = block_base + local;
        }
        for (int64_t r = r0; r < r1; ++r) {
            for (int64_t p = crow[r]; p < crow[r + 1]; ++p) {
                const index_t c = col[p];
                if (shared_boundary && w == boundary_window && !boundary_to_aic[static_cast<int64_t>(c)]) continue;
                cube_values[block_slot[static_cast<int64_t>(c)] * window_size + (r - r0)] = val[p];
            }
        }
        crow_window_ptr[slot + 1] = static_cast<index_t>(unique_col_ids.size());
    }

    std::vector<index_t> vrow_window_ptr(output_rows + 1, 0);
    std::vector<index_t> vcol_indices;
    std::vector<val_t> vvalue;
    vcol_indices.reserve(nnz);
    vvalue.reserve(nnz);
    for (int64_t out_row = 0; out_row < output_rows; ++out_row) {
        vrow_window_ptr[out_row] = static_cast<index_t>(vcol_indices.size());
        const int64_t original_row = permutation[out_row];
        if (original_row < 0) continue;
        const int64_t original_window = original_row / window_size;
        const bool vector_owned = out_row >= aic_padded_rows;
        const bool boundary_residual = shared_boundary && original_window == boundary_window;
        if (!vector_owned && !boundary_residual) continue;
        for (int64_t p = crow[original_row]; p < crow[original_row + 1]; ++p) {
            const index_t c = col[p];
            if (boundary_residual && boundary_to_aic[static_cast<int64_t>(c)]) continue;
            vcol_indices.push_back(c);
            vvalue.push_back(val[p]);
        }
    }
    vrow_window_ptr[output_rows] = static_cast<index_t>(vcol_indices.size());

    std::vector<index_t> renumbered_unique_col_ids;
    std::vector<index_t> renumbered_vcol_indices;
    std::map<std::string, at::Tensor> transpose_mapping;
    if (packed_layout_supported) {
        // permutation[packed_row] = original_row, so inv_permutation maps
        // every original node id to the column id consumed by packed output.
        // These columns form P*A*P^T for persistent graph relabeling.
        renumbered_unique_col_ids =
            ascend_spmm::renumber_columns_for_packed_layout(
                unique_col_ids, inv_permutation);
        renumbered_vcol_indices =
            ascend_spmm::renumber_columns_for_packed_layout(
                vcol_indices, inv_permutation);

        std::vector<index_t> source_crow(nrows + 1);
        std::vector<index_t> source_col(nnz);
        std::vector<val_t> source_values(nnz);
        for (int64_t r = 0; r <= nrows; ++r) source_crow[r] = crow[r];
        for (int64_t p = 0; p < nnz; ++p) {
            source_col[p] = col[p];
            source_values[p] = val[p];
        }
        std::vector<index_t> transpose_crow;
        std::vector<index_t> transpose_col;
        std::vector<val_t> transpose_values;
        ascend_spmm::transpose_into_packed_layout(
            source_crow, source_col, source_values, inv_permutation,
            static_cast<std::size_t>(output_rows), transpose_crow,
            transpose_col, transpose_values);
        transpose_mapping = pack_fixed_layout<index_t, val_t>(
            transpose_crow, transpose_col, transpose_values,
            output_rows, output_rows, window_size);
    }

    auto safe_copy_to_tensor = [](const void* src_data, std::vector<int64_t> shape, at::ScalarType dtype) {
        at::Tensor tensor = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
        size_t elements = 1;
        for (const int64_t extent : shape) elements *= static_cast<size_t>(extent);
        if (elements > 0) std::memcpy(tensor.data_ptr(), src_data, elements * at::elementSize(dtype));
        return tensor;
    };

    std::map<std::string, at::Tensor> out;
    out["crow_window_ptr"] = safe_copy_to_tensor(
        crow_window_ptr.data(), {static_cast<int64_t>(crow_window_ptr.size())},
        c10::CppTypeToScalarType<index_t>::value);
    out["unique_col_ids"] = safe_copy_to_tensor(
        unique_col_ids.data(), {static_cast<int64_t>(unique_col_ids.size())},
        c10::CppTypeToScalarType<index_t>::value);
    out["value"] = safe_copy_to_tensor(
        cube_values.data(), {static_cast<int64_t>(cube_values.size())},
        c10::CppTypeToScalarType<val_t>::value);
    out["vrow_window_ptr"] = safe_copy_to_tensor(
        vrow_window_ptr.data(), {static_cast<int64_t>(vrow_window_ptr.size())},
        c10::CppTypeToScalarType<index_t>::value);
    out["vcol_indices"] = safe_copy_to_tensor(
        vcol_indices.data(), {static_cast<int64_t>(vcol_indices.size())},
        c10::CppTypeToScalarType<index_t>::value);
    out["vvalue"] = safe_copy_to_tensor(
        vvalue.data(), {static_cast<int64_t>(vvalue.size())},
        c10::CppTypeToScalarType<val_t>::value);
    out["renumbered_unique_col_ids"] = safe_copy_to_tensor(
        renumbered_unique_col_ids.data(),
        {static_cast<int64_t>(renumbered_unique_col_ids.size())},
        c10::CppTypeToScalarType<index_t>::value);
    out["renumbered_vcol_indices"] = safe_copy_to_tensor(
        renumbered_vcol_indices.data(),
        {static_cast<int64_t>(renumbered_vcol_indices.size())},
        c10::CppTypeToScalarType<index_t>::value);
    if (packed_layout_supported) {
        out["transpose_crow_window_ptr"] = transpose_mapping.at("crow_window_ptr");
        out["transpose_unique_col_ids"] = transpose_mapping.at("unique_col_ids");
        out["transpose_value"] = transpose_mapping.at("value");
        out["transpose_vrow_window_ptr"] = transpose_mapping.at("vrow_window_ptr");
        out["transpose_vcol_indices"] = transpose_mapping.at("vcol_indices");
        out["transpose_vvalue"] = transpose_mapping.at("vvalue");
    } else {
        out["transpose_crow_window_ptr"] = safe_copy_to_tensor(
            nullptr, {0}, c10::CppTypeToScalarType<index_t>::value);
        out["transpose_unique_col_ids"] = safe_copy_to_tensor(
            nullptr, {0}, c10::CppTypeToScalarType<index_t>::value);
        out["transpose_value"] = safe_copy_to_tensor(
            nullptr, {0}, c10::CppTypeToScalarType<val_t>::value);
        out["transpose_vrow_window_ptr"] = safe_copy_to_tensor(
            nullptr, {0}, c10::CppTypeToScalarType<index_t>::value);
        out["transpose_vcol_indices"] = safe_copy_to_tensor(
            nullptr, {0}, c10::CppTypeToScalarType<index_t>::value);
        out["transpose_vvalue"] = safe_copy_to_tensor(
            nullptr, {0}, c10::CppTypeToScalarType<val_t>::value);
    }
    out["permutation"] = safe_copy_to_tensor(
        permutation.data(), {static_cast<int64_t>(permutation.size())}, at::kLong);
    out["inv_permutation"] = safe_copy_to_tensor(
        inv_permutation.data(), {static_cast<int64_t>(inv_permutation.size())}, at::kLong);

    const int64_t shared_windows = shared_boundary ? 1 : 0;
    out["packed_rows"] = safe_copy_to_tensor(&output_rows, {1}, at::kLong);
    out["packed_layout_supported"] = safe_copy_to_tensor(
        &packed_layout_supported, {1}, at::kLong);
    out["marginal_window"] = safe_copy_to_tensor(&boundary_window, {1}, at::kLong);
    out["marginal_aic_blocks"] = safe_copy_to_tensor(&boundary_aic_blocks, {1}, at::kLong);
    out["marginal_total_blocks"] = safe_copy_to_tensor(&boundary_blocks, {1}, at::kLong);
    out["estimated_aic_cost"] = safe_copy_to_tensor(&estimated_aic_cost, {1}, at::kDouble);
    out["estimated_aiv_cost"] = safe_copy_to_tensor(&estimated_aiv_cost, {1}, at::kDouble);
    out["estimated_max_cost"] = safe_copy_to_tensor(&estimated_max_cost, {1}, at::kDouble);
    out["shared_windows"] = safe_copy_to_tensor(&shared_windows, {1}, at::kLong);
    return out;
}

std::map<std::string, at::Tensor> preprocess(
    at::Tensor crow_ptr,
    at::Tensor col_indices,
    at::Tensor values,
    int64_t nrows,
    int64_t ncols,
    int64_t window_size = 16,
    bool graph_layout = false)
{
    crow_ptr = crow_ptr.to(torch::kCPU).contiguous();
    col_indices = col_indices.to(torch::kCPU).contiguous();
    values = values.to(torch::kCPU).contiguous();
    std::map<std::string, at::Tensor> result;
    const int64_t nnz = col_indices.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "taskmapping_value", ([&] {
        using val_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(crow_ptr.scalar_type(), "taskmapping_index", ([&] {
            auto crow = crow_ptr.accessor<index_t, 1>();
            auto col = col_indices.accessor<index_t, 1>();
            auto val = values.accessor<val_t, 1>();
            result = taskmapping<index_t, val_t>(
                crow, col, val, nrows, ncols, nnz, window_size, graph_layout);
        }));
    }));
    return result;
}

template <typename index_t>
struct BlockInfo {
    int64_t win;  // row window id
    index_t col;  // column id of the (window_size x 1) block
    index_t cnt;  // real nnz inside the block (density = cnt / window_size)
};

template <typename index_t, typename val_t>
std::map<std::string, at::Tensor> hybrid_split_impl(
    at::TensorAccessor<index_t, 1> crow,
    at::TensorAccessor<index_t, 1> col,
    at::TensorAccessor<val_t, 1> val,
    int64_t nrows,
    int64_t ncols,
    int64_t nnz,
    double a,          // target ratio (#BSR blocks) / (#CSR nnz)
    int64_t window_size = 16)
{
    const int64_t num_windows = (nrows + window_size - 1) / window_size;
    const int64_t m_pad = num_windows * window_size;

    // ========================================================================
    // Phase 1: block profiling — collect every (window, column) block that
    // contains >= 1 nnz, together with its real nnz count.  O(nnz).
    // ========================================================================
    std::vector<BlockInfo<index_t>> blocks;          // window-major order
    std::vector<int64_t> win_block_start(num_windows + 1, 0);
    std::vector<int64_t> col_epoch(ncols, -1);       // epoch marks, no memset
    std::vector<index_t> col_cnt(ncols, 0);

    for (int64_t w = 0; w < num_windows; ++w) {
        const int64_t b0 = (int64_t)blocks.size();
        const int64_t r0 = w * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);

        for (int64_t r = r0; r < r1; ++r) {
            for (int64_t p = crow[r]; p < crow[r + 1]; ++p) {
                const index_t c = col[p];
                if (col_epoch[c] != w) {
                    col_epoch[c] = w;
                    col_cnt[c] = 0;
                    blocks.push_back({w, c, 0});
                }
                col_cnt[c]++;
            }
        }
        // fill in the real counts of this window's blocks
        for (int64_t b = b0; b < (int64_t)blocks.size(); ++b)
            blocks[b].cnt = col_cnt[blocks[b].col];
        win_block_start[w + 1] = (int64_t)blocks.size();
    }
    const int64_t n_blocks = (int64_t)blocks.size();

    // ========================================================================
    // Phase 2: sort blocks by ascending density (capacity is constant
    // window_size, so sorting by cnt == sorting by density) and pick k
    // minimizing | (n_blocks - k) / prefix[k] - a |.
    // ========================================================================
    std::vector<int64_t> order(n_blocks);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t x, int64_t y) { return blocks[x].cnt < blocks[y].cnt; });

    // prefix[k] = nnz moved to CSR if the k least-dense blocks are moved.
    std::vector<int64_t> prefix(n_blocks + 1, 0);
    for (int64_t i = 0; i < n_blocks; ++i)
        prefix[i + 1] = prefix[i] + blocks[order[i]].cnt;

    int64_t best_k = 0;
    double best_diff = std::numeric_limits<double>::infinity();
    for (int64_t k = 0; k <= n_blocks; ++k) {
        const int64_t rem = n_blocks - k;
        const int64_t nz = prefix[k];
        const double ratio = (nz == 0)
            ? (rem > 0 ? std::numeric_limits<double>::infinity() : 0.0)
            : (double)rem / (double)nz;
        const double diff = std::fabs(ratio - a);
        if (diff < best_diff) {
            best_diff = diff;
            best_k = k;
        }
    }

    std::vector<char> moved(n_blocks, 0);            // 1 = block goes to CSR
    for (int64_t i = 0; i < best_k; ++i) moved[order[i]] = 1;

    // ========================================================================
    // Phase 3a: BSR part — block-row pointer + ascending column ids + flat
    // zero-padded values.
    // ========================================================================
    std::vector<index_t> crow_window_ptr(num_windows + 1, 0);
    for (int64_t w = 0; w < num_windows; ++w) {
        int64_t kept = 0;
        for (int64_t b = win_block_start[w]; b < win_block_start[w + 1]; ++b)
            kept += moved[b] ? 0 : 1;
        crow_window_ptr[w + 1] = crow_window_ptr[w] + (index_t)kept;
    }
    const int64_t n_kept = crow_window_ptr[num_windows];

    std::vector<index_t> unique_col_ids(n_kept, 0);
    std::vector<val_t> value(n_kept * window_size, val_t(0));

    // (window, col) -> output slot lookup, epoch-marked.  O(nnz).
    std::vector<int64_t> slot_epoch(ncols, -1);
    std::vector<int64_t> slot_of(ncols, -1);
    std::vector<index_t> kept_cols;                  // reusable scratch
    kept_cols.reserve(256);

    for (int64_t w = 0; w < num_windows; ++w) {
        kept_cols.clear();
        for (int64_t b = win_block_start[w]; b < win_block_start[w + 1]; ++b)
            if (!moved[b]) kept_cols.push_back(blocks[b].col);
        std::sort(kept_cols.begin(), kept_cols.end());  // ascending in-window

        int64_t pos = crow_window_ptr[w];
        for (const index_t c : kept_cols) {
            unique_col_ids[pos] = c;
            slot_epoch[c] = w;
            slot_of[c] = pos;
            ++pos;
        }

        // scatter this window's nnz into the flat value array; entries that
        // belong to moved blocks are skipped (slot_epoch != w), padding
        // stays zero.
        const int64_t r0 = w * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);
        for (int64_t r = r0; r < r1; ++r) {
            for (int64_t p = crow[r]; p < crow[r + 1]; ++p) {
                const index_t c = col[p];
                if (slot_epoch[c] == w)
                    value[slot_of[c] * window_size + (r - r0)] = val[p];
            }
        }
    }

    // ========================================================================
    // Phase 3b: CSR part — residual nnz of the moved blocks, padded row
    // count m_pad (tail pointers repeat).
    // ========================================================================
    const int64_t vnnz_total = prefix[best_k];       // known from Phase 2
    std::vector<index_t> vrow_window_ptr(m_pad + 1, 0);
    std::vector<index_t> vcol_indices(vnnz_total, 0);
    std::vector<val_t> vvalue(vnnz_total, val_t(0));

    std::vector<int64_t> moved_epoch(ncols, -1);     // (window, col) -> moved?

    int64_t vnnz = 0;
    for (int64_t w = 0; w < num_windows; ++w) {
        for (int64_t b = win_block_start[w]; b < win_block_start[w + 1]; ++b)
            if (moved[b]) moved_epoch[blocks[b].col] = w;

        const int64_t r0 = w * window_size;
        const int64_t r1 = std::min(r0 + window_size, nrows);
        for (int64_t r = r0; r < r1; ++r) {
            vrow_window_ptr[r] = (index_t)vnnz;
            for (int64_t p = crow[r]; p < crow[r + 1]; ++p) {
                const index_t c = col[p];
                if (moved_epoch[c] == w) {
                    vcol_indices[vnnz] = c;
                    vvalue[vnnz] = val[p];
                    ++vnnz;
                }
            }
        }
        // padding rows inside this window (only non-empty for the last one)
        for (int64_t r = r1; r < r0 + window_size; ++r)
            vrow_window_ptr[r] = (index_t)vnnz;
    }
    vrow_window_ptr[m_pad] = (index_t)vnnz;
    // vnnz == vnnz_total by construction

    // ========================================================================
    // Phase 4: pack into tensors (same helper style as the reference code).
    // ========================================================================
    auto safe_copy_to_tensor = [](const void* src_data, std::vector<int64_t> shape,
                                  at::ScalarType dtype) {
        at::Tensor t = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
        size_t element_size = at::elementSize(dtype);
        size_t num_elements = 1;
        for (auto s : shape) num_elements *= (size_t)s;
        if (num_elements > 0) {
            std::memcpy(t.data_ptr(), src_data, num_elements * element_size);
        }
        return t;
    };

    std::map<std::string, at::Tensor> out;
    out["crow_window_ptr"] = safe_copy_to_tensor(
        crow_window_ptr.data(), {num_windows + 1},
        c10::CppTypeToScalarType<index_t>::value);
    out["unique_col_ids"] = safe_copy_to_tensor(
        unique_col_ids.data(), {n_kept},
        c10::CppTypeToScalarType<index_t>::value);
    out["value"] = safe_copy_to_tensor(
        value.data(), {n_kept * window_size},
        c10::CppTypeToScalarType<val_t>::value);

    out["vrow_window_ptr"] = safe_copy_to_tensor(
        vrow_window_ptr.data(), {m_pad + 1},
        c10::CppTypeToScalarType<index_t>::value);
    out["vcol_indices"] = safe_copy_to_tensor(
        vcol_indices.data(), {vnnz_total},
        c10::CppTypeToScalarType<index_t>::value);
    out["vvalue"] = safe_copy_to_tensor(
        vvalue.data(), {vnnz_total},
        c10::CppTypeToScalarType<val_t>::value);

    return out;
}

template <typename index_t, typename val_t>
std::map<std::string, at::Tensor> pack_fixed_layout(
    const std::vector<index_t>& crow,
    const std::vector<index_t>& col,
    const std::vector<val_t>& val,
    int64_t nrows,
    int64_t ncols,
    int64_t window_size)
{
    auto index_options = at::TensorOptions()
        .dtype(c10::CppTypeToScalarType<index_t>::value)
        .device(at::kCPU);
    auto value_options = at::TensorOptions()
        .dtype(c10::CppTypeToScalarType<val_t>::value)
        .device(at::kCPU);
    at::Tensor crow_tensor = at::empty(
        {static_cast<int64_t>(crow.size())}, index_options);
    at::Tensor col_tensor = at::empty(
        {static_cast<int64_t>(col.size())}, index_options);
    at::Tensor val_tensor = at::empty(
        {static_cast<int64_t>(val.size())}, value_options);
    if (!crow.empty()) {
        std::memcpy(crow_tensor.data_ptr(), crow.data(), crow.size() * sizeof(index_t));
    }
    if (!col.empty()) {
        std::memcpy(col_tensor.data_ptr(), col.data(), col.size() * sizeof(index_t));
        std::memcpy(val_tensor.data_ptr(), val.data(), val.size() * sizeof(val_t));
    }
    return hybrid_split_impl<index_t, val_t>(
        crow_tensor.accessor<index_t, 1>(),
        col_tensor.accessor<index_t, 1>(),
        val_tensor.accessor<val_t, 1>(),
        nrows, ncols, static_cast<int64_t>(col.size()),
        295.0 / 505.0, window_size);
}


// ----------------------------------------------------------------------------
// Entry point: type dispatch, same pattern as the reference preprocess.
// ----------------------------------------------------------------------------
std::map<std::string, at::Tensor> block_level_split(
    at::Tensor crow_ptr,
    at::Tensor col_indices,
    at::Tensor values,
    int64_t nrows,
    int64_t ncols,
    double a,                    // target ratio (#BSR blocks) / (#CSR nnz)
    int64_t window_size = 16)
{
    std::map<std::string, at::Tensor> res;
    const int64_t nnz = col_indices.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "hybrid_split_val", ([&] {
        using val_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(crow_ptr.scalar_type(), "hybrid_split_idx", ([&] {
            auto crow_a = crow_ptr.accessor<index_t, 1>();
            auto col_a = col_indices.accessor<index_t, 1>();
            auto val_a = values.accessor<val_t, 1>();
            res = hybrid_split_impl<index_t, val_t>(
                crow_a, col_a, val_a, nrows, ncols, nnz, a, window_size);
        }));
    }));
    return res;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_csr", &spmm_csr_impl_npu,
          "NPU CSR SpMM; FP16 width divisible by 16 with MAXD=1024, FP32 width divisible by 8 with MAXD=512");
    m.def("spmm_cube", &spmm_cube_impl_npu,
          "NPU Cube SpMM; width divisible by 16, MAXD=1024 for FP16 and 512 for FP32");
    m.def("spmm_ascend_cube", &spmm_ascend_cube_impl_npu,
          "NPU mixed AIC/AIV SpMM; width divisible by 16, MAXD=1024 for FP16 and 512 for FP32");
    m.def("spmm_ascend_cube_aic_without_pipeline",
          &spmm_ascend_cube_aic_without_pipeline_impl_npu,
          "AIC ablation: reduction batch 16 instead of the hierarchy-aware batch 96");
    m.def("spmm_ascend_cube_aiv_mb", &spmm_ascend_cube_aiv_mb_impl_npu,
          "AIV ablation: explicit eight-slot Multiple Buffer schedule");
    m.def("spmm_csr_aiv_mb_batch", &spmm_csr_aiv_mb_batch_impl_npu,
          "AIV ablation: batch size 8 over 64 ring-buffer slots");
    m.def("csr_to_ascend_cube", &csr_to_ascend_cube,
    R"doc(
    Convert CSR sparse matrix to Ascend Cube format.
    Requires: each row's col_indices must be sorted in ascending order.

    Returns dict with keys:
        crow_window_ptr   Tensor  [nwindows + 1]
        unique_col_ids    Tensor  [total_blocks * 16]
        value             Tensor  [total_blocks * 16 * 16]
    )doc");
    m.def("csr_to_bcsr", &csr_to_bcsr,
    R"doc(
    Convert CSR sparse matrix to BCSR(16 * 1).
    Requires: each row's col_indices must be sorted in ascending order.

    Returns dict with keys:
        crow_window_ptr   Tensor  [nwindows + 1]
        unique_col_ids    Tensor  [total_blocks]
        value             Tensor  [total_blocks * 16]
    )doc");
    m.def("preprocess", &preprocess,
    R"doc(
    Partition coalesced CSR by the calibrated AIC/AIV bandwidths
    (295/505 GB/s). Duplicate coordinates must be merged before this call.
    All dense windows before the marginal window are assigned to AIC, all
    sparse windows after it are assigned to AIV, and only the marginal
    window is split at block granularity.  Its c_k blocks are ordered by a
    stable occupancy bucket sort in O(c_k + W), every candidate prefix is
    evaluated, and the split minimizing max(AIC cost, AIV cost) is selected.

    Returns the packed AIC/AIV tensors, row permutations, and audit fields.
    With graph_layout=True on a square graph, one call also returns the relabeled forward matrix
    P*A*P^T and its fixed-layout transpose P*A^T*P^T. Applications may
    relabel node features, labels, edge endpoints, and split indices once.
    For X_p=P*X, (P*A*P^T)*X_p=P*(A*X), so forward activations and
    backward gradients stay in the packed node-id space across all layers.
    No inverse permutation belongs in an individual SpMM call. Apply P^T
    at most once at an application boundary that requires original node ids.
    Graph relabeling is a preprocessing cost and is reported separately
    from device-kernel SpMM timing.

        permutation              packed row -> original row (-1 is padding)
        inv_permutation          original row -> packed row
        packed_rows              Number of rows in packed graph tensors
        packed_layout_supported  1 when graph_layout=True on a square matrix
        renumbered_unique_col_ids Packed-layout forward AIC columns, or empty
        renumbered_vcol_indices  Packed-layout forward AIV columns, or empty
        transpose_*              Fixed-layout transpose AIC/AIV tensors
        marginal_window       Original row-window id, or -1
        marginal_aic_blocks   Marginal blocks assigned to AIC
        marginal_total_blocks Total nonempty blocks in the marginal window
        estimated_aic_cost    Normalized AIC cost at the selected split
        estimated_aiv_cost    Normalized AIV cost at the selected split
        estimated_max_cost    Maximum of the two selected costs
        shared_windows        0 or 1
    )doc",
          pybind11::arg("crow_ptr"),
          pybind11::arg("col_indices"),
          pybind11::arg("values"),
          pybind11::arg("nrows"),
          pybind11::arg("ncols"),
          pybind11::arg("window_size") = 16,
          pybind11::arg("graph_layout") = false);
    m.def("block_level_split", &block_level_split,
          "Split CSR into (16,1)-BSR + CSR hybrid dict by block density");
}