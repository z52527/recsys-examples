#include <ATen/Functions.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include "../include/utils.h"

constexpr int kMaxNumTensors = 128;
template <typename T>
struct InputJaggedTensor {
	T* value_list[kMaxNumTensors];
	int32_t* offsets_list[kMaxNumTensors];
};

// float4 vectorized copy function
template<typename T>
__device__ __forceinline__ void copy_float4(T* dst, const T* src) {
    if (sizeof(T) == sizeof(float) && 
        reinterpret_cast<uintptr_t>(dst) % 16 == 0 && 
        reinterpret_cast<uintptr_t>(src) % 16 == 0) {
        // float4 copy
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
    } else if (sizeof(T) == sizeof(at::Half) && 
               reinterpret_cast<uintptr_t>(dst) % 16 == 0 && 
               reinterpret_cast<uintptr_t>(src) % 16 == 0) {
        // half8 copy (8 half = 1 float4)
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
    } else {
        // fallback to scalar copy
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[i] = src[i];
        }
    }
}

template <typename T>
__launch_bounds__(1024, 2) __global__ void concat_2D_jagged_tensors_forward_kernel_opt_v4(
// __global__ void concat_2D_jagged_tensors_forward_kernel_opt_v4(
    const InputJaggedTensor<T> input_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t hidden_dim,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    const int32_t seqlen_per_block,
    int* workload_offset,
    T* merged_values,
    int* merged_offsets) {
    // each block processes workload_offset[i+1]-workload_offset[i] sequences
    for(int block_id = blockIdx.x; block_id < total_blocks; block_id += gridDim.x){
        if(workload_offset[block_id] == workload_offset[block_id + 1]) continue;
        
        int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
        int tensor_id = (block_id / num_bucket_per_batch) % num_tensors;
        int batch_id =  block_id / num_bucket_per_batch / num_tensors; 
        int idx = block_id % num_bucket_per_batch; // which bucket
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        const int32_t* offsets = input_jagged_tensor.offsets_list[tensor_id];
        const T* values = input_jagged_tensor.value_list[tensor_id];
        int seq_start = offsets[batch_id];
        
        // for merged_values, each block processes workload_offset[block_id+1]-workload_offset[block_id] sequences starting from workload_offset[block_id] 
        // for source_values, copy len sequences from input_jagged_tensor.value_list[tensor_id] starting from seq_start

        for(int seq_offset = warp_id; seq_offset < workload_offset[block_id+1]-workload_offset[block_id]; seq_offset += 32){
            int src_row = seq_start + seq_offset + idx*seqlen_per_block; // add bucket offset
            int dst_row = workload_offset[block_id] + seq_offset;
            // warp-level parallelization for hidden_dim
            // each thread in warp handles multiple hidden dimensions
            int elements_per_thread = (hidden_dim + 32 - 1) / 32;
            int thread_start = lane_id * elements_per_thread;
            int thread_end = min(thread_start + elements_per_thread, hidden_dim);

            // vectorized copy 
            if (hidden_dim % 4 == 0 && thread_start % 4 == 0) {
                int vectorized_start = (thread_start + 3) / 4 * 4;
                int vectorized_end = thread_end / 4 * 4;
                
                for (int h = thread_start; h < vectorized_start && h < thread_end; ++h) {
                    merged_values[dst_row * hidden_dim + h] = values[src_row * hidden_dim + h];
                }
                
                for (int h = vectorized_start; h < vectorized_end; h += 4) {
                    copy_float4(&merged_values[dst_row * hidden_dim + h],
                                &values[src_row * hidden_dim + h]);
                }
                
                // handle remaining elements
                for (int h = vectorized_end; h < thread_end; ++h) {
                    merged_values[dst_row * hidden_dim + h] = values[src_row * hidden_dim + h];
                }
            } else {
                // scalar copy
                for (int h = thread_start; h < thread_end; ++h) {
                    merged_values[dst_row * hidden_dim + h] = values[src_row * hidden_dim + h];
                }
            }
        }
    }
}


template <typename T, int HIDDEN_DIM>
__launch_bounds__(1024, 2) __global__ void concat_2D_jagged_tensors_forward_kernel_alignment(
    const InputJaggedTensor<T> input_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    const int32_t seqlen_per_block,
    int* workload_offset,
    T* merged_values,
    int* merged_offsets) {
    // each block processes workload_offset[i+1]-workload_offset[i] sequences
    constexpr int elements_per_thread = (HIDDEN_DIM + 31) / 32;
    for(int block_id = blockIdx.x; block_id < total_blocks; block_id += gridDim.x){
        if(workload_offset[block_id] == workload_offset[block_id + 1]) continue;
        
        int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
        int tensor_id = (block_id / num_bucket_per_batch) % num_tensors;
        int batch_id =  block_id / num_bucket_per_batch / num_tensors; 
        int idx = block_id % num_bucket_per_batch; // which bucket
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        const int32_t* offsets = input_jagged_tensor.offsets_list[tensor_id];
        const T* values = input_jagged_tensor.value_list[tensor_id];
        int seq_start = offsets[batch_id];
        
        // for merged_values, each block processes workload_offset[block_id+1]-workload_offset[block_id] sequences starting from workload_offset[block_id] 
        // for source_values, copy len sequences from input_jagged_tensor.value_list[tensor_id] starting from seq_start

        for(int seq_offset = warp_id; seq_offset < workload_offset[block_id+1]-workload_offset[block_id]; seq_offset += 32){
            int src_row = seq_start + seq_offset + idx*seqlen_per_block; // add bucket offset
            int dst_row = workload_offset[block_id] + seq_offset;
            int h_start = lane_id * elements_per_thread;
            #pragma unroll
            for(int i = 0; i < elements_per_thread; i += 4) {
                copy_float4(&merged_values[dst_row * HIDDEN_DIM + h_start + i], &values[src_row * HIDDEN_DIM + h_start + i]);
            }
        }
    }
}
template <typename T>
__launch_bounds__(1024, 2) __global__ void concat_2D_jagged_tensors_forward_kernel_warp(
    const InputJaggedTensor<T> input_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t hidden_dim,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    const int32_t seqlen_per_block,
    int* workload_offset,
    T* merged_values,
    int* merged_offsets) {
    // each block processes workload_offset[i+1]-workload_offset[i] sequences
    for(int block_id = blockIdx.x; block_id < total_blocks; block_id += gridDim.x){
        if(workload_offset[block_id] == workload_offset[block_id + 1]) continue;
        
        int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
        int tensor_id = (block_id / num_bucket_per_batch) % num_tensors;
        int batch_id =  block_id / num_bucket_per_batch / num_tensors; 
        int idx = block_id % num_bucket_per_batch; // which bucket
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        const int32_t* offsets = input_jagged_tensor.offsets_list[tensor_id];
        const T* values = input_jagged_tensor.value_list[tensor_id];
        int seq_start = offsets[batch_id];
        
        // for merged_values, each block processes workload_offset[block_id+1]-workload_offset[block_id] sequences starting from workload_offset[block_id] 
        // for source_values, copy len sequences from input_jagged_tensor.value_list[tensor_id] starting from seq_start

        for(int seq_offset = warp_id; seq_offset < workload_offset[block_id+1]-workload_offset[block_id]; seq_offset += 32){
            int src_row = seq_start + seq_offset + idx*seqlen_per_block; // add bucket offset
            int dst_row = workload_offset[block_id] + seq_offset;
            int vec4_count = hidden_dim / 4;
            for (int i = lane_id; i < vec4_count; i += 32) {
                int h = i * 4;
                copy_float4(&merged_values[dst_row * hidden_dim + h], &values[src_row * hidden_dim + h]);
            }
        }
    }
}
void concat_2D_jagged_tensors_cuda_forward (
    const std::vector<torch::Tensor>& values_list,
    const std::vector<torch::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    int block_size,
    int grid_size,
    torch::Tensor workload_offset,
    torch::Tensor merged_values,
    torch::Tensor merged_offsets){

    int num_tensors = values_list.size();
    if (num_tensors == 0) {
        return; 
    }

    int hidden_dim = values_list[0].size(-1);

    int batch_size = offsets_list[0].size(0) - 1;


    assert(merged_values.is_contiguous());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        values_list[0].scalar_type(), 
        "concat_2D_jagged_tensors_forward_kernel",
        [&] {
            InputJaggedTensor<scalar_t> input_jagged_tensor_typed;
            for (int i = 0; i < num_tensors; ++i) {
                TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
                input_jagged_tensor_typed.value_list[i] = values_list[i].data_ptr<scalar_t>();
                input_jagged_tensor_typed.offsets_list[i] = offsets_list[i].data_ptr<int32_t>();
            }
            int blocks_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
            int total_blocks = batch_size * blocks_per_batch * num_tensors;

            // warp configuration: ensure not exceeding 1024 threads, each warp processes 1 sequence
            int target_warps = min(32, max(1, seqlen_per_block)); 
            int threads = min(block_size, target_warps * 32);
            //todo:python side max_grid_size not work now, use cudaDeviceProp to get max_grid_size
            // cudaDeviceProp prop;
            // cudaGetDeviceProperties(&prop, 0);
            // int max_grid_size = prop.maxGridSize[0];

            dim3 opt_blocks(min(grid_size, total_blocks));
            // dim3 opt_blocks(total_blocks);
            dim3 opt_threads(threads);
            
            //int elements_per_thread = (hidden_dim + 32 - 1) / 32;
            if(hidden_dim % 4 == 0){
                concat_2D_jagged_tensors_forward_kernel_warp<scalar_t><<<opt_blocks, opt_threads, 0, stream>>>(
                    input_jagged_tensor_typed,
                    num_tensors,
                    batch_size,
                    hidden_dim,
                    max_seqlen,
                    total_blocks,
                    seqlen_per_block,
                    workload_offset.data_ptr<int>(),
                    merged_values.data_ptr<scalar_t>(),
                    merged_offsets.data_ptr<int>()
                );
            }
            else {
                concat_2D_jagged_tensors_forward_kernel_opt_v4<scalar_t><<<opt_blocks, opt_threads, 0, stream>>>(
                    input_jagged_tensor_typed,
                    num_tensors,
                    batch_size,
                    hidden_dim,
                    max_seqlen,
                    total_blocks,
                    seqlen_per_block,
                    workload_offset.data_ptr<int>(),
                    merged_values.data_ptr<scalar_t>(),
                    merged_offsets.data_ptr<int>()
                );
            }

            
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return; 
}

template <typename T>
__launch_bounds__(1024, 2) __global__ void concat_2D_jagged_tensors_backward_kernel_opt_v4(
    const InputJaggedTensor<T> grad_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t hidden_dim,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    const int32_t seqlen_per_block,
    int* workload_offset,
    T* grad_output,
    int* merged_offsets) {
    // each block processes workload_offset[i+1]-workload_offset[i] sequences
    for(int block_id = blockIdx.x; block_id < total_blocks; block_id += gridDim.x){
        if(workload_offset[block_id] == workload_offset[block_id + 1]) continue;
        
        int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
        int tensor_id = (block_id / num_bucket_per_batch) % num_tensors;
        int batch_id =  block_id / num_bucket_per_batch / num_tensors; 
        int idx = block_id % num_bucket_per_batch; // which bucket
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        const int32_t* offsets = grad_jagged_tensor.offsets_list[tensor_id];
        T* values = grad_jagged_tensor.value_list[tensor_id];
        int seq_start = offsets[batch_id];
        int seq_end = offsets[batch_id + 1];
        int len = seq_end - seq_start;

        for(int seq_offset = warp_id; seq_offset < workload_offset[block_id+1]-workload_offset[block_id]; seq_offset += 32){
            // add bucket offset
            int src_row = seq_start + seq_offset + idx*seqlen_per_block; 
            int dst_row = workload_offset[block_id] + seq_offset;
            // warp-level parallelization for hidden_dim
            // each thread in warp handles multiple hidden dimensions
            int elements_per_thread = (hidden_dim + 32 - 1) / 32;
            int thread_start = lane_id * elements_per_thread;
            int thread_end = min(thread_start + elements_per_thread, hidden_dim);

            // vectorized copy 
            if (hidden_dim % 4 == 0 && thread_start % 4 == 0) {
                int vectorized_start = (thread_start + 3) / 4 * 4;
                int vectorized_end = thread_end / 4 * 4;
                
                for (int h = thread_start; h < vectorized_start && h < thread_end; ++h) {
                    values[src_row * hidden_dim + h] = grad_output[dst_row * hidden_dim + h];
                }
                
                for (int h = vectorized_start; h < vectorized_end; h += 4) {
                    copy_float4(&values[src_row * hidden_dim + h],
                                &grad_output[dst_row * hidden_dim + h]);
                }
                
                // handle remaining elements
                for (int h = vectorized_end; h < thread_end; ++h) {
                    values[src_row * hidden_dim + h] = grad_output[dst_row * hidden_dim + h];
                }
            } else {
                // scalar copy
                for (int h = thread_start; h < thread_end; ++h) {
                    values[src_row * hidden_dim + h] = grad_output[dst_row * hidden_dim + h];
                }
            }
        }
    }
}
template <typename T>
__launch_bounds__(1024, 2) __global__ void concat_2D_jagged_tensors_backward_kernel_warp(
    const InputJaggedTensor<T> grad_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t hidden_dim,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    const int32_t seqlen_per_block,
    int* workload_offset,
    T* grad_output,
    int* merged_offsets) {
    // each block processes workload_offset[i+1]-workload_offset[i] sequences
    for(int block_id = blockIdx.x; block_id < total_blocks; block_id += gridDim.x){
        if(workload_offset[block_id] == workload_offset[block_id + 1]) continue;
        
        int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
        int tensor_id = (block_id / num_bucket_per_batch) % num_tensors;
        int batch_id =  block_id / num_bucket_per_batch / num_tensors; 
        int idx = block_id % num_bucket_per_batch; // which bucket
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        const int32_t* offsets = grad_jagged_tensor.offsets_list[tensor_id];
        T* values = grad_jagged_tensor.value_list[tensor_id];
        int seq_start = offsets[batch_id];
        
        // for backward pass: copy gradients from merged grad_output back to individual input gradients
        // dst_row in grad_output -> src_row in values (individual tensor gradients)

        for(int seq_offset = warp_id; seq_offset < workload_offset[block_id+1]-workload_offset[block_id]; seq_offset += 32){
            int src_row = seq_start + seq_offset + idx*seqlen_per_block; // add bucket offset
            int dst_row = workload_offset[block_id] + seq_offset;
            int vec4_count = hidden_dim / 4;
            for (int i = lane_id; i < vec4_count; i += 32) {
                int h = i * 4;
                copy_float4(&values[src_row * hidden_dim + h], &grad_output[dst_row * hidden_dim + h]);
            }
        }
    }
}
void concat_2D_jagged_tensors_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_lengths,  
    int seqlen_per_block,
    int max_seqlen,
    int block_size,
    int grid_size,
    torch::Tensor workload_offset,
    const std::vector<torch::Tensor>& grad_inputs,
    const std::vector<torch::Tensor>& offsets_list,
    torch::Tensor merged_offsets) {

    int num_tensors = offsets_list.size();
    int batch_size = grad_lengths.size(0);
    int hidden_dim = grad_output.size(-1);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        grad_output.scalar_type(), 
        "concat_2D_jagged_tensors_backward_kernel",
        [&] {
            InputJaggedTensor<scalar_t> grad_jagged_tensor;
            for (int i = 0; i < num_tensors; ++i) {
                TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
                grad_jagged_tensor.value_list[i] = grad_inputs[i].data_ptr<scalar_t>();
                grad_jagged_tensor.offsets_list[i] = offsets_list[i].data_ptr<int32_t>();
            }
            int blocks_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
            int total_blocks = batch_size * blocks_per_batch * num_tensors;
            
            // warp configuration: ensure not exceeding 1024 threads, each warp processes 1 sequence
            int target_warps = min(32, max(1, seqlen_per_block)); 
            int threads = min(block_size, target_warps * 32);
            //todo:python side max_grid_size not work now, use cudaDeviceProp to get max_grid_size
            // cudaDeviceProp prop;
            // cudaGetDeviceProperties(&prop, 0);
            // int max_grid_size = prop.maxGridSize[0];
            dim3 opt_blocks(min(grid_size, total_blocks));
            dim3 opt_threads(threads);
            if(hidden_dim % 4 == 0){
                concat_2D_jagged_tensors_backward_kernel_warp<scalar_t><<<opt_blocks, opt_threads, 0, stream>>>(
                    grad_jagged_tensor,
                    num_tensors,
                    batch_size,
                    hidden_dim,
                    max_seqlen,
                    total_blocks,
                    seqlen_per_block,
                    workload_offset.data_ptr<int>(),
                    grad_output.data_ptr<scalar_t>(),
                    merged_offsets.data_ptr<int>()
                );
            }
            else {
                concat_2D_jagged_tensors_backward_kernel_opt_v4<scalar_t><<<opt_blocks, opt_threads, 0, stream>>>(
                    grad_jagged_tensor,
                    num_tensors,
                    batch_size,
                    hidden_dim,
                    max_seqlen,
                    total_blocks,
                    seqlen_per_block,
                    workload_offset.data_ptr<int>(),
                    grad_output.data_ptr<scalar_t>(),
                    merged_offsets.data_ptr<int>()
                );
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
    return;
}

__global__ void compute_block_workloads_cuda_kernel(
    const InputJaggedTensor<int32_t> input_jagged_tensor,
    const int32_t num_tensors,
    const int32_t batch_size,
    const int32_t seqlen_per_block,
    const int32_t max_seqlen,
    const int32_t total_blocks,
    int* block_workloads) {
    
    int work_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_id >= total_blocks) return;
    
    int num_bucket_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;

    int tensor_id = (work_id / num_bucket_per_batch) % num_tensors;
    int batch_id = work_id / num_bucket_per_batch / num_tensors; 
    int idx = work_id % num_bucket_per_batch; // which bucket
    const int32_t* offsets = input_jagged_tensor.offsets_list[tensor_id];
    int seq_start = offsets[batch_id];
    int seq_end = offsets[batch_id + 1];
    int len = seq_end - seq_start;

    int remaining_len = len - idx * seqlen_per_block;
    block_workloads[work_id] = max(0, min(remaining_len, seqlen_per_block));

    return;
}

/*
target: determine how many rows each block should process from merged_offsets
e.g. [4, 1, 4, 4, 4, 0, 4, 1, ...]

Working principle:
1. Each block is responsible for one bucket (sequence segment of seqlen_per_block size) of a batch
2. Calculate the total number of sequences for all tensors in that batch
3. Calculate the actual workload for that block based on bucket position
4. Output to block_workloads array for subsequent prefix sum calculation
*/
void compute_block_workloads_cuda(
    const std::vector<torch::Tensor>& offsets_list,
    int seqlen_per_block,
    int max_seqlen,
    torch::Tensor block_workloads) {

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    int num_tensors = offsets_list.size();
    int batch_size = offsets_list[0].size(0) - 1;
    int blocks_per_batch = (max_seqlen + seqlen_per_block - 1) / seqlen_per_block;
    int total_blocks = batch_size * blocks_per_batch * num_tensors;

    InputJaggedTensor<int32_t> offsets_jagged_tensor;
    for (int i = 0; i < num_tensors; ++i) {
        TORCH_CHECK(i < kMaxNumTensors, "Number of tensors exceeds kMaxNumTensors");
        offsets_jagged_tensor.offsets_list[i] = offsets_list[i].data_ptr<int32_t>();
    }
    int threads_per_block = min(1024, total_blocks);
    int num_blocks = (total_blocks + threads_per_block - 1) / threads_per_block;
    dim3 blocks(num_blocks);
    dim3 threads(threads_per_block);
    
    compute_block_workloads_cuda_kernel<<<blocks, threads, 0, stream>>>(
        offsets_jagged_tensor,
        num_tensors,
        batch_size,
        seqlen_per_block,
        max_seqlen,
        total_blocks,
        block_workloads.data_ptr<int>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
}