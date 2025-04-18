# HSTU Attention Backend Benchmark
## Methodology
We developed a generative recommender ranking model utilizing the HSTU block and benchmarked it against the `CUTLASS` and `Triton` backends. 
* Model Structure
    * 400M-row, 256-dimensional embedding table.
    * 8 HSTU layers, with 4 attention heads and a head dimension of 256.
    * The total model parameter size is ~120B.
* Input Specs
    * 8192 sequence length per sample with 4096 item tokens and 4096 action tokens
* Training Specs
    * Adam optimizer
    * CPU offloading is used to make this model trainable in single `DGX`. The total size of the embedding table and its optimizer state is approximately 1.2 TB, which exceeds the GPU memory capacity of a single `DGX-H100` or `DGX-A100`. Consequently, we retain only 10% of the embedding table in GPU memory and offload the remainder to CPU memory.
    * Batch_size is 32
    * Precision is bfloat16
    * For detailed model configuration, please refer to [benchmark_ranking.gin](../../examples/hstu/benchmark_ranking.gin).

Our benchmark was conducted on two platforms: the `DGX-H100` and `DGX-A100`, each equipped with 8xGPUs connected via `NVLink`.

## How to Run
We recommend using single `DGX-H100` or single `DGX-A100` to run the benchmark and ensuring that there is sufficient CPU memory (greater than 1.1 TB) to accommodate the offloaded embedding table. You can modify the `NetworkArgs.kernel_backend` in the configuration file to switch between the `Triton` and `CUTLASS` backends.
```bash
torchrun --nproc_per_node 8 --master_addr localhost --master_port 6000 examples/hstu/pretrain_gr_ranking.py --gin-config-file examples/hstu/benchmark_ranking.gin
```
To collect to the nsys results, you can use the following command:
```bash
nsys profile -s none -t cuda,nvtx -f true -o hstu_backend.%p -c cudaProfilerApi --cpuctxsw none --cuda-flush-interval 100 --capture-range-end=stop --cuda-graph-trace=node torchrun --nproc_per_node 8 --master_addr localhost --master_port 6000 examples/hstu/pretrain_gr_ranking.py --gin-config-file examples/hstu/benchmark_ranking.gin
```
## Benchmark Result
`CUTLASS` backend can achieve around `3.25x` end-to-end speedup on `DGX-A100`, and `1.89x` on `DGX-H100` compared with `Triton` backend.

| Platform | Triton(k tokens/s) | CUTLASS(k tokens/s) | Speedup |
| --- | --- | --- | --- |
| DGX-H100 | 103.2 | 195.6 | 1.89x |
| DGX-A100 | 26.5 | 86.3 | 3.25x |
*The version of distributed-recommender is v0.0.1*
<!-- H100, CUTLASS, 52428800/268s, 195.6k tokens/s -->
<!-- H100, Triton, 52428800/508s, 103.2k tokens/s -->
<!-- A100, CUTLASS, 52428800/607s, 86.3k tokens/s -->
<!-- A100, Triton, 52428800/1976s, 26.5k tokens/s-->