#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

hbm=4
use_index_dedups=("True")
batch_sizes=(65536 1048576)
capacities=("8" "64")
optimizer_types=("sgd" "adam")

rm benchmark_results.json
for use_index_dedup in "${use_index_dedups[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for capacity in "${capacities[@]}"; do
      for optimizer_type in "${optimizer_types[@]}"; do
  
        echo "####" $use_index_dedup $batch_size $capacity $hbm $optimizer_type
        torchrun --nnodes 1 --nproc_per_node 1 \
          ./benchmark/benchmark_batched_dynamicemb_tables.py  \
            --use_index_dedup $use_index_dedup \
            --batch_size $batch_size \
            --num_embeddings_per_feature $capacity \
            --hbm_for_embeddings $hbm \
            --optimizer_type $optimizer_type
      done
    done
  done
done