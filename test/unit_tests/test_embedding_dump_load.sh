#!/bin/bash 
set -e

export PYHKV_DEBUG=1
export PYHKV_DEBUG_ITER=10
export DYNAMICEMB_DUMP_LOAD_DEBUG=1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True --batch_size 1024 || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --batch_size 1024 || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 129 --use_index_dedup True --batch_size 1024 || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 129 --use_index_dedup False --batch_size 1024 || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 15 --use_index_dedup True --batch_size 1023 --multi_hot_sizes=20,1,101,49 || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 15 --use_index_dedup False --batch_size 1023  --multi_hot_sizes=20,1,101,49 || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 15 --use_index_dedup True --batch_size 1023  --multi_hot_sizes=20,49,101,1 || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 15 --use_index_dedup False --batch_size 1023  --multi_hot_sizes=20,1,101,49 --score_type="step" || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 15 --use_index_dedup True --batch_size 1023  --multi_hot_sizes=20,49,101,1  --score_type="step" || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 15 --use_index_dedup False --batch_size 1023  --multi_hot_sizes=20,1,101,49 --score_type="custimized" || exit 1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_embedding_dump_load.py --print_sharding_plan --optimizer_type "adam" --embedding_dim 15 --use_index_dedup True --batch_size 1023  --multi_hot_sizes=20,49,101,1  --score_type="custimized" || exit 1
