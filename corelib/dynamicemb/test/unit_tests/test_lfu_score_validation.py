# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb.dump_load import (
    DynamicEmbDump,
    find_sharded_modules,
    get_dynamic_emb_module,
)
from dynamicemb.dynamicemb_config import (
    dtype_to_bytes,
    dyn_emb_to_torch,
)
from dynamicemb_extensions import dyn_emb_cols
from test_embedding_dump_load import (
    create_model,
    get_optimizer_kwargs,
    idx_to_name,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def generate_deterministic_sparse_features_with_frequency_tracking(
    num_embedding_collections: int,
    num_embeddings: List[int],
    multi_hot_sizes: List[int],
    rank: int,
    world_size: int,
    batch_size: int,
    num_iterations: int,
    seed: int = 42,
) -> Tuple[List[KeyedJaggedTensor], Dict[str, Dict[int, int]]]:
    """
    Generate deterministic sparse features and track frequency for each embedding table.

    Returns:
        kjts: List of KeyedJaggedTensor for each iteration
        table_frequency_counters: Dict mapping table_name -> {key: frequency}
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size_per_rank = batch_size // world_size
    kjts = []
    table_frequency_counters = {}

    # Initialize frequency counters for each table
    for embedding_collection_id in range(num_embedding_collections):
        for embedding_id, num_embedding in enumerate(num_embeddings):
            _, embedding_name = idx_to_name(embedding_collection_id, embedding_id)
            table_frequency_counters[embedding_name] = defaultdict(int)

    for iteration in range(num_iterations):
        cur_indices = []
        cur_lengths = []
        keys = []

        for embedding_collection_id in range(num_embedding_collections):
            for embedding_id, num_embedding in enumerate(num_embeddings):
                feature_name, embedding_name = idx_to_name(
                    embedding_collection_id, embedding_id
                )

                for sample_id in range(batch_size):
                    hotness = random.randint(
                        1, multi_hot_sizes[embedding_collection_id]
                    )  # At least 1
                    # Generate indices with smaller range to ensure duplicates
                    max_key = min(
                        num_embedding - 1, 100
                    )  # Limit to first 100 keys for more duplicates
                    indices = [random.randint(0, max_key) for _ in range(hotness)]

                    # Track frequency for all generated indices
                    for idx in indices:
                        table_frequency_counters[embedding_name][idx] += 1

                    if sample_id // batch_size_per_rank == rank:
                        cur_indices.extend(indices)
                        cur_lengths.append(hotness)

                keys.append(feature_name)

        kjts.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(cur_indices, dtype=torch.int64).cuda(),
                lengths=torch.tensor(cur_lengths, dtype=torch.int64).cuda(),
            )
        )

    # Convert defaultdicts to regular dicts
    for table_name in table_frequency_counters:
        table_frequency_counters[table_name] = dict(
            table_frequency_counters[table_name]
        )

    return kjts, table_frequency_counters


def load_dumped_keys_scores_only(
    path: str,
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    optim: bool = False,
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[int, int]]:
    """
    仿照DynamicEmbLoad的逻辑，但只读取key和score到字典中，不插入到表

    Returns:
        Dict[table_name, Dict[key, score]]
    """
    if not os.path.exists(path):
        raise Exception("can't find path to load, path:", path)

    collections_list = find_sharded_modules(model, "")
    if len(collections_list) == 0:
        print("Warning: No sharded embedding collections found")
        return {}

    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    all_table_scores = {}

    for _, current_collection in enumerate(collections_list):
        (
            collection_path,
            current_collection_name,
            current_collection_module,
        ) = current_collection
        full_collection_path = os.path.join(path, collection_path)
        current_dynamic_emb_module_list = get_dynamic_emb_module(
            current_collection_module
        )

        for _, dynamic_emb_module in enumerate(current_dynamic_emb_module_list):
            current_table_names = dynamic_emb_module.table_names
            current_tables = dynamic_emb_module.tables

            for dynamic_table_name, dynamic_table in zip(
                current_table_names, current_tables
            ):
                if table_names is not None and dynamic_table_name not in set(
                    table_names[current_collection_name]
                ):
                    continue

                print(f"Loading scores for table: {dynamic_table_name}")

                # 读取这个表的key和score
                table_scores = load_table_keys_scores_only(
                    dynamic_table,
                    full_collection_path,
                    dynamic_table_name,
                    rank,
                    world_size,
                    optim=optim,
                )

                all_table_scores[dynamic_table_name] = table_scores
                print(
                    f"Loaded {len(table_scores)} key-score pairs for {dynamic_table_name}"
                )

    return all_table_scores


def load_table_keys_scores_only(
    dynamic_table,
    root_path: str,
    name: str,
    rank: int,
    world_size: int,
    batch_size: int = 65536,
    optim: bool = False,
) -> Dict[int, int]:
    """
    仿照distributed_load的逻辑，但只读取key和score到字典中

    Returns:
        Dict[key, score]
    """
    # 修正文件名格式，包含rank和world_size后缀
    key_name = f"{name}_emb_keys.rank_{rank}.world_size_{world_size}"
    value_name = f"{name}_emb_values.rank_{rank}.world_size_{world_size}"
    score_name = f"{name}_emb_scores.rank_{rank}.world_size_{world_size}"

    key_path = os.path.join(root_path, key_name)
    value_path = os.path.join(root_path, value_name)
    score_path = os.path.join(root_path, score_name)

    if not os.path.exists(key_path):
        print(f"Warning: Key file not found: {key_path}")
        return {}

    if not os.path.exists(value_path):
        print(f"Warning: Value file not found: {value_path}")
        return {}

    if not os.path.exists(score_path):
        print(f"Warning: Score file not found: {score_path}")
        return {}

    key_file_size = os.path.getsize(key_path)
    value_file_size = os.path.getsize(value_path)
    score_file_size = os.path.getsize(score_path)

    if key_file_size == 0 or value_file_size == 0 or score_file_size == 0:
        print(f"Warning: Empty files for table {name}")
        return {}

    key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
    value_dtype = dyn_emb_to_torch(dynamic_table.value_type())

    key_bytes = dtype_to_bytes(key_dtype)
    value_bytes = dtype_to_bytes(value_dtype)

    total_keys = key_file_size // key_bytes
    total_dim = value_file_size // (total_keys * value_bytes)

    dim = dyn_emb_cols(dynamic_table)
    optstate_dim = dynamic_table.optstate_dim()

    if total_dim < dim or ((total_dim != dim + optstate_dim) and optim):
        print(f"Warning: Dimension mismatch for table {name}")
        return {}

    keys_read_bytes = batch_size * 8  # key in file always int64
    values_read_bytes = batch_size * total_dim * 4  # value in file always float
    scores_read_bytes = batch_size * 8  # score in file always uint64

    table_key_scores = {}

    with open(key_path, "rb") as fkey, open(value_path, "rb") as fvalue, open(
        score_path, "rb"
    ) as fscore:
        while True:
            remaining_key_bytes = key_file_size - fkey.tell()
            remaining_value_bytes = value_file_size - fvalue.tell()
            remaining_score_bytes = score_file_size - fscore.tell()

            if (
                remaining_key_bytes <= 0
                or remaining_value_bytes <= 0
                or remaining_score_bytes <= 0
            ):
                break

            key_bytes_to_read = min(keys_read_bytes, remaining_key_bytes)
            value_bytes_to_read = min(values_read_bytes, remaining_value_bytes)
            score_bytes_to_read = min(scores_read_bytes, remaining_score_bytes)

            key_bytes_data = fkey.read(key_bytes_to_read)
            value_bytes_data = fvalue.read(value_bytes_to_read)
            score_bytes_data = fscore.read(score_bytes_to_read)

            num_keys = len(key_bytes_data) // 8  # key in file always int64

            key_array = np.frombuffer(key_bytes_data, dtype=np.int64)
            value_array = np.frombuffer(value_bytes_data, dtype=np.float32).reshape(
                -1, total_dim
            )
            score_array = np.frombuffer(score_bytes_data, dtype=np.uint64)

            if len(key_array) != len(score_array):
                print(
                    f"Warning: Key-score length mismatch in table {name}: {len(key_array)} vs {len(score_array)}"
                )
                continue

            # 根据rank过滤数据（模仿distributed_load的逻辑）
            mask = key_array % world_size == rank
            masked_keys = key_array[mask]
            masked_scores = score_array[mask]

            # 存储到字典中
            for key, score in zip(masked_keys, masked_scores):
                table_key_scores[int(key)] = int(score)

    return table_key_scores


def validate_lfu_scores(
    expected_frequencies: Dict[str, Dict[int, int]],
    actual_scores: Dict[str, Dict[int, int]],
    tolerance: float = 0.0,
) -> Tuple[bool, str]:
    """
    Validate that actual scores match expected frequencies for LFU strategy.

    Returns:
        (is_valid, error_message)
    """
    all_errors = []

    for table_name in expected_frequencies:
        if table_name not in actual_scores:
            all_errors.append(f"Table {table_name} missing from actual scores")
            continue

        expected = expected_frequencies[table_name]
        actual = actual_scores[table_name]

        # Check missing keys
        missing_keys = set(expected.keys()) - set(actual.keys())
        if missing_keys:
            all_errors.append(
                f"Table {table_name}: missing keys {list(missing_keys)[:5]}"
            )

        # Check frequency matching
        frequency_errors = []
        for key in set(expected.keys()) & set(actual.keys()):
            exp_freq = expected[key]
            act_score = actual[key]

            if tolerance > 0:
                if abs(act_score - exp_freq) / max(exp_freq, 1) > tolerance:
                    frequency_errors.append(
                        f"key {key}: expected {exp_freq}, got {act_score}"
                    )
            else:
                if act_score != exp_freq:
                    frequency_errors.append(
                        f"key {key}: expected {exp_freq}, got {act_score}"
                    )

        if frequency_errors:
            all_errors.append(
                f"Table {table_name}: {frequency_errors[:3]}"
            )  # Show first 3

    is_valid = len(all_errors) == 0
    error_message = (
        "; ".join(all_errors)
        if all_errors
        else "All LFU scores match expected frequencies"
    )

    return is_valid, error_message


@click.command()
@click.option("--num-embedding-collections", type=int, default=1)
@click.option("--num-embeddings", type=str, default="1000")
@click.option("--multi-hot-sizes", type=str, default="3")
@click.option("--embedding-dim", type=int, default=32)
@click.option("--save-path", type=str, default="debug_weight")
@click.option(
    "--optimizer-type",
    type=click.Choice(["sgd", "adam", "adagrad", "rowwise_adagrad"]),
    default="sgd",
)
@click.option("--batch-size", type=int, default=16)
@click.option("--num-iterations", type=int, default=3)
@click.option("--tolerance", type=float, default=0.0)
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option(
    "--use-dump-validation",
    is_flag=True,
    help="Use dump files for validation instead of direct model access",
)
def test_lfu_score_validation(
    num_embedding_collections: int,
    num_embeddings: str,
    multi_hot_sizes: str,
    embedding_dim: int,
    optimizer_type: str,
    save_path: str,
    batch_size: int,
    num_iterations: int,
    tolerance: float,
    debug: bool,
    use_dump_validation: bool,
):
    """Test LFU score correctness by comparing with naive frequency counting."""
    if dist.get_world_size() > 1:
        raise ValueError(
            "Multi-rank LFU testing not yet supported due to all-to-all complexity"
        )

    num_embeddings = [int(v) for v in num_embeddings.split(",")]
    multi_hot_sizes = [int(v) for v in multi_hot_sizes.split(",")]

    print(f"\n{'='*60}")
    print(f"LFU SCORE VALIDATION TEST")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Embedding collections: {num_embedding_collections}")
    print(f"  - Num embeddings: {num_embeddings}")
    print(f"  - Multi-hot sizes: {multi_hot_sizes}")
    print(f"  - Embedding dim: {embedding_dim}")
    print(f"  - Optimizer: {optimizer_type}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Tolerance: {tolerance}")
    print(f"  - Use dump validation: {use_dump_validation}")

    # Create model
    optimizer_kwargs = get_optimizer_kwargs(optimizer_type)
    model = create_model(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        optimizer_kwargs=optimizer_kwargs,
    )

    # Generate features with frequency tracking
    (
        kjts,
        expected_frequencies,
    ) = generate_deterministic_sparse_features_with_frequency_tracking(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        multi_hot_sizes=multi_hot_sizes,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        batch_size=batch_size,
        num_iterations=num_iterations,
    )

    print(f"\nExpected frequencies:")
    for table_name, freqs in expected_frequencies.items():
        print(
            f"  {table_name}: {len(freqs)} unique keys, total frequency: {sum(freqs.values())}"
        )
        print(f"    Sample: {dict(list(freqs.items())[:10])}")

    # Run forward passes to populate frequency information
    print(f"\nRunning {num_iterations} iterations...")
    for iteration, kjt in enumerate(kjts):
        ret = model(kjt)
        loss = ret.sum() * dist.get_world_size()
        loss.backward()
        if debug:
            print(f"  Iteration {iteration + 1} completed")

    print(f"\nDumping model to {save_path}...")
    shutil.rmtree(save_path, ignore_errors=True)
    DynamicEmbDump(save_path, model, optim=True)

    # Load keys and scores from dump files (without inserting to tables)
    print(f"\nLoading keys and scores from dump files...")
    actual_scores = load_dumped_keys_scores_only(save_path, model, optim=True)

    print(f"\nActual scores:")
    for table_name, scores in actual_scores.items():
        print(f"  {table_name}: {len(scores)} keys")
        if len(scores) > 0:
            print(f"    Sample: {dict(list(scores.items())[:3])}")

    # Validate scores
    is_valid, error_message = validate_lfu_scores(
        expected_frequencies, actual_scores, tolerance
    )

    # Report results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULT: {'PASS' if is_valid else 'FAIL'}")
    print(f"{'='*60}")
    print(f"Details: {error_message}")

    if not is_valid:
        print(f"\nDEBUG INFO:")
        for table_name in expected_frequencies:
            expected = expected_frequencies[table_name]
            actual = actual_scores.get(table_name, {})
            print(f"\nTable {table_name}:")
            print(
                f"  Expected: {len(expected)} keys, sample: {dict(list(expected.items())[:5])}"
            )
            print(
                f"  Actual: {len(actual)} keys, sample: {dict(list(actual.items())[:5])}"
            )

            # Show mismatches
            common_keys = set(expected.keys()) & set(actual.keys())
            mismatches = [
                (k, expected[k], actual[k])
                for k in common_keys
                if expected[k] != actual[k]
            ]
            if mismatches:
                print(f"  Mismatches (first 5): {mismatches[:5]}")

        raise AssertionError(f"LFU score validation failed: {error_message}")

    print(f"\n✓ LFU score validation PASSED! All frequencies match correctly.")
    return True


if __name__ == "__main__":
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(backend="nccl")
    try:
        test_lfu_score_validation()
    finally:
        dist.destroy_process_group()
