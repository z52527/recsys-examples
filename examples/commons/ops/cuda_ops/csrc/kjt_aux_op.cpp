#include <ATen/ATen.h>
#include <torch/library.h>
#include <vector>

namespace {

std::vector<at::Tensor> split_by_lengths_impl(
    const at::Tensor& values,
    const at::Tensor& lengths_1d,
    int64_t num_splits) {
  // Compute split sizes on CPU so we can robustly support:
  // - values on CPU with lengths on CPU/CUDA
  // - values on CUDA with lengths on CPU/CUDA
  const int64_t batch = lengths_1d.numel() / num_splits;
  at::Tensor lengths_i64_cpu = lengths_1d.to(at::kCPU, at::kLong).reshape({num_splits, batch});
  at::Tensor per_split_sizes_cpu = lengths_i64_cpu.sum(/*dim=*/1); // [num_splits] on CPU

    const int64_t total_values = values.size(0);
  const int64_t total_split_sizes = per_split_sizes_cpu.sum().item<int64_t>();
  TORCH_CHECK(
      total_split_sizes == total_values,
      "sum(lengths_1d)=", total_split_sizes, " must equal values.size(0)=", total_values);

  std::vector<at::Tensor> out;
  out.reserve(static_cast<size_t>(num_splits));

  int64_t start = 0;
  for (int64_t i = 0; i < num_splits; ++i) {
    const int64_t len = per_split_sizes_cpu[i].item<int64_t>();
    out.push_back(values.narrow(/*dim=*/0, /*start=*/start, /*length=*/len));
    start += len;
  }

  return out;
}

std::vector<at::Tensor> split_by_lengths_cpu(
    const at::Tensor& values,
    const at::Tensor& lengths_1d,
    int64_t num_splits) {
  TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor");
  TORCH_CHECK(
      lengths_1d.device().is_cpu() || lengths_1d.is_cuda(),
      "lengths_1d must be CPU or CUDA, got ", lengths_1d.device());
  TORCH_CHECK(
      values.dim() == 1 || values.dim() == 2,
      "values must be 1D or 2D, got dim=", values.dim());
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  return split_by_lengths_impl(values, lengths_1d, num_splits);
}

std::vector<at::Tensor> split_by_lengths_cuda(
    const at::Tensor& values,
    const at::Tensor& lengths_1d,
    int64_t num_splits) {
  TORCH_CHECK(values.device().is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(
      lengths_1d.device().is_cpu() || lengths_1d.is_cuda(),
      "lengths_1d must be CPU or CUDA, got ", lengths_1d.device());
  TORCH_CHECK(
      values.dim() == 1 || values.dim() == 2,
      "values must be 1D or 2D, got dim=", values.dim());
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  return split_by_lengths_impl(values, lengths_1d, num_splits);
}

at::Tensor lengths_reduce_dim1_impl(
    const at::Tensor& lengths_1d,
    int64_t num_splits,
    bool expect_cuda) {
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  if (expect_cuda) {
    TORCH_CHECK(lengths_1d.is_cuda(), "lengths_1d must be a CUDA tensor for CUDA impl");
  } else {
    TORCH_CHECK(lengths_1d.device().is_cpu(), "lengths_1d must be a CPU tensor for CPU impl");
  }

  const int64_t batch = lengths_1d.numel() / num_splits;
  return lengths_1d.view({num_splits, batch}).sum(1);
}

at::Tensor lengths_reduce_dim1_cpu(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_reduce_dim1_impl(lengths_1d, num_splits, /*expect_cuda=*/false);
}

at::Tensor lengths_reduce_dim1_cuda(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_reduce_dim1_impl(lengths_1d, num_splits, /*expect_cuda=*/true);
}

std::vector<at::Tensor> lengths_splits_impl(
    const at::Tensor& lengths_1d,
    int64_t num_splits,
    bool expect_cuda) {
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  if (expect_cuda) {
    TORCH_CHECK(lengths_1d.is_cuda(), "lengths_1d must be a CUDA tensor for CUDA impl");
  } else {
    TORCH_CHECK(lengths_1d.device().is_cpu(), "lengths_1d must be a CPU tensor for CPU impl");
  }

  const int64_t batch = lengths_1d.numel() / num_splits;
  std::vector<at::Tensor> out;
  out.reserve(static_cast<size_t>(num_splits));

  for (int64_t i = 0; i < num_splits; ++i) {
    out.push_back(lengths_1d.narrow(/*dim=*/0, /*start=*/i*batch, /*length=*/batch));
  }

  return out;
}

std::vector<at::Tensor> lengths_splits_cpu(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_splits_impl(lengths_1d, num_splits, /*expect_cuda=*/false);
}

std::vector<at::Tensor> lengths_splits_cuda(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_splits_impl(lengths_1d, num_splits, /*expect_cuda=*/true);
}


std::vector<at::Tensor> permute_and_split_impl(
    const at::Tensor& jagged_features,
    const at::Tensor& jagged_lengths,
    const at::Tensor& jagged_offsets,
    int64_t num_static_features,
    int64_t num_dynamic_features,
    const std::vector<int64_t>& features_order,
    bool expect_cuda) {
  int64_t num_features = num_static_features + num_dynamic_features;
  TORCH_CHECK(jagged_features.dim() == 1, "jagged_features must be 1D, got dim=", jagged_features.dim());
  TORCH_CHECK(jagged_lengths.dim() == 1, "jagged_lengths must be 1D, got dim=", jagged_lengths.dim());
  TORCH_CHECK(num_static_features > 0, "num_static_features must be > 0");
  TORCH_CHECK(num_dynamic_features > 0, "num_dynamic_features must be > 0");
  TORCH_CHECK(
      jagged_lengths.numel() % num_features == 0,
      "jagged_lengths.numel()=", jagged_lengths.numel(), " must be divisible by num_features=", num_features);

  if (expect_cuda) {
    TORCH_CHECK(jagged_features.is_cuda(), "jagged_features must be a CUDA tensor for CUDA impl");
    TORCH_CHECK(jagged_lengths.is_cuda(), "jagged_lengths must be a CUDA tensor for CUDA impl");
  } else {
    TORCH_CHECK(jagged_features.device().is_cpu(), "jagged_features must be a CPU tensor for CPU impl");
    TORCH_CHECK(jagged_lengths.device().is_cpu(), "jagged_lengths must be a CPU tensor for CPU impl");
  }
  TORCH_CHECK(num_features == features_order.size(), "features_order size must match total number of features");

  const int64_t batch = jagged_lengths.numel() / num_features;

  std::vector<at::Tensor> permuted_lengths_vec(num_features);
  for (int64_t i = 0; i < num_features; ++i) {
    permuted_lengths_vec[i] = jagged_lengths.narrow(/*dim=*/0, /*start=*/features_order[i] * batch, /*length=*/batch);
  }
  auto permuted_lengths = at::cat(permuted_lengths_vec, /*dim=*/0);

  std::vector<at::Tensor> static_features_vec(num_static_features);
  std::vector<at::Tensor> dynamic_features_vec(num_dynamic_features);

  auto jagged_offsets_cpu = jagged_offsets.to(at::kCPU);

  for (int64_t i = 0; i < num_static_features; ++i) {
    auto numel = permuted_lengths_vec[i].sum().item<int64_t>();
    auto original_index = features_order[i] * batch;
    static_features_vec[i] = jagged_features.narrow(/*dim=*/0, /*start=*/jagged_offsets_cpu[original_index].item<int64_t>(), /*length=*/numel);
  }

  for (int64_t i = 0; i < num_dynamic_features; ++i) {
    auto numel = permuted_lengths_vec[num_static_features + i].sum().item<int64_t>();
    auto original_index = features_order[num_static_features + i] * batch;
    dynamic_features_vec[i] = jagged_features.narrow(/*dim=*/0, /*start=*/jagged_offsets_cpu[original_index].item<int64_t>(), /*length=*/numel);
  }

  auto static_lengths = permuted_lengths.narrow(/*dim=*/0, /*start=*/0, /*length=*/batch * num_static_features);
  auto dynamic_lengths = permuted_lengths.narrow(/*dim=*/0, /*start=*/batch * num_static_features, /*length=*/batch * num_dynamic_features);

  auto static_features = at::cat(static_features_vec, /*dim=*/0);
  auto dynamic_features = at::cat(dynamic_features_vec, /*dim=*/0);

  std::vector<at::Tensor> out{ static_features, dynamic_features, static_lengths, dynamic_lengths };
  return out;
}

std::vector<at::Tensor> permute_and_split_cpu(
    const at::Tensor& jagged_features,
    const at::Tensor& jagged_lengths,
    const at::Tensor& jagged_offsets,
    int64_t num_static_features,
    int64_t num_dynamic_features,
    const std::vector<int64_t>& features_order
) {
  return permute_and_split_impl(jagged_features, jagged_lengths, jagged_offsets, num_static_features, num_dynamic_features, features_order, /*expect_cuda=*/false);
}

std::vector<at::Tensor> permute_and_split_cuda(
    const at::Tensor& jagged_features,
    const at::Tensor& jagged_lengths,
    const at::Tensor& jagged_offsets,
    int64_t num_static_features,
    int64_t num_dynamic_features,
    const std::vector<int64_t>& features_order
) {
  return permute_and_split_impl(jagged_features, jagged_lengths, jagged_offsets, num_static_features, num_dynamic_features, features_order, /*expect_cuda=*/true);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(hstu_cuda_ops, m) {
  m.def("split_by_lengths(Tensor values, Tensor lengths_1d, int num_splits) -> Tensor[]");
  m.def("lengths_reduce_dim1(Tensor lengths_1d, int num_splits) -> Tensor");
  m.def("lengths_splits(Tensor lengths_1d, int num_splits) -> Tensor[]");
  m.def("permute_and_split(Tensor jagged_features, Tensor jagged_lengths, Tensor jagged_offsets, int num_static_features, int num_dynamic_features, int[] features_order) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(hstu_cuda_ops, CPU, m) {
  m.impl("split_by_lengths", split_by_lengths_cpu);
  m.impl("lengths_reduce_dim1", lengths_reduce_dim1_cpu);
  m.impl("lengths_splits", lengths_splits_cpu);
  m.impl("permute_and_split", permute_and_split_cpu);
}

TORCH_LIBRARY_IMPL(hstu_cuda_ops, CUDA, m) {
  m.impl("split_by_lengths", split_by_lengths_cuda);
  m.impl("lengths_reduce_dim1", lengths_reduce_dim1_cuda);
  m.impl("lengths_splits", lengths_splits_cuda);
  m.impl("permute_and_split", permute_and_split_cuda);
}