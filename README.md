# NVIDIA RecSys Examples

# Overview
NVIDIA RecSys Examples is a collection of optimized recommender models and components. 

The project includes:
- Examples for large-scale HSTU ranking and retrieval models through [TorchRec](https://github.com/pytorch/torchrec) and [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) integration
- HSTU (Hierarchical Sequential Transduction Unit) attention operator support
- Dynamic Embeddings with GPU acceleration

# What's New
- **[2025/7/4]** 🎉v25.06 released!
  - Dynamicemb lookup module performance improvement and LFU eviction support. 
  - Pipeline support for HSTU example, recompute support for HSTU layer and customized cuda ops for jagged tensor concat.
- **[2025/5/29]** 🎉v25.05 released! 
  - Enhancements to the dynamicemb functionality, including support for EmbeddingBagCollection, truncated normal initialization, and initial_accumulator_value for Adagrad.
  - Fusion of operations like layernorm and dropout in the HSTU layer, resulting in about 1.2x end-to-end speedup.
  - Fix convergence issues on the Kuairand dataset.

For more detailed release notes, please refer our [releases](https://github.com/NVIDIA/recsys-examples/releases).

# Environment Setup
## Start from dockerfile

We provide [dockerfile](./docker/Dockerfile) for users to build environment. 
```
docker build -f docker/Dockerfile --platform linux/amd64 -t recsys-examples:latest .
```
If you want to build image for Grace, you can use 
```
docker build -f docker/Dockerfile --platform linux/arm64 -t recsys-examples:latest .
```
You can also set your own base image with args `--build-arg <BASE_IMAGE>`.

## Start from source file
Before running examples, build and install libs under corelib following instruction in documentation:
- [HSTU attention documentation](./corelib/hstu/README.md)
- [Dynamic Embeddings documentation](./corelib/dynamicemb/README.md)

On top of those two core libs, Megatron-Core along with other libs are required. You can install them via pypi package:

```bash
pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath megatron-core==0.9.0
```

If you fail to install the megatron-core package, usually due to the python version incompatibility, please try to clone and then install the source code. 

```bash
git clone -b core_r0.9.0 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
pip install -e ./megatron-lm
```

We provide our custom HSTU CUDA operators for enhanced performance. You need to install these operators using the following command:

```bash
cd /workspace/recsys-examples/examples/hstu && \
python setup.py install
```

# Get Started
The examples we supported:
- [HSTU recommender examples](./examples/hstu/README.md)

# Contribution Guidelines
Please see our [contributing guidelines](./CONTRIBUTING.md) for details on how to contribute to this project.

# Resources
## Video
- [RecSys Examples 中的训练与推理优化实践](https://www.bilibili.com/video/BV1msMwzpE5B?buvid=638d217658211387f0a20e730604a780&from_spmid=united.player-video-detail.drama-float.0&is_story_h5=false&mid=V%2FD40L0stVy%2BZTgWdpjtGA%3D%3D&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=2DD6CE30-B189-4EEC-9FD4-8BAD6AEFE720&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1749773222&unique_k=Sjcfmgy&up_id=1320140761&vd_source=7372540fd02b24a46851135aa003577c)
- [基于CUTLASS 3 的HSTU attention 算子开发与优化](https://www.bilibili.com/video/BV1TsMwzWEzS?buvid=638d217658211387f0a20e730604a780&from_spmid=united.player-video-detail.drama-float.0&is_story_h5=false&mid=V%2FD40L0stVy%2BZTgWdpjtGA%3D%3D&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=2DD6CE30-B189-4EEC-9FD4-8BAD6AEFE720&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1749773222&unique_k=Sjcfmgy&up_id=1320140761&vd_source=7372540fd02b24a46851135aa003577c&spm_id_from=333.788.videopod.sections)

## Blog
- [NVIDIA recsys-examples: 生成式推荐系统大规模训练推理的高效实践（上篇）](https://mp.weixin.qq.com/s/K9xtYC3azAccShpJ3ZxKbg)

# Community
Join our community channels to ask questions, provide feedback, and interact with other users and developers:
- GitHub Issues: For bug reports and feature requests
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

# References
If you use RecSys Examples in your research, please cite:

```
@Manual{,
  title = {RecSys Examples: A collection of recommender system implementations},
  author = {NVIDIA Corporation},
  year = {2024},
  url = {https://github.com/NVIDIA/recsys-examples},
}
```

For more citation information and referenced papers, see [CITATION.md](./CITATION.md).

# License
This project is licensed under the Apache License - see the [LICENSE](./LICENSE) file for details.