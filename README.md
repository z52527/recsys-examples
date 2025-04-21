# NVIDIA RecSys Examples

# Overview
NVIDIA RecSys Examples is a collection of optimized recommender models and components. 

The project includes:
- Examples for large-scale HSTU ranking and retrieval models through TorchRec and Megatron-Core integration
- HSTU (Hierarchical Sequential Transduction Unit) attention operator support
- Dynamic Embeddings with GPU acceleration

# Get Started

Before running examples, install libs under corelib following instruction in documentaion:
- [HSTU attention documentation](./corelib/hstu/README.md)
- [Dynamic Embeddings documentation](./corelib/dynamicemb/README.md)

The examples we supported:
- [HSTU recommender examples](./examples/hstu/README.md)

# Contribution Guidelines
Please see our [contributing guidelines](./CONTRIBUTING.md) for details on how to contribute to this project.

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