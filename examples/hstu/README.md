# Examples: to show how to train generative ranking/retrieval models
## Environment
You can refer to Installation in the README to set up the environment.
## Prepare Dataset
### Dataset Information
#### **MovieLens**
refer to [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
#### **KuaiRand**

| dataset       | # users | seqlen max | seqlen min | seqlen mean | seqlen median | # items    |
|---------------|---------|------------|------------|-------------|---------------|------------|
| kuairand_pure | 27285   | 910        | 1          | 1           | 39            | 7551       |
| kuairand_1k   | 1000    | 49332      | 10         | 5038        | 3379          | 4369953    |
| kuairand_27k  | 27285   | 228000     | 100        | 11796       | 8591          | 32038725   |
 
refer to [KuaiRand](https://kuairand.com/).

### Preprocessing
We provides preprocessor scripts to assist in downloading raw data if it is not already present. It processes the raw data into csv files.
```
mkdir -p ./tmp_data && python3 data/preprocessor.py --dataset_name ml-1m
```
The following dataset name is supported:
* ml-1m
* ml-20m
* kuairand-pure
* kuairand-1K
* kuairand-27K
* all: preprocess all supported dataset


# Run training
The entrypoint for training are `pretrain_gr_retrieval.py` or `pretrain_gr_ranking.py`. We use gin-config to specify the model structure, training arguments, hyper-params etc.
To run retrieval task with `MovieLens 20m` dataset:
```python
torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  pretrain_gr_retrieval.py --gin-config-file movielen_retrieval.gin
```

To run ranking task with `MovieLens 20m` dataset:
```python
torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  pretrain_gr_ranking.py --gin-config-file movielen_ranking.gin
```
