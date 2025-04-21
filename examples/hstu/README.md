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
The **distributed-recommender** provides preprocessor scripts to assist in downloading raw data if it is not already present. It processes the raw data into NPZ files and splits the data into training and testing sets.
```
mkdir -p ./tmp && python3 -m distributed_recommender.data.preprocessor --dataset_name ml-1m
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
torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  examples/hstu/pretrain_gr_retrieval.py --gin-config-file examples/configs/movielen_retrieval.gin
```

To run ranking task with `MovieLens 20m` dataset:
```python
torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  examples/hstu/pretrain_gr_ranking.py --gin-config-file examples/configs/movielen_ranking.gin
```


INFO:main:Processed file saved to data/ml-1m/processed_seqs.csv
INFO:main:num users: 6040
INFO:main:feature_name      min    max
--------------  -----  -----
sex                 1      2
age_group           1      7
occupation          1     21
zip_code            1   3439
INFO:main:feature_name      min    max    min_seqlen    max_seqlen    average_seqlen
--------------  -----  -----  ------------  ------------  ----------------
movie_id            1   3952            20          2314               165
rating              2      10            20          2314               165

INFO:main:Processed file saved to data/ml-20m/processed_seqs.csv
INFO:main:num users: 138493
INFO:main:feature_name    min    max
--------------  -----  -----
INFO:main:feature_name      min     max    min_seqlen    max_seqlen    average_seqlen
--------------  -----  ------  ------------  ------------  ----------------
movie_id            1  131262            20          9254               144
rating              0       10            20          9254               144

INFO:main:Processed file saved to data/KuaiRand-Pure/data/processed_seqs.csv
INFO:main:num users: 25010
INFO:main:feature_name             min    max
---------------------  -----  -----
user_active_degree         1      9
follow_user_num_range      1      8
fans_user_num_range        1      9
friend_user_num_range      1      7
register_days_range        1      7
INFO:main:feature_name      min    max    min_seqlen    max_seqlen    average_seqlen
--------------  -----  -----  ------------  ------------  ----------------
video_id            0   7582             2           910                56
action_weights      0    225             2           910                56


INFO:main:Processed file saved to data/KuaiRand-1K/data/processed_seqs.csv
INFO:main:num users: 983
INFO:main:feature_name             min    max
---------------------  -----  -----
user_active_degree         1      7
follow_user_num_range      1      8
fans_user_num_range        1      8
friend_user_num_range      1      7
register_days_range        1      7
INFO:main:feature_name      min      max    min_seqlen    max_seqlen    average_seqlen
--------------  -----  -------  ------------  ------------  ----------------
video_id            1  4371899           198        127557             11876
action_weights      0      232           198        127557             11876


INFO:main:Processed file saved to data/KuaiRand-27K/data/processed_seqs.csv
INFO:main:num users: 26811
INFO:main:feature_name             min    max
---------------------  -----  -----
user_active_degree         1      9
follow_user_num_range      1      8
fans_user_num_range        1      9
friend_user_num_range      1      7
register_days_range        1      7
INFO:main:feature_name      min       max    min_seqlen    max_seqlen    average_seqlen
--------------  -----  --------  ------------  ------------  ----------------
video_id            0  32038724           101        228000             11958
action_weights      0       245           101        228000             11958
