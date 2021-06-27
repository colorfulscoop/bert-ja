# Train BERT

## Prepare data

```sh
$ mkdir data
$ cd data/
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ apt update && apt install -y wget git
```

Check the latest date in the list from https://dumps.wikimedia.org/jawiki/ .

```sh
(container)$ bash get_jawiki.sh 20210620
```

```sh
(container)$ exit
$ cd ..
```

## Training details

<details>
<summary>BERT-base-ja v1</summary>

### base-v1

#### Prepare working directory

Create a working directory and clone a repository.

[TODO] Need to write a specific version to clone.

```sh
$ mkdir train/base-v1
$ cd train/base-v1
$ git clone https://github.com/colorfulscoop/convmodel
```

Copy training data to working directory.

```sh
$ cp -r ../../data/data/jawiki/20210620/data train/base-v1/convmodel/trainer/bert
```

```sh
$ cd convmodel/
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
(container)$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
(container)$ pip install .
(container)$ cd trainer/bert/
(container)$ pip3 install -r requirements.txt
```

#### Train tokenizer

```sh
(container)$ python3 train_tokenizer.py --train_file data/train.txt --input_sentence_size 1000000
```

The raw trained model is saved under output/spm, while transformers model is saved under output/model.

#### Train BERT

Prepare JSON Lines format training dataset.

```sh
(container)$ python3 prepare_train_data.py --filename data/train.txt  --buffer_size 10000 >data/train.jsonl
(container)$ python3 prepare_train_data.py --filename data/valid.txt  --buffer_size 10000 >data/valid.jsonl
(container)$ python3 prepare_train_data.py --filename data/test.txt  --buffer_size 10000 >data/test.jsonl
```

Set up config file.
First, generate a default config file.

```
(container)$ python3 trainer.py --print_config >default_config.yaml
(container)$ cp default_config.yaml config.yaml
```

Then modify config.yaml.

```sh
(container)$ diff default_config.yaml config.yaml
1c1
< seed_everything: null
---
> seed_everything: 1000
7c7
<   gradient_clip_val: 0.0
---
>   gradient_clip_val: 1.0
12c12
<   gpus: null
---
>   gpus: 1
21,22c21,22
<   accumulate_grad_batches: 1
<   max_epochs: null
---
>   accumulate_grad_batches: 16
>   max_epochs: 1
36c36
<   precision: 32
---
>   precision: 16
44c44
<   deterministic: false
---
>   deterministic: true
57a58,60
>   callbacks:
>     - class_path: pytorch_lightning.callbacks.LearningRateMonitor
>     - class_path: pytorch_lightning.callbacks.GPUStatsMonitor
59,62c62,65
<   tokenizer_model: null
<   train_file: null
<   valid_file: null
<   test_file: null
---
>   tokenizer_model: output/model
>   train_file: data/train.jsonl
>   valid_file: data/valid.jsonl
>   test_file: data/test.jsonl
74a78
>
```

Finally, start training


```sh
(container)$ python3 trainer.py --config config.yaml
```

#### Check log

Use tensorboard to check log.

```sh
$ docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
```
