# Train BERT

This repository provides [BERT](https://arxiv.org/abs/1810.04805) Japanese models trained as Hugging Face [transformers](https://github.com/huggingface/transformers) models.

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
<summary>bert-base-ja v1</summary>

### base-v1

#### Prepare working directory

Create a working directory and clone a repository.

[TODO] Need to write a specific version to clone.

```sh
$ git clone https://github.com/colorfulscoop/convmodel
$ cd convmodel
$ git checkout 163078
$ cd ..
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
(container)$ python3 prepare_train_data.py --filename data/train.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw False >data/train.jsonl
(container)$ python3 prepare_train_data.py --filename data/valid.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw True >data/valid.jsonl
(container)$ python3 prepare_train_data.py --filename data/test.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw True >data/test.jsonl

```

Generate a default PyTorch Lightning config file and copy it.

```
(container)$ python3 trainer.py --print_config >default_config.yaml
(container)$ cp default_config.yaml config.yaml
```

Modify config.yaml

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
>   accumulate_grad_batches: 32
>   max_epochs: 13
31c31
<   val_check_interval: 1.0
---
>   val_check_interval: 320000
36c36
<   precision: 32
---
>   precision: 16
44c44
<   deterministic: false
---
>   deterministic: true
57a58,67
>   callbacks:
>    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
>    - class_path: pytorch_lightning.callbacks.GPUStatsMonitor
>    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
>      init_args:
>        monitor: val_loss
>        mode: min
>        every_n_train_steps: 10000
>        save_top_k: 3
>
59,62c69,72
<   tokenizer_model: null
<   train_file: null
<   valid_file: null
<   test_file: null
---
>   tokenizer_model: output/model
>   train_file: data/train.jsonl
>   valid_file: data/valid.jsonl
>   test_file: data/test.jsonl
69c79
<   batch_size: 2
---
>   batch_size: 8
72c82
<   shuffle_buffer_size: 1000
---
>   shuffle_buffer_size: 10000
74c84
<   num_warmup_steps: 0
---
>   num_warmup_steps: 10000
```

Start training.

```sh
(container)$ python3 trainer.py --config config.yaml
```

To check progress, open another terminal and run Tensorboard.

```sh
docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
```

#### Export Hugging Face transofmrers model

After completing your training, convert PyTorch Lightning model to Hugging Face transformers model.

```sh
(container)$ python export_model.py --ckpt_path lightning_logs/version_5/checkpoints/epoch\=2-step\=189999.ckpt --config lightning_logs/version_5/config.yaml --output_dir model
```

#### Convert tokenizer model

The behavior of AlbertTokenizer and AlbertTokenizerFast is different. Our expectation is AlbertTokenizer; however, the default behavior of AutoTokenizer is using AlbertTokenizerFast.
When using AutoTokenizer, `use_fast=False` can solve this inconsistency behavior.
However, when using Hugging Face Inference API, there are no ways to pass `use_fast` via config.json or model card.

We realized this after completing training.
We decided to use [DebertaV2Tokenizer](https://huggingface.co/transformers/model_doc/deberta_v2.html?highlight=sentencepiece#transformers.DebertaV2Tokenizer)
because it does not provide any Fast tokenizers and its default behavior is what we expected.
