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
$ git checkout 24820b
$ cd ..
```

Copy training data to working directory.

```sh
$ cp -r data/data/jawiki/20210620/data convmodel/trainer/bert/
```

```sh
$ cd convmodel/
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
(container)$ apt update && apt install -y python3 python3-pip git
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

#### Train BERT model

Prepare JSON Lines format training dataset.

```sh
(container)$ python3 prepare_train_data.py --filename data/train.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw False >data/train.jsonl
(container)$ python3 prepare_train_data.py --filename data/valid.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw True >data/valid.jsonl
(container)$ python3 prepare_train_data.py --filename data/test.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw True >data/test.jsonl
```

Generate a default PyTorch Lightning config file and copy it.

```sh
(container)$ python3 trainer.py --print_config >default_config.yaml
(container)$ cp default_config.yaml config.yaml
```

Modify config.yaml to realize our training configuration

* Model size is the same as BERT base (hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12, max_position_embeddings: 512)
* gradient update is every 256 samples (batch size: 8, accumulate_grad_batches: 32)
* gradient clip norm is 1.0 (gradient_clip_val: 1.0)
* Learning rate starts from 0 and linearly increased to 0.0001 with 10,000 steps (lr: 0.0001, num_warmup_steps: 10000)

This training set contains around 20M samples. Because 80k * 256 ~ 20M, 1 epochs has around 80k steps.
If we realize 1M steps mentioned in BERT paper, at most 1M steps / 80k steps = 12.5 <= 13 epochs is required. Therefore we set the largest epoch size to 13 (max_epoch :13)

Because val_check_interval is counted per batch, if we want to check validation every 10,000 steps, then the value should be 32 * 10,000 = 320,000.


Actual config is as follows.

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

Finally the validation loss has reached to 2.854

#### Run test

After completing training, I ran the test.

```sh
(container)$ python3 test.py --config config.yaml --ckpt_path lightning_logs/version_5/checkpoints/epoch\=2-step\=209999.ckpt
Testing: 62500it [1:05:33, 15.89it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_loss': 2.798677921295166}
--------------------------------------------------------------------------------
```

**Note:**

When running the following code, some error happened in the callback.

```sh
(container)$ python3 test.py --config config.yaml --ckpt_path lightning_logs/version_5/checkpoints/epoch\=2-step\=209999.ckpt
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
Using native 16bit precision.
Traceback (most recent call last):
  File "test.py", line 26, in <module>
    fire.Fire(main)
  File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 141, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 466, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/usr/local/lib/python3.8/dist-packages/fire/core.py", line 681, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "test.py", line 19, in main
    trainer = pl.Trainer(**config_yaml["trainer"])
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/env_vars_connector.py", line 40, in insert_env_defaults
    return fn(self, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/trainer.py", line 361, in __init__
    self.on_init_start()
  File "/usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/callback_hook.py", line 60, in on_init_start
    callback.on_init_start(self)
AttributeError: 'dict' object has no attribute 'on_init_start'
```

Therefore, I commented out all callbacks in config file when running this test. After testing, I removed those comment out.

```sh
  #callbacks:
  # - class_path: pytorch_lightning.callbacks.LearningRateMonitor
  # - class_path: pytorch_lightning.callbacks.GPUStatsMonitor
  # - class_path: pytorch_lightning.callbacks.ModelCheckpoint
  #   init_args:
  #     monitor: val_loss
  #     mode: min
  #     every_n_train_steps: 10000
  #     save_top_k: 3
```

#### Export Hugging Face transofmrers model

After completing your training, convert PyTorch Lightning model to Hugging Face transformers model.

```sh
(container)$ python3 export_model.py --ckpt_path lightning_logs/version_5/checkpoints/epoch\=2-step\=209999.ckpt --config lightning_logs/version_5/config.yaml --output_dir model
```

#### Convert tokenizer model

The behavior of AlbertTokenizer and AlbertTokenizerFast is different. Our expectation is AlbertTokenizer; however, the default behavior of AutoTokenizer is using AlbertTokenizerFast.
When using AutoTokenizer, `use_fast=False` can solve this inconsistency behavior.
However, when using Hugging Face Inference API, there are no ways to pass `use_fast` via config.json or model card.

We realized this after completing training.
We decided to use [DebertaV2Tokenizer](https://huggingface.co/transformers/model_doc/deberta_v2.html?highlight=sentencepiece#transformers.DebertaV2Tokenizer)
because it does not provide any Fast tokenizers and its default behavior is what we expected.


First, convert an AlbertTokenizer model to a DebertaV2Tokenzier

```py
(container)$ python3
>>> ot = transformers.AutoTokenizer.from_pretrained("output/model", use_fast=False)
>>> nt = transformers.DebertaV2Tokenizer("output/spm/sp.model", unk_token=ot.unk_token, sep_token=ot.sep_token, pad_token=ot.pad_token, cls_token=ot.cls_token, mask_token=ot.mask_token)
>>> nt.save_pretrained("model")
('model/tokenizer_config.json', 'model/special_tokens_map.json', 'model/spm.model', 'model/added_tokens.json')
:w
```

Then modify a model/config.json file to update the tokenizer name.

```json
# Modify the tokenizer_class in model/config.json file
  "tokenizer_class": "DebertaV2Tokenizer",
```

Finally, remove unnecessaly file

```sh
(container)$ rm model/spiece.model
```

#### Example usage

```py
(container)$ python3
>>> import transformers
>>> pl = transformers.pipeline("fill-mask", "model")
>>> pl("専門として[MASK]を専攻しています")
[{'sequence': '専門として工学を専攻しています', 'score': 0.03630176931619644, 'token': 3988, 'token_str': '工学'}, {'sequence': '専門として政治学を専攻しています', 'score': 0.03547220677137375, 'token': 22307, 'token_str': '政治学'}, {'sequence': '専門として教育を専攻しています', 'score': 0.03162326663732529, 'token': 414, 'token_str': '教育'}, {'sequence': '専門として経済学を専攻しています', 'score': 0.026036914438009262, 'token': 6814, 'token_str': '経済学'}, {'sequence': '専門として法学を専攻しています', 'score': 0.02561848610639572, 'token': 10810, 'token_str': '法学'}]
```

#### Upload to Hugging Face Model Hub

Finally, upload the trained model to HuggingFace's model hub. Following the official document, the following process is executed.

First, create a repository named "bert-base-ja" from HuggingFace's website.

Then, prepare git lfs. In a MacOS environment, git lfs can be installed as follows.

```sh
$ brew install git-lfs
$ git lfs install
Updated git hooks.
Git LFS initialized.
```

Then clone repository to local

```sh
$ git clone https://huggingface.co/colorfulscoop/bert-base-ja release/bert-base-ja
```

Copy model

```sh
$ cp convmodel/trainer/bert/model/* release/bert-base-ja/
```

Then prepare Tensorflow model

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ pip install torch==1.8.0 transformers==4.8.2 sentencepiece==0.1.95 tensorflow==2.5.0
(container)$ python3
>>> import transformers
>>> tf_model = transformers.TFBertForPreTraining.from_pretrained("release/bert-base-ja", from_pt=True)
>>> tf_model.save_pretrained("release/bert-base-ja")
```

Finally, copy model card and changelog files

```sh
$ cp model_card.md release/bert-base-ja/README.md
$ cp CHANGELOG.md release/bert-base-ja
```

To enable an inference API on Model Hub, modify an architecture in a release/bert-base-ja/config.json file

```sh
 {
   "_name_or_path": "release/bert-base-ja",
   "architectures": [
-    "BertForPreTraining"
+    "BertForMaskedLM"
   ],
   "attention_probs_dropout_prob": 0.1,
   "bos_token_id": 2,
```

Finally commit it and push to Model Hub.

```sh
$ cd release/bert-base-ja/
$ git add .
$ git commit -m "Add models and model card"
$ git push origin
```
