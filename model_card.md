---
language: ja
datasets: wikipedia
pipeline_tag: fill-mask
widget:
- text: "得意な科目は[MASK]です。"
license: cc-by-sa-3.0
---

# BERT base Japanese model

This repository contains a BERT base model trained on Japanese Wikipedia dataset.

## Training data

[Japanese Wikipedia](https://ja.wikipedia.org/wiki/Wikipedia:データベースダウンロード) dataset as of June 20, 2021 which is released under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) is used for training.
The dataset is splitted into three subsets - train, valid and test. Both tokenizer and model are trained with the train split.

## Model description

The model architecture is the same as BERT base model (hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12, max_position_embeddings: 512) except for a vocabulary size.
The vocabulary size is set to 32,000 instead of an original size of 30,522.

For the model, `transformers.BertForPreTraining` is used.

## Tokenizer description

[SentencePiece](https://github.com/google/sentencepiece) tokenizer is used as a tokenizer for this model.

While training, the tokenizer model was trained with 1,000,000 samples which were extracted from the train split.
The vocabulary size is set to 32,000. A `add_dummy_prefix` option is set to `True` because words are not separated by whitespaces in Japanese.

After training, the model is imported to `transformers.DebertaV2Tokenizer` because it supports SentencePiece models and its behavior is consistent when `use_fast` option is set to `True` or `False`.

**Note:**
The meaning of "consistent" here is as follows.
For example, AlbertTokenizer provides AlbertTokenizer and AlbertTokenizerFast. Fast model is used as default. However, the tokenization behavior between them is different and a behavior this mdoel expects is the verions of not fast.
Although `use_fast=False` option passing to AutoTokenier or pipeline solves this problem to force to use not fast version of the tokenizer, this option cannot be passed to config.json or model card.
Therefore unexpected behavior happens when using Inference API. To avoid this kind of problems, `transformers.DebertaV2Tokenizer` is used in this model.

## Training

Training details are as follows.

* gradient update is every 256 samples (batch size: 8, accumulate_grad_batches: 32)
* gradient clip norm is 1.0
* Learning rate starts from 0 and linearly increased to 0.0001 in the first 10,000 steps
* The training set contains around 20M samples. Because 80k * 256 ~ 20M, 1 epochs has around 80k steps.

Trainind was conducted on Ubuntu 18.04.5 LTS with one RTX 2080 Ti.

The training continued until validation loss got worse. Totally the number of training steps were around 214k.
The test set loss was 2.80 .

Training code is available in [a GitHub repository](https://github.com/colorfulscoop/bert-ja).

## Usage

First, install dependecies.

```sh
$ pip install torch==1.8.0 transformers==4.8.2 sentencepiece==0.1.95
```

Then use `transformers.pipeline` to try mask fill task.

```sh
>>> import transformers
>>> pipeline = transformers.pipeline("fill-mask", "colorfulscoop/bert-base-ja", revision="v1.0")
>>> pipeline("専門として[MASK]を専攻しています")
[{'sequence': '専門として工学を専攻しています', 'score': 0.03630176931619644, 'token': 3988, 'token_str': '工学'}, {'sequence': '専門として政治学を専攻しています', 'score': 0.03547220677137375, 'token': 22307, 'token_str': '政治学'}, {'sequence': '専門として教育を専攻しています', 'score': 0.03162326663732529, 'token': 414, 'token_str': '教育'}, {'sequence': '専門として経済学を専攻しています', 'score': 0.026036914438009262, 'token': 6814, 'token_str': '経済学'}, {'sequence': '専門として法学を専攻しています', 'score': 0.02561848610639572, 'token': 10810, 'token_str': '法学'}]
```

Note: specifying a `revision` option is recommended to keep reproducibility when downloading a model via `transformers.pipeline` or `transformers.AutoModel.from_pretrained` .

## License

Copyright (c) 2021 Colorful Scoop

All the models included in this repository are licensed under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

**Disclaimer:** The model potentially has possibility that it generates similar texts in the training data, texts not to be true, or biased texts. Use of the model is at your sole risk. Colorful Scoop makes no warranty or guarantee of any outputs from the model. Colorful Scoop is not liable for any trouble, loss, or damage arising from the model output.

---

This model utilizes the following data as training data

* **Name:** ウィキペディア (Wikipedia): フリー百科事典
* **Credit:** https://ja.wikipedia.org/
* **License:** [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
* **Link:** https://ja.wikipedia.org/
