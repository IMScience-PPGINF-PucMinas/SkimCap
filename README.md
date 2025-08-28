SkimCap: A Transformer-Based Video Captioning Method with Adaptive Attention and Hierarchical Skimming Features
=====
PyTorch code for our PRL 2025 paper "SkimCap: A Transformer-Based Video Captioning Method with Adaptive Attention and Hierarchical Skimming Features" Enhanced
by [Leonardo Vilela Cardoso](http://lattes.cnpq.br/6741312586742178), Bernardo Palmer, [Silvio Jamil F. Guimarães](http://lattes.cnpq.br/8522089151904453) and 
[Zenilton K. G. Patrocínio Jr](http://lattes.cnpq.br/8895634496108399), 

We present SkimCap, a transformer-based video captioning framework that integrates a memory-augmented architecture with adaptive attention and a novel feature selection strategy grounded in hierarchical video skimming. Unlike traditional approaches that rely on uniformly sampled frames or pre-defined temporal segments, SkimCap performs unsupervised hierarchical clustering to identify and extract semantically salient video shots. These condensed representations provide a compact yet information-rich input to the captioning model, enabling more accurate and contextually grounded sentence generation. The memory module enhances long-range dependency modeling, while adaptive attention improves temporal alignment between visual cues and generated tokens. We evaluate SkimCap on ActivityNet, achieving CIDEr-D of 25.44, a BLEU-4 (B@4) of 10.77, and a lower Repetition-4 (R@4) score of 5.84, representing consistent caption quality and relevance improvements. An ablation study confirms the effectiveness of hierarchical skimming as a feature selection mechanism, highlighting its contribution to overall performance. SkimCap sets a new direction for incorporating structured visual summarization into end-to-end captioning systems.

## Getting started
### Prerequisites
0. Clone this repository
```
# no need to add --recursive as all dependencies are copied into this repo.
git clone "your_github_url"
cd "your_method"
```

1. Prepare feature files

Some details of where the dataset is found or how the dataset will be generated

2. Install dependencies (for example)
- Python 2.7
- PyTorch 1.1.0
- nltk
- easydict
- tqdm
- tensorboardX

### Training and Inference
We give examples on how to perform training and inference with "your_method".


1. "your_method" training

The general training command is:
```
bash scripts/train.sh
```

To train our "your_method" model on "dataset":
```
bash scripts/train.sh anet
```

2. "your_github_url" test

The general test command is:
```
bash scripts/test.sh
```

The results should be comparable with the results we present at Table "number" of the paper. 
E.g., "your_results".

## Citations
If you find this code useful for your research, consider cite our paper:
```
"your_bibtex_code"
```

## Others
This code used resources from the following projects: 
[method_name]("url_from_the_code").

## Contact
"your_name" with this e-mail: "your_email"

