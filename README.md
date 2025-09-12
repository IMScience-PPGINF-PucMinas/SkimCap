SkimCap: A Transformer-Based Video Captioning Method with Adaptive Attention and Hierarchical Skimming Features
=====
PyTorch code for our PRL 2025 paper "SkimCap: A Transformer-Based Video Captioning Method with Adaptive Attention and Hierarchical Skimming Features" Enhanced
by [Leonardo Vilela Cardoso](http://lattes.cnpq.br/6741312586742178), Bernardo Palmer, [Silvio Jamil F. Guimarães](http://lattes.cnpq.br/8522089151904453) and 
[Zenilton K. G. Patrocínio Jr](http://lattes.cnpq.br/8895634496108399), 

We present SkimCap, a transformer-based video captioning framework that integrates a memory-augmented architecture with adaptive attention and a novel feature selection strategy grounded in hierarchical video skimming. Unlike traditional approaches that rely on uniformly sampled frames or pre-defined temporal segments, SkimCap performs unsupervised hierarchical clustering to identify and extract semantically salient video shots. These condensed representations provide a compact yet information-rich input to the captioning model, enabling more accurate and contextually grounded sentence generation. The memory module enhances long-range dependency modeling, while adaptive attention improves temporal alignment between visual cues and generated tokens. We evaluate SkimCap on ActivityNet, achieving CIDEr-D of 25.44, a BLEU-4 (B@4) of 10.77, and a lower Repetition-4 (R@4) score of 5.84, representing consistent caption quality and relevance improvements. An ablation study confirms the effectiveness of hierarchical skimming as a feature selection mechanism, highlighting its contribution to overall performance. SkimCap sets a new direction for incorporating structured visual summarization into end-to-end captioning systems.

## Main dependencies
Developed, checked and verified on an `Ubuntu 22.04` PC with a `GTX 1080 SUPER` GPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.9 | 2.4.1 | 11.0 | 8005 | 2.4.1 | 2.3.0 | 1.20.2 | 2.10.0

## Data
<div align="justify">
Original videos and annotations for each dataset are also available in the dataset providers' webpages:
<br />
- <a href="http://activity-net.org/"><img src="https://img.shields.io/badge/Dataset-ActivityNet-green"/></a> <a href="http://youcook2.eecs.umich.edu/"><img src="https://img.shields.io/badge/Dataset-YouCookII-blue"/></a>
</div>

## Getting started
### Prerequisites
0. Clone this repository
```
# no need to add --recursive as all dependencies are copied into this repo.
git clone https://github.com/IMScience-PPGINF-PucMinas/Adaptive-Transformer.git
cd Adaptive-Transformer
```

1. Prepare feature files

Download features from Google Drive: [rt_anet_feat.tar.gz (39GB)](https://drive.google.com/file/d/1mbTmMOFWcO30PIcuSpYiZ1rqoy5ltE3A/view?usp=sharing) 
and [rt_yc2_feat.tar.gz (12GB)](https://drive.google.com/file/d/1mj76DwNexFCYovUt8BREeHccQn_z_By9/view?usp=sharing).
These features are repacked from features provided by [densecap](https://github.com/salesforce/densecap#annotation-and-feature). 
```
mkdir video_feature && cd video_feature
tar -xf path/to/rt_anet_feat.tar.gz 
tar -xf path/to/rt_yc2_feat.tar.gz 
```

2. Install dependencies
- Python 3.9
- PyTorch 2.4.1
- nltk
- easydict
- tqdm
- tensorboardX

3. Add project root to `PYTHONPATH`
```
source setup.sh
```
Note that you need to do this each time you start a new session.


### Training and Inference
We give examples on how to perform training and inference with Adaptive-Transformer.

0. Build Vocabulary
```
bash scripts/build_vocab.sh
```
`DATASET_NAME` can be `anet` for ActivityNet Captions or `yc2` for YouCookII.


1. Adaptive-Transformer training

The general training command is:
```
bash scripts/train.sh
```

To train our Adaptive-Transformer model on ActivityNet Captions:
```
bash scripts/train.sh anet
```

Training log and model will be saved at `results/anet_re_*`.  
Once you have a trained model, you can follow the instructions below to generate captions. 


2. Generate captions 
```
bash scripts/translate_greedy.sh anet_re_* val
```
Replace `anet_re_*` with your own model directory name. 
The generated captions are saved at `results/anet_re_*/greedy_pred_val.json`


3. Evaluate generated captions
```
bash scripts/eval.sh anet val results/anet_re_*/greedy_pred_val.json
```
The results should be comparable with the results we present at Table 2 of the paper. 
E.g., B@4 10.77; C 25.44 R@4 5.84.

## Citations
If you find this code useful for your research, consider citing one of our papers:
```
@article{cardoso2023hierarchical,
  title={Hierarchical time-aware summarization with an adaptive transformer for video captioning},
  author={Cardoso, Leonardo Vilela and Guimar{\~a}es, Silvio Jamil Ferzoli and do Patroc{\'\i}nio J{\'u}nior, Zenilton Kleber Gon{\c{c}}alves},
  journal={International Journal of Semantic Computing},
  volume={17},
  number={04},
  pages={569--592},
  year={2023},
  publisher={World Scientific}
}
@inproceedings{cardoso2022exploring,
  title={Exploring adaptive attention in memory transformer applied to coherent video paragraph captioning},
  author={Cardoso, Leonardo Vilela and Guimaraes, Silvio Jamil F and Patrocinio, Zenilton KG},
  booktitle={2022 IEEE Eighth International Conference on Multimedia Big Data (BigMM)},
  pages={37--44},
  year={2022},
  organization={IEEE}
}

@inproceedings{cardoso2021enhanced,
  title={Enhanced-Memory Transformer for Coherent Paragraph Video Captioning},
  author={Cardoso, Leonardo Vilela and Guimaraes, Silvio Jamil F and Patroc{\'\i}nio, Zenilton KG},
  booktitle={2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI)},
  pages={836--840},
  year={2021},
  organization={IEEE}
}
```

## Others
This code used resources from the following projects: 
[emt](https://github.com/IMScience-PPGINF-PucMinas/EMT),
[mart](https://github.com/jayleicn/recurrent-transformer),
[transformers](https://github.com/huggingface/transformers), 
[transformer-xl](https://github.com/kimiyoung/transformer-xl), 
[densecap](https://github.com/salesforce/densecap),
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## Contact
Leonardo Vilela Cardoso with this e-mail: lvcardoso@sga.pucminas.br

