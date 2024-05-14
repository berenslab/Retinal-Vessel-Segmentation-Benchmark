# Retinal Vessel Segmenation Model Benchmark
Code associated with paper: *Benchmarking **Retinal Blood Vessel Segmentation Models for** Cross-Dataset and Cross-Disease Generalization*.

This is a benchmark with the aim to investigate the performance of various deep learning model for retinal vessel segmentation 
It implements 5 backbone models on 3 datasets and compare the performances for different loss functions, image qualities and pathological conditions as well as the cross dataset generalization capabilities of the models.

We are actively expanding the benchmark to include more models, UQ methods, and datasets.


## Backbones

| Backbone Models      | Paper | Official Repo |
| ----------- | ----------- | ----------- |

| UNet |[link](https://arxiv.org/abs/1505.04597) | | 
| FR-UNet | [link](https://ieeexplore.ieee.org/abstract/document/9815506) | [link](https://github.com/lseventeen/FR-UNet)|
| MA-Net| [link](https://ieeexplore.ieee.org/document/9201310) |  |
| SA-UNet | [link](https://arxiv.org/abs/2004.03696) | [link](https://github.com/clguo/SA-UNet) |
| W-Net | [link](https://doi.org/10.1038/s41598-022-09675-y) | [link](https://github.com/agaldran/lwnet) |


## Data

* FIVES
* CHASEDB1
* DRIVE

## Environment Setup

Our code is developed with `Python 3.10`, The required packages are listed in `requirements.txt` and the environment can be created by running:
```bash
pip install -r requirements.txt
```
in your created environment.

### Specify Arguments using `.yaml` Files

Specify the arguments in any of the files in the `config` folder with the respective model and loss function

### Training

To run the code for training the baseline models:

```bash
# 
python train_baseline.py --config config/fives.yaml
```
This example is for fives dataset with the models specified inside.

To train the `3vs1` subgroup setting. It requires an additional argument with the `train_disease.py` file models on a particular dis:

```bash
# 
python train_disease.py --config config/fives.yaml --disease D
```
This example is for fives dataset with the models specified inside and to train on other group of disease except D `Diabetic-Retinppathy`


