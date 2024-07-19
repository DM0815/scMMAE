# scMMAE
A cross-attention network based on masked autoencoder called single-cell multimodal masked autoencoder
![Image text](https://github.com/DM0815/scMMAE/blob/main/framework.jpg)

## Prerequisite
* python 3.11.9
* timm 1.0.7
* pytorch 2.3.1
* cudnn 8.9.2.26
* scanpy 1.10.2
* anndata 0.10.8
* scikit-learn 1.5.1
The above packages are the main packages used for the experiment, most 2.0+ torch environments can run the experiment dircetly, just in case we have provided a requirement file for all packages.  

## Getting started
If you want to use your own datasets in scMMAE, you should change six parameters: <br>
config.RNA_tokens = config.RNA_component \* config.emb_RNA, RNA_tokens represents the gene number you used (I used 4000 highly variable genes), and config.emb_RNA needs to be divisible by head numbers; <br>
config.ADT_tokens = config.ADT_component \* config.emb_ADT, ADT_tokens represents the protein number you used (I used all proteins), and config.emb_ADT needs to be divisible by head numbers.
## Important
The input data is two matrix (RNA: cell_numbers\*1\*gene_numbers, PROTEIN:cell_numbers\*1\*protein_numbers). In addition, input data should be normalized before running the model.
## Example
Use Anaconda to create a Python virtual environment. Here, we will create a Python 3.11 environment named scMMAE
```cmd
conda create -n scMMAE python=3.11.9
```
Install  packages
```cmd
pip install -r requirements.txt
```
Then you can run './scMMAE/code/tutorial.ipynb' to reproduce the results for IFNB scRNA-seq dataset. Of note, due to the large size of the dataset, we have uploaded a rar archive inside the dataset folder, which you will need to extract to the current directory.
You can run './scMMAE/code/stage1.py', './scMMAE/code/stage2.py', and './scMMAE/code/stage3.py' directly as long as you unzip the dataset in the './scMMAE/dataset/CITE-seq/*.rar' ,and './scMMAE/dataset/RNA-seq/*.rar'.

## Checkpoint
If you need checkpoints for pre-training models and fine-tuning models for the dataset in the experiment, please contact [dmeng@gbu.edu.cn](mailto:dmeng@gbu.edu.cn)

