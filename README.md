# scMMAE
A cross-attention network based on masked autoencoder called single-cell multimodal masked autoencoder
![Image text](https://github.com/DM0815/scMMAE/blob/main/workflow.jpg)

## Prerequisite
* python 3.10.13
* pytorch 2.1.0
* scanpy
* anndata
* sklearn
## Getting started
If you want to use your own datasets in scMMAE, you should change six parameters: <br>
config.omics_tokens = config.omics_genecomponent \* config.emb_gene; omics_tokens represents the gene number you used (I used 4000 highly variable genes), and config.emb_gene needs to be divisible by head numbers; <br>
config.omics_tokens1 = config.omics_genecomponent1 \* config.emb_gene1 omics_tokens represents the protein number you used (I used all proteins), and config.emb_gene1 needs to be divisible by head numbers.
## Important
The input data is two matrix (RNA: cell_numbers\*1\*gene_numbers, PROTEIN:cell_numbers\*1\*protein_numbers). In addition, input data should be normalized before running the model.
