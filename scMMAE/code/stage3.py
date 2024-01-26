import torch
import timm
import numpy as np
import os
import math
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torchvision.transforms import ToTensor, Compose, Normalize
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import anndata as ad
from sklearn.model_selection import train_test_split
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

class Config(object):
    """para"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
        self.dropout = 0.1                                    
        self.num_classes = 10                                             
        self.batch_size = 128                                       
        self.lr = 5e-4                      
        self.encoder_layer = 6
        self.encoder_head = 2
        self.decoder_layer = 4
        self.decoder_head = 2
        self.mask_ratio = 0.15        
        self.omics_tokens = 4000 ##omics cells
        self.omic_genecomponent = 400 ##channel        
        self.emb_gene = 10
        
        self.mask_ratio1 = 0.15        
        self.omics_tokens1 = 14 ##omics proteins
        self.omic_genecomponent1 =14 ##channel
        self.emb_gene1 = 1 
        
        self.emb_dim = 128       
        self.total_epoch = 500
        self.warmup_epoch = 10
config = Config()        

class Omics_Class_encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_gene=10,omic_genecomponent=400,omics_tokens=4000, encoder_head=4,encoder_layer=6) -> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = omics_tokens, out_features = omics_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_gene, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, omic_genecomponent ,emb_dim)))#
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, encoder_head,attn_drop=0.1 ,proj_drop=0.1) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        #self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, patches):
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0),config.omic_genecomponent,config.emb_gene)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding
        patches = rearrange(patches, 'b c s -> c b s')
        
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        #print(features.shape)
        
        return features
##cross attention
class CrossAttention(torch.nn.Module):
    def __init__(self, emb_dim=64)-> None:
        super(CrossAttention, self).__init__()
        self.query_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.key_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.value_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.scale_factor = 1.0 / (emb_dim ** 0.5)
        # Layer Normalization
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
    def forward(self, query, key, value):
        query_proj = self.query_linear(query)  # 
        key_proj = self.key_linear(key)  # 
        value_proj = self.value_linear(value)  # 

        attention_scores = torch.matmul(query_proj, key_proj.transpose(-2, -1))#T)  # 
        attention_scores = attention_scores * self.scale_factor # 
        attention_weights = torch.softmax(attention_scores, dim=-1)  # 

        attended_values = torch.matmul(attention_weights, value_proj)  # 
        attended_values = self.layer_norm(attended_values)
        return attended_values
    
class Omics_Pred(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.omics1_encoder = Omics_Class_encoder(config.emb_dim,config.emb_gene,config.omic_genecomponent,config.omics_tokens,config.encoder_head,config.encoder_layer)

        
        self.omics2_in_omics1_Att = CrossAttention(config.emb_dim)
        self.fc = torch.nn.Linear(config.emb_dim, config.emb_dim)
        self.head = torch.nn.Linear(config.emb_dim, config.num_classes)
    def forward(self, patches1):
      
        patches1 = self.omics1_encoder(patches1)

        omics_feature1 = rearrange(patches1.clone(), 't b c -> b t c')
        omics_feature1_cls = omics_feature1[:,0,:].unsqueeze(1).clone() #cls
 
        omics_feature2_cls = self.fc(omics_feature1_cls)
        
        x1 = self.omics2_in_omics1_Att(omics_feature2_cls, omics_feature1_cls, omics_feature1_cls)##只用cls_token交互

        x1 += omics_feature2_cls

        final_results = x1.squeeze(1)##
        
        logits = self.head(x1)
        logits = logits.squeeze(1)
        # print(logits.shape)
        # print(final_results.shape)
        return logits,final_results