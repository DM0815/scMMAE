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
        self.num_classes = 35                                             
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
##preprocessing
def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        #print(C)
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

## omics encoder
class Omics_Encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_gene=10,omic_genecomponent=400,omics_tokens=4000, encoder_head=4,encoder_layer=6,mask_ratio=0.1
                 )-> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = omics_tokens, out_features = omics_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_gene, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, omic_genecomponent ,emb_dim)))##
        self.shuffle = PatchShuffle(mask_ratio)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, encoder_head,attn_drop=0.1 ,proj_drop=0.1) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, patches):

        #print(patches.shape)
       
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0), config.omic_genecomponent, config.emb_gene)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding
        #print(self.pos_embedding.shape)
        #print(patches.shape)
        patches = rearrange(patches, 'b c s -> c b s')
        
        patches, forward_indexes, backward_indexes = self.shuffle(patches)    
        #print(patches.shape)
        #print(self.cls_token.expand(patches.shape[1],-1, -1).shape)
        patches = torch.cat([self.cls_token.expand(-1,patches.shape[1],-1), patches], dim=0)
        #print(patches.shape)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        #print(features.shape)
        return features, backward_indexes

## omics encoder11-omic2
class Omics_Encoder1(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_gene1=10,omic_genecomponent1=11,omics_tokens1=110, encoder_head=4,encoder_layer=6,mask_ratio1=0.1
                 )-> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = omics_tokens1, out_features = omics_tokens1))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_gene1, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, omic_genecomponent1 ,emb_dim)))##
        self.shuffle = PatchShuffle(mask_ratio1)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, encoder_head,attn_drop=0.1 ,proj_drop=0.1) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)        
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, patches):

        patches = self.tokens(patches)
        #print(patches.shape)
        patches = patches.view(patches.size(0), config.omic_genecomponent1, config.emb_gene1)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding
        #print(self.pos_embedding.shape)
        #print(patches.shape)
        patches = rearrange(patches, 'b c s -> c b s')

        #print(patches.shape)
        patches, forward_indexes, backward_indexes = self.shuffle(patches)    
        #print(patches.shape)
        patches = torch.cat([self.cls_token.expand(-1,patches.shape[1],-1), patches], dim=0)
        #print(patches.shape)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        #print(features.shape)
        return features, backward_indexes

##Omics decoder
class Omics_Decoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_gene=10,omic_genecomponent=400,omics_tokens=4000,decoder_head=4,decoder_layer=2
                 )-> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((omic_genecomponent + 1, 1, emb_dim)))##

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, decoder_head,attn_drop=0.1 ,proj_drop=0.1) for _ in range(decoder_layer)])        
        #elf.head = torch.nn.Linear(emb_dim, emb_dim)
        #self.patch2img = Rearrange('s b c  -> b c s')
        self.decoding = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_gene))
        self.init_weight()
        
    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        #print(T)
        #print(features.shape)
        #print(backward_indexes.shape)
        #print(torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes).shape)
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        #print(backward_indexes.shape)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        #print(features.shape)
        features = take_indexes(features, backward_indexes)
        #print(features.shape)
        # print(self.pos_embedding.shape)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        #print(features.shape)
        all_cls = []
        first_element = features[0]
        all_cls.append(first_element)
        
        patches = features[1:] # remove global feature
        #print(patches.shape)
        patches = self.decoding(patches)##64-10
        #print(patches.shape)
        #print(patches.shape)
        mask = torch.zeros_like(patches)
        # mask = rearrange(mask, 't b c -> b t c')
        #mask = self.decoding(mask)
        #print(mask.shape)
        mask[T-1:] = 1
        #print(mask.shape)
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        #print(mask.shape)
        patches = rearrange(patches, 't b c -> b t c')
        mask = rearrange(mask, 't b c -> b t c')
        #print(patches.shape)
        
        #mask = self.patch2img(mask)        
        patches = patches.reshape(patches.size(0),1,-1)
        mask = mask.reshape(mask.size(0),1,-1)
        
        #print (patches.shape)

        #print (patches.shape)
        return patches, mask, all_cls

##Omics decoder1
class Omics_Decoder1(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_gene1=10,omic_genecomponent1=11,omics_tokens1=110,decoder_head=4,decoder_layer=2
                 )-> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((omic_genecomponent1 + 1, 1, emb_dim)))##

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, decoder_head,attn_drop=0.1 ,proj_drop=0.1) for _ in range(decoder_layer)])        
        #elf.head = torch.nn.Linear(emb_dim, emb_dim)
        self.patch2img = Rearrange('s b c  -> b c s')
        self.decoding = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_gene1))
        self.init_weight()
        
    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        #print(T)
        # print(features.shape)
        #print(torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes).shape)
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        # print(backward_indexes.shape)
        # print(self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1).shape)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        #print(features.shape)
        # print(self.pos_embedding.shape)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        #print(features.shape)
        all_cls = []
        first_element = features[0]
        all_cls.append(first_element)
        
        patches = features[1:] # remove global feature
        patches = self.decoding(patches)##64-10
        
        
        mask = torch.zeros_like(patches)
        #print(mask.shape)
        #mask = self.decoding(mask)
        #print(mask.shape)
        mask[T-1:] = 1
        #print(mask.shape)
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        #print(mask.shape)
        patches = rearrange(patches, 't b c -> b t c')
        mask = rearrange(mask, 't b c -> b t c')
        
        #mask = self.patch2img(mask)
        patches = patches.reshape(patches.size(0),1,-1)
        mask = mask.reshape(mask.size(0),1,-1)
        
        return patches, mask, all_cls

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

class Omics_attention(torch.nn.Module):
    def __init__(self,config
                 ):
        super().__init__()

        self.omics1_encoder = Omics_Encoder(config.emb_dim,config.emb_gene,config.omic_genecomponent,config.omics_tokens,config.encoder_head,config.encoder_layer,config.mask_ratio)
        self.omics2_encoder = Omics_Encoder1(config.emb_dim,config.emb_gene1,config.omic_genecomponent1,config.omics_tokens1,config.encoder_head,config.encoder_layer,config.mask_ratio1)
       
 ##cross
        self.omics1_in_omics2_Att = CrossAttention(config.emb_dim)
        self.omics2_in_omics1_Att = CrossAttention(config.emb_dim)        
        
        self.omics1_decoder = Omics_Decoder(config.emb_dim,config.emb_gene,config.omic_genecomponent,config.omics_tokens,config.decoder_head,config.decoder_layer)
        self.omics2_decoder = Omics_Decoder1(config.emb_dim,config.emb_gene1,config.omic_genecomponent1,config.omics_tokens1,config.decoder_head,config.decoder_layer)

        
    def forward(self, patches1,patches2):
        #print(patches1.shape)
        #print(patches2.shape)

        omics_encoder_feature1, backward_indexes1 = self.omics1_encoder(patches1)
        omics_encoder_feature2, backward_indexes2 = self.omics2_encoder(patches2)
        
        #print(omics_feature1.shape)
        # print(omics_feature2.shape)
        #print(omics_feature2.shape)

        omics_feature1 = rearrange(omics_encoder_feature1.clone(), 't b c -> b t c')
        omics_feature1_cls = omics_feature1[:,0, :].unsqueeze(1).clone() #cls
        omics_feature2 = rearrange(omics_encoder_feature2.clone(), 't b c -> b t c')
        omics_feature2_cls = omics_feature2[:,0, :].unsqueeze(1).clone() #cls
#[0, :, :].unsqueeze(0)
        #print(omics_feature1_cls.shape)
        #print(omics_feature2_cls.shape)
        x2 = self.omics2_in_omics1_Att(omics_feature1_cls, omics_feature2_cls, omics_feature2_cls)## 
        x1 = self.omics1_in_omics2_Att(omics_feature2_cls, omics_feature1_cls, omics_feature1_cls)##

        #print(x2.shape)
        #print(x1.shape)
        
        omics_feature1[:,0, :] = x2.clone().squeeze(1)
        omics_feature2[:,0, :] = x1.clone().squeeze(1)
        omics_feature1 = rearrange(omics_feature1.clone(), 'b t c -> t b c')
        omics_feature2 = rearrange(omics_feature2.clone(), 'b t c -> t b c')

        omics_feature1 += omics_encoder_feature1
        omics_feature2 += omics_encoder_feature2
        #print(omics_feature1.shape)
        #print(omics_feature2.shape)
        
        omics_patches1, mask1, all_cls1 = self.omics1_decoder(omics_feature1,backward_indexes1) 
        omics_patches2, mask2, all_cls2 = self.omics2_decoder(omics_feature2,backward_indexes2)
        #final_result = omics_patches1 + omics_patches2
        #print(mask1)
        #print(omics_patches2.shape)        
        return omics_patches1,omics_patches2, mask1,mask2, all_cls1, all_cls2#,final_result

#############training
early_stopping_patience = 5  
best_val_loss = float('inf')  
no_improvement_count = 0
weight_a = 0.7
weight_b = 0.3 

model = Omics_attention(config).to(config.device)
#model = DataParallel(model)
if __name__ == '__main__':
    
        batch_size = config.batch_size
        load_batch_size = 128
        assert batch_size % load_batch_size == 0
        steps_per_update = batch_size // load_batch_size

        optim = torch.optim.AdamW(model.parameters(), lr=config.lr * config.batch_size / 256, betas=(0.9, 0.999), weight_decay=1e-4)
        lr_func = lambda epoch: min((epoch + 1) / (config.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

        best_val_acc = 0
        step_count = 0
        optim.zero_grad()
        for e in range(config.total_epoch):
            model.train()
            train_losses = []
            for tk in tqdm(iter(train_dataloader)):
                #print(train_dataset['mod1'].shape)
                step_count += 1

                
                omics_patches1,omics_patches2, mask1,mask2,logits_per_mod1,logits_per_mod2,final = model(tk['mod1'], tk['mod2'])
                #print(omics_patches1.shape)
                loss_a = torch.mean((omics_patches1 - tk['mod1']) ** 2 * mask1) / config.mask_ratio
                loss_b = torch.mean((omics_patches2 - tk['mod2']) ** 2 * mask2) / config.mask_ratio1
                train_loss = clip_loss(logits_per_mod1) + weight_a * loss_a + weight_b * loss_b
                train_loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                train_losses.append(train_loss.item())
            lr_scheduler.step()
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f'In epoch {e}, average training loss is {avg_train_loss}.')

            model.eval()
            with torch.no_grad():
                val_losses = []
                for td in tqdm(iter(val_dataloader)):
                    omics_patches1_val,omics_patches2_val, mask1_val,mask2_val,logits_per_mod1_val,logits_per_mod2_val,final_val = model(td['mod1'], td['mod2'])
                    
                    loss_c = torch.mean((omics_patches1_val - td['mod1']) ** 2 * mask1_val) / config.mask_ratio
                    loss_d =torch.mean((omics_patches2_val - td['mod2']) ** 2 * mask2_val) / config.mask_ratio1           
                    val_loss = clip_loss(logits_per_mod1_val)+ weight_a * loss_c + weight_b * loss_d
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f'In epoch {e}, average validation loss is {avg_val_loss}.')  

        # 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0  #
                print(f'Saving best model with loss {best_val_loss} at epoch {e}!')

                # save best paras
                #torch.save(model.state_dict(), f'/mnt/sdb/dmengsave/pbmc5k_clip_{e}epoch_best_model.pth')  # 
            else:
                no_improvement_count += 1

            # np.save('trloss.npy', train_losses[0])
            # np.save('valloss.npy', val_losses[0])
            if no_improvement_count >= early_stopping_patience:
                print(f'No improvement in validation loss for {early_stopping_patience} epochs. Early stopping!')
                break  # 



