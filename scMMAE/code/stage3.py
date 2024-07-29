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
        self.RNA_tokens = 4000 ##omics cells
        self.RNA_component = 400 ##channel        
        self.emb_RNA = 10
        

        
        self.emb_dim = 128       
        self.total_epoch = 500
        self.warmup_epoch = 10
config = Config()        

class RNA_Encoder(torch.nn.Module):
    def __init__(self,emb_dim=128,emb_RNA=10,RNA_component=400,RNA_tokens=4000, encoder_head=4,encoder_layer=6) -> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = RNA_tokens, out_features = RNA_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_RNA, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, RNA_component ,emb_dim)))#
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, encoder_head,attn_drop=0.1 ,proj_drop=0.1) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)


    def forward(self, patches):
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0),config.RNA_component,config.emb_RNA)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding
        patches = rearrange(patches, 'b c s -> c b s')
        
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')        
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
        self.RNA_Encoder = RNA_Encoder(config.emb_dim,config.emb_RNA,config.RNA_component,config.RNA_tokens,config.encoder_head,config.encoder_layer)

        
        self.ADT_in_RNA_Att = CrossAttention(config.emb_dim)
        self.fc = torch.nn.Linear(config.emb_dim, config.emb_dim)
        self.head = torch.nn.Linear(config.emb_dim, config.num_classes)
    def forward(self, patches1):
      
        patches1 = self.RNA_Encoder(patches1)

        omics_feature1 = rearrange(patches1.clone(), 't b c -> b t c')
        omics_feature1_cls = omics_feature1[:,0,:].unsqueeze(1).clone() #cls
 
        omics_feature2_cls = self.fc(omics_feature1_cls)
        
        x1 = self.ADT_in_RNA_Att(omics_feature2_cls, omics_feature1_cls, omics_feature1_cls)##只用cls_token交互

        x1 += omics_feature2_cls

        final_results = x1.squeeze(1)##
        
        logits = self.head(x1)
        logits = logits.squeeze(1)
        # print(logits.shape)
        # print(final_results.shape)
        return logits,final_results
    
    
######################################loading dataset
from dataloaderwithlabel import *
omic1 = torch.load('../dataset/RNA-seq/ifnb_rna_rpkm_normalized.pth')
labels = pd.read_csv('../dataset/RNA-seq/ifnb/ifnb_label.csv')
labels = labels['x']
labels = np.array(labels)
labels[labels == 13] = 0

config = Config()
##RNA
train_dataset,val_dataset,y_train,y_test = train_test_split(omic1,labels,test_size=0.7, random_state=42)
###second split,just 30 percent data to finetune
train_dataset,val_dataset,y_train,y_test = train_test_split(train_dataset,y_train,test_size=0.1, random_state=42)
train_dataset = train_dataset.to(torch.float).to(config.device)
val_dataset = val_dataset.to(torch.float).to(config.device)
y_train = torch.tensor(y_train, dtype=torch.long).to(config.device)
y_test = torch.tensor(y_test, dtype=torch.long).to(config.device)

multi_modal_trian_dataset = SingleModalDataset(train_dataset,y_train)
multi_modal_test_dataset = SingleModalDataset(val_dataset,y_test)

train_dataloader = torch.utils.data.DataLoader(multi_modal_trian_dataset, 128, shuffle=True,num_workers=0)
val_dataloader = torch.utils.data.DataLoader(multi_modal_test_dataset, 128, shuffle=False,num_workers=0)


##loading pretrained stage2-model
#config = Config()
#model = Omics_Pred(config).to(config.device)
#model.load_state_dict(torch.load(f'{Your Path}/{Dataset}_finetune_best_model.pth'),strict=False) ##If error, becasue haed number (number of class, cell type) is different between multimodal and unimodal,you should remove the paras and weigths of head)
####################################Training
early_stopping_patience = 5  
best_val_loss = float('inf')  
no_improvement_count = 0
##loss
lossfun = torch.nn.CrossEntropyLoss()
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

model = Omics_Pred(config).to(config.device)
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
            train_acces  = []
            for tk in tqdm(iter(train_dataloader)):
                #print(train_dataset['mod1'].shape)
                step_count += 1
                
                train_logits,final_results = model(tk['mod1'])
                train_loss = lossfun(train_logits,tk['label'])
                train_acc = acc_fn(train_logits,tk['label'])
                train_loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                train_losses.append(train_loss.item())
                train_acces.append(train_acc.item())
            lr_scheduler.step()
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_acc = sum(train_acces) / len(train_acces)
            print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

            model.eval()
            with torch.no_grad():
                val_losses = []
                val_acces = []
                for td in tqdm(iter(val_dataloader)):
                    val_logits,final_results2= model(td['mod1'])
                    
                    #td['label'] = td['label'].to(torch.long)
                    val_loss = lossfun(val_logits,td['label'])
                    val_acc = acc_fn(val_logits,td['label'])
                    val_losses.append(val_loss.item())
                    val_acces.append(val_acc.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_acc = sum(val_acces) / len(val_acces)
                print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        # 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0  #
                print(f'Saving best model with loss {best_val_loss} at epoch {e}!')

                # save best paras
                #torch.save(model.state_dict(), f'./scMMAE_{dataset}_stage3_best_model.pth')  # 
            else:
                no_improvement_count += 1

            # np.save('trloss.npy', train_losses[0])
            # np.save('valloss.npy', val_losses[0])
            if no_improvement_count >= early_stopping_patience:
                print(f'No improvement in validation loss for {early_stopping_patience} epochs. Early stopping!')
                break  # 
