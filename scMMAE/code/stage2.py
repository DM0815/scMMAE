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
        self.num_classes = 14                                             
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
        
        self.mask_ratio1 = 0.15        
        self.ADT_tokens = 14 ##omics proteins
        self.ADT_component =14 ##channel
        self.emb_ADT = 1 
        
        self.emb_dim = 128       
        self.total_epoch = 500
        self.warmup_epoch = 10
config = Config()        


###Starting Stage2
class RNA_Class_encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_RNA=10,RNA_component=400,RNA_tokens=4000, encoder_head=4,encoder_layer=6) -> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = RNA_tokens, out_features = RNA_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_RNA, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, RNA_component ,emb_dim)))##
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

class ADT_Class_encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_ADT=10,ADT_component=400,ADT_tokens=4000, encoder_head=4,encoder_layer=6) -> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = ADT_tokens, out_features = ADT_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_ADT, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, ADT_component ,emb_dim)))#
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, encoder_head,attn_drop=0.1 ,proj_drop=0.1) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)


    def forward(self, patches):
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0),config.ADT_component,config.emb_ADT)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding
        patches = rearrange(patches, 'b c s -> c b s')
        
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')

        features = self.layer_norm(self.transformer(patches)) 
        features = rearrange(features, 'b t c -> t b c')

        
        return features
    
#cross attention
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

        
    
class Omics_Classifier(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.RNA_encoder = RNA_Class_encoder(config.emb_dim,config.emb_RNA,config.RNA_component,config.RNA_tokens,config.encoder_head,config.encoder_layer)
        self.ADT_encoder = ADT_Class_encoder(config.emb_dim,config.emb_ADT,config.ADT_component,config.ADT_tokens,config.encoder_head,config.encoder_layer)
        
        self.RNA_in_ADT_Att = CrossAttention(config.emb_dim)
        self.ADT_in_RNA_Att = CrossAttention(config.emb_dim)

        self.head = torch.nn.Linear(config.emb_dim, config.num_classes)
    def forward(self, patches1,patches2):
      
        patches1 = self.RNA_encoder(patches1)
        patches2 = self.ADT_encoder(patches2)

        omics_feature1 = rearrange(patches1.clone(), 't b c -> b t c')
        omics_feature1_cls = omics_feature1[:,0,:].unsqueeze(1).clone() #cls
        omics_feature2 = rearrange(patches2.clone(), 't b c -> b t c')
        omics_feature2_cls = omics_feature1[:,0,:].unsqueeze(1).clone() #cls
        
        
        ##save encoder's cls_token as rna prot representation, respectively
        #rna_cls = []
        rna_cls = omics_feature1_cls
        #prot_cls = []
        prot_cls = omics_feature2_cls
        
        x2 = self.ADT_in_RNA_Att(omics_feature1_cls, omics_feature2_cls, omics_feature2_cls)## 
        x1 = self.RNA_in_ADT_Att(omics_feature2_cls, omics_feature1_cls, omics_feature1_cls)##
        x2 += omics_feature1_cls
        x1 += omics_feature2_cls
        
        
        final_results = []
        #final_result = torch.cat((x1, x2), dim=2)
        final_result = (x1+x2)/2
        final_results.append(final_result.squeeze(1))##
        
        logits = self.head(final_result)
        logits = logits.squeeze(1)

        return logits,final_results,rna_cls,prot_cls
    
    
#### paired omics with label  Dataset
from dataloaderwithlabel import *
from dataloader import MultiModalDataset
omic1 = torch.load('../dataset/CITE-seq/malt_10k_rna_rpkm.pth')
omic2 = torch.load('../dataset/CITE-seq/malt_10k_prot_clred.pth')
labels = np.load('../dataset/CITE-seq/malt10k_6838wnn_labels.npy',allow_pickle=True)
labels = labels.astype(int)

##Dataloader
##RNA
train_dataset,val_dataset,y_train,y_test = train_test_split(omic1,labels,test_size=0.7, random_state=42)
##ADT
train_dataset1,val_dataset1,y_train1,y_test1 = train_test_split(omic2,labels,test_size=0.7, random_state=42)

###Second split,just only use 30% dataset to finetune
train_dataset,val_dataset,y_train,y_test = train_test_split(train_dataset,y_train,test_size=0.1, random_state=42)
train_dataset = train_dataset.to(torch.float).to(config.device)
val_dataset = val_dataset.to(torch.float).to(config.device)
y_train = torch.tensor(y_train, dtype=torch.long).to(config.device)
y_test = torch.tensor(y_test, dtype=torch.long).to(config.device)
##ADT
train_dataset1, val_dataset1, y_train1, y_test1 = train_test_split(train_dataset1,y_train1,test_size=0.1, random_state=42)
train_dataset1 = train_dataset1.to(torch.float).to(config.device)
val_dataset1 = val_dataset1.to(torch.float).to(config.device)

multi_modal_trian_dataset = MultiModalDataset_label(train_dataset, train_dataset1,y_train)
multi_modal_test_dataset = MultiModalDataset_label(val_dataset, val_dataset1,y_test)

train_dataloader = torch.utils.data.DataLoader(multi_modal_trian_dataset, 128, shuffle=True,num_workers=0)
val_dataloader = torch.utils.data.DataLoader(multi_modal_test_dataset, 128, shuffle=False,num_workers=0)

train_dataset.shape,train_dataset1.shape,val_dataset.shape,val_dataset1.shape,y_train.shape,y_test.shape



##loading pretrained stage1-model
#model = Omics_Classifier(config).to(config.device)
#model = DataParallel(model)
#model.load_state_dict(torch.load(f'{Your Path}/scMMAE_{dataset}_pretrain_{e}epoch_best_model.pth'),strict=False)

###########################################Training
early_stopping_patience = 5  
best_val_loss = float('inf')  
no_improvement_count = 0
##loss
lossfun = torch.nn.CrossEntropyLoss()
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

model = Omics_Classifier(config).to(config.device)
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
        train_loss_list = []
        train_acc_list  = []
        val_loss_list = []
        val_acc_list  = []           
        for e in range(config.total_epoch):
            model.train()
            train_losses = []
            train_acces  = []
            for tk in tqdm(iter(train_dataloader)):
                step_count += 1
                
                train_logits,final_results,rna_cls,prot_cls = model(tk['mod1'], tk['mod2'])
                #print(omics_patches1.shape)
                #tk['label'] = tk['label'].to(torch.long)
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
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)
            print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

            model.eval()
            with torch.no_grad():
                val_losses = []
                val_acces = []
                for td in tqdm(iter(val_dataloader)):
                    val_logits,final_results2,rna_cls2,prot_cls2 = model(td['mod1'], td['mod2'])
                    
                    #td['label'] = td['label'].to(torch.long)
                    val_loss = lossfun(val_logits,td['label'])
                    val_acc = acc_fn(val_logits,td['label'])
                    val_losses.append(val_loss.item())
                    val_acces.append(val_acc.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_acc = sum(val_acces) / len(val_acces)
                val_loss_list.append(avg_val_loss)
                val_acc_list.append(avg_val_acc)
                print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        # 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0  #
                print(f'Saving best model with loss {best_val_loss} at epoch {e}!')

                # save best paras
                #torch.save(model.state_dict(), f'{Your Path}/{Dataset}_finetune_best_model.pth')   # 
            else:
                no_improvement_count += 1

            # np.save('trloss.npy', train_losses[0])
            # np.save('valloss.npy', val_losses[0])
            if no_improvement_count >= early_stopping_patience:
                print(f'No improvement in validation loss for {early_stopping_patience} epochs. Early stopping!')
                break  # 
                
##########Plotting the training and validation loss
plt.figure()
plt.plot(range(len(train_loss_list)), train_loss_list, label='Training Loss')
#plt.plot(range(len(train_acc_list)), train_acc_list, label='Training Acc')
plt.plot(range(len(val_loss_list)), val_loss_list, label='Val Loss')
#plt.plot(range(len(val_acc_list)), val_acc_list, label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


#################################Predict
MultiModal_dataset = MultiModalDataset(omic1,omic2)
all_dataloader = torch.utils.data.DataLoader(MultiModal_dataset, 128, shuffle=False,num_workers=0)

model.eval() 
fin = []
with torch.no_grad():
    for batch in all_dataloader:
        inputs1 = batch['mod1'].to(config.device)
        inputs2 = batch['mod2'].to(config.device)
        try:
            final_logits,final_results,rna_cls,prot_cls = model(inputs1, inputs2)
            fin.append(final_results[0].cpu().numpy())
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Not enough video memory, try reducing the batch size")
            else:
                raise e

fin = np.concatenate(fin, axis=0)
fin.shape


############UMAP results and calculatin metrics
from umappre import *
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.manifold import TSNE
import pandas as pd

# community detection
adj_matrix = knn_adj_matrix(fin)
y_pred = RunLeiden(adj_matrix)
##kcluster
#y_pred = RunSpectral(adj_matrix, k=14)
reducer = umap.UMAP()
embedding = reducer.fit_transform(fin)

df = pd.DataFrame({'x': pd.Series(embedding[:, 0]), 'y': pd.Series(embedding[:, 1]), 'label': y_pred})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='x', y='y', hue='label', s=10,palette = sns.color_palette(mpl.colors.TABLEAU_COLORS ,len(set(y_pred))),)
plt.title('UMAP projection of the MALT10K', fontsize=20)
plt.legend(bbox_to_anchor=(1.05, -0), loc=3, borderaxespad=4)
plt.show()


###calculating

y_test = labels
from metric import *
from sklearn import metrics
metrics_dict = {
    'Mean Average Precision': mean_average_precision(embedding, np.ravel(y_test)),
    'Avg Silhouette Width': avg_silhouette_width(embedding, np.ravel(y_test)),
    'Graph Connectivity': graph_connectivity(embedding, np.ravel(y_test)),
    'ARI': metrics.adjusted_rand_score(np.ravel(y_test), np.ravel(y_pred)),
    'NMI': metrics.normalized_mutual_info_score(np.ravel(y_test), np.ravel(y_pred)),
    'FMI': metrics.fowlkes_mallows_score(np.ravel(y_test), np.ravel(y_pred)),
    'Silhouette Coefficient': metrics.silhouette_score(embedding, y_pred, metric='euclidean'),
    'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embedding, y_pred),
    'Davies-Bouldin Index': metrics.davies_bouldin_score(embedding, y_pred),
    'Purity': purity(y_pred, y_test),
    'AMI': metrics.adjusted_mutual_info_score(y_test, y_pred),
    'Homogeneity': metrics.homogeneity_score(y_test, y_pred),
    'Completeness': metrics.completeness_score(y_test, y_pred),
    'V-measure': metrics.v_measure_score(y_test, y_pred),
    'F-measure': F_measure(y_pred, y_test),
    'Jaccard Index': jaccard(y_pred, y_test),
    'Dice Index': Dice(y_pred, y_test)
}

df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'scMMAE_MALT10K'])
#df.to_csv('/{Your Path}/scMMAE_{Dataset}_cluster_metrics.csv', index=False)
df
