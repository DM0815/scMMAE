##Dataloader
from torch.utils.data import Dataset, DataLoader
class MultiModalDataset_label(Dataset):
    def __init__(self, mod1_data, mod2_data, labels):
        self.mod1_data = mod1_data
        self.mod2_data = mod2_data
        self.labels = labels

    def __len__(self):
         return len(self.mod1_data)

    def __getitem__(self, idx):
        mod1_feature = self.mod1_data[idx]
        mod2_feature = self.mod2_data[idx]
        label = self.labels[idx]
        return {'mod1': mod1_feature, 'mod2': mod2_feature, 'label': label}

##Dataloader  single modality
class SingleModalDataset(Dataset):
    def __init__(self, mod1_data, labels):
        self.mod1_data = mod1_data
        #self.mod2_data = mod2_data
        self.labels = labels

    def __len__(self):
         return len(self.mod1_data)

    def __getitem__(self, idx):
        mod1_feature = self.mod1_data[idx]
        #mod2_feature = self.mod2_data[idx]
        label = self.labels[idx]
        return {'mod1': mod1_feature, 'label': label}
    
