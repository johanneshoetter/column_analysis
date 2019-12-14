import torch
from torch.utils import data
import os


class ExtendedWikiSQL(data.Dataset):
    
    def __init__(self):
        self.inputs, self.targets = [], []
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'target': self.targets[idx]
        }
    
    def load_from_df(self, ewikisql_df, embedding, initialize=True):
        if initialize:
            self.inputs, self.targets = [],[]
            
        for idx, row in ewikisql_df.iterrows():
            header, question, target = row[['header', 'question', 'targets']]
            input_pre = [embedding(word.lower()) for word in question]
            input_suf = [embedding(word.lower()) for word in header]
            input_sequence = [embedding('<BEG>')] + input_pre + [embedding('<SEQ>')] + input_suf + [embedding('<END>')]

            target_pre = [0 for word in question] # question
            target_suf = [1 if indicator == '1' else 0 for indicator in target] # header
            target_sequence = [0] + target_pre + [0] + target_suf + [0]
            
            self.inputs.append(torch.Tensor(input_sequence))
            self.targets.append(torch.LongTensor(target_sequence))
                
    def load_from_torch(self, path):
        self.inputs = torch.load('{}_inputs.pt'.format(path))
        self.targets = torch.load('{}_targets.pt'.format(path))
    
    def save_to_torch(self, path):
        torch.save(self.inputs, '{}_inputs.pt'.format(path))
        torch.save(self.targets, '{}_targets.pt'.format(path))