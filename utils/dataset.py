import torch
import os


class ExtendedWikiSQL():
    
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

            target = [float(val) for val in target]

            self.inputs.append(torch.Tensor(input_sequence))
            self.targets.append(torch.Tensor(target))
                
    def load_from_torch(self, path):
        self.inputs = torch.load('{}_inputs.pt'.format(path))
        self.targets = torch.load('{}_targets.pt'.format(path))
    
    def save_to_torch(self, path):
        torch.save(self.inputs, '{}_inputs.pt'.format(path))
        torch.save(self.targets, '{}_targets.pt'.format(path))