from torch.utils.data import Dataset

class VWWDataWrapper(Dataset):
    def __init__(self, data_generator):
        self.data_generator = data_generator
    
    def __len__(self):
        return len(self.data_generator)
    
    def __getitem__(self, idx):
        return self.data_generator[idx][0], self.data_generator[idx][1]