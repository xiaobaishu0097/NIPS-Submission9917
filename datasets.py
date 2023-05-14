from torch.utils.data import Dataset


class GetData(Dataset):
    def __init__(self, data_feature, data_target):
        self.len = len(data_target[0, ])
        self.feature = data_feature
        self.target = data_target

    def __getitem__(self, idx):
        return self.feature[idx, ...], self.target[:, idx]

    def __len__(self):
        return self.len


class GetData1(Dataset):
    def __init__(self, data_feature, data_target):
        self.len = len(data_target)
        self.feature = data_feature
        self.target = data_target

    def __getitem__(self, idx):
        return self.feature[idx, ...], self.target[idx, ...]

    def __len__(self):
        return self.len