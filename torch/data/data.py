import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



class Dataset_name(Dataset):
    def __init__(self, flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        self.__load_data__()

    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

    def __load_data__(self, csv_paths: list):
        pass
        print(
            "train_X.shape:{}\ntrain_Y.shape:{}\nvalid_X.shape:{}\nvalid_Y.shape:{}\n"
            .format(self.train_X.shape, self.train_Y.shape, self.valid_X.shape, self.valid_Y.shape))
