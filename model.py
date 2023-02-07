import os

import qlib

import torch
import time
import numpy as np
import pandas as pd
from torch import nn
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import ZScoreNorm, Fillna
from qlib.utils import init_instance_by_config
from torch.utils.data import DataLoader, Dataset, Sampler
from audtorch.metrics.functional import pearsonr
import math
from qlib.contrib.model.pytorch_transformer import Transformer
from qlib.contrib.model.pytorch_alstm import ALSTMModel
from qlib.contrib.evaluate import backtest_daily

import torch.nn.functional as F
from torch.autograd import Function
import logging
import warnings
from qlib.utils.time import Freq
import pandas as pd
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import WeightStrategyBase
from qlib.contrib.strategy import TopkDropoutStrategy
from tqdm import tqdm
from qlib.contrib.model.pytorch_nn import DNNModelPytorch
# from torch_qp_solver import torch_qp_solver
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)
device = "cuda:1" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=128, num_layers=3, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d'%i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d'%i, nn.Linear(
                d_feat if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('bd_%d' % i, nn.BatchNorm1d(hidden_size))
            self.mlp.add_module('relu_%d'%i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))

    def forward(self, x):

        return self.mlp(x).squeeze()


class ECONet:
    pass




