import torch
import torch.nn as nn
from typing import Tuple


class FullBasicModel(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, out_size, W_tensor, N, p,
                 thetahat_tensor):
        super().__init__()
        self.beta1 = nn.parameter.Parameter(
            torch.tensor(0, dtype=torch.float32))
        self.beta2 = nn.parameter.Parameter(
            torch.tensor(0, dtype=torch.float32))
        self.h1 = nn.Linear(in_size, h1_size)
        self.relu = nn.ReLU()
        self.h2 = nn.Linear(h1_size, h2_size)
        self.out = nn.Linear(h2_size, out_size)
        self.W = W_tensor
        self.N = N
        self.p = p

    def forward(self, x):
        #print(x.shape)
        #print(x[:,:,3:].shape)
        h1_relu = self.relu(self.h1(x[:, :, 3:]))
        h2_relu = self.relu(self.h2(h1_relu))
        out = self.out(h2_relu).squeeze(-1)
        predict = out + self.beta1 * x[:, :, 1] + self.beta2 * x[:, :, 2]
        return predict, out


class FullBasicModel1(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, h3_size, out_size, W_tensor,
                 N, p, thetahat_tensor):
        super().__init__()
        self.beta1 = nn.parameter.Parameter(
            torch.tensor(0, dtype=torch.float32))
        self.beta2 = nn.parameter.Parameter(
            torch.tensor(0, dtype=torch.float32))
        self.h1 = nn.Linear(in_size, h1_size)
        self.relu = nn.ReLU()
        self.h2 = nn.Linear(h1_size, h2_size)
        self.h3 = nn.Linear(h3_size, h3_size)
        self.out = nn.Linear(h3_size, out_size)
        self.W = W_tensor
        self.N = N
        self.p = p

    def forward(self, x):
        h1_relu = self.relu(self.h1(x[:, :, 3:]))
        h2_relu = self.relu(self.h2(h1_relu))
        h3_relu = self.relu(self.h3(h2_relu))
        out = self.out(h3_relu).squeeze(-1)
        predict = out + self.beta1 * x[:, :, 1] + self.beta2 * x[:, :, 2]
        return predict, out


class gstarhat(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, out_size, W_tensor, N, p,
                 thetahat_tensor):
        super().__init__()
        self.h1 = nn.Linear(in_size, h1_size)
        self.relu = nn.ReLU()
        self.h2 = nn.Linear(h1_size, h2_size)
        self.out = nn.Linear(h2_size, out_size)
        self.W = W_tensor
        self.N = N
        self.p = p

    def forward(self, x):
        h1_relu = self.relu(self.h1(x))
        h2_relu = self.relu(self.h2(h1_relu))
        out = self.out(h2_relu).squeeze(-1)
        return out


class gstarhat1(nn.Module):
    def __init__(self, in_size, h1_size, h2_size, h3_size, out_size, W_tensor,
                 N, p, thetahat_tensor):
        super().__init__()
        self.h1 = nn.Linear(in_size, h1_size)
        self.relu = nn.ReLU()
        self.h2 = nn.Linear(h1_size, h2_size)
        self.h3 = nn.Linear(h2_size, h3_size)
        self.out = nn.Linear(h3_size, out_size)
        self.W = W_tensor
        self.N = N
        self.p = p

    def forward(self, x):
        h1_relu = self.relu(self.h1(x))
        h2_relu = self.relu(self.h2(h1_relu))
        h3_relu = self.relu(self.h3(h2_relu))
        out = self.out(h2_relu).squeeze(-1)
        return out


class BaselineRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, input_size)
        # output: (batch_size, seq_len, output_size)
        # h_n: (num_layers, batch_size, output_size)
        output, h_n = self.rnn(x[..., 3:])
        return output, h_n