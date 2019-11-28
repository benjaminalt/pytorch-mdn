import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

cuda = torch.cuda.is_available()


# 2-layer lstm with mixture of gaussian parameters as outputs
# with skip connections
class LSTMRandWriter(nn.Module):
    def __init__(self, input_size, output_size, cell_size, num_clusters):
        super(LSTMRandWriter, self).__init__()

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = cell_size, num_layers = 1, batch_first=True)
        self.dropout1 = nn.Dropout2d()
        self.lstm2 = nn.LSTM(input_size = cell_size+input_size, hidden_size = cell_size, num_layers = 1, batch_first=True)
        self.dropout2 = nn.Dropout2d()
        self.lstm3 = nn.LSTM(input_size = cell_size * 2, hidden_size = cell_size, num_layers = 1, batch_first=True)
        self.dropout3 = nn.Dropout2d()
        self.lstm4 = nn.LSTM(input_size = cell_size * 2, hidden_size = cell_size, num_layers = 1, batch_first=True)
        self.dropout4 = nn.Dropout2d()
        self.linear1 = nn.Linear(cell_size*2, 1+ num_clusters*6)
        self.tanh = nn.Tanh()

    def forward(self, x, prev, prev2, prev3, prev4):
        h1, (h1_n, c1_n) = self.lstm(x, prev)

        h1 = self.dropout1(h1)
        x2 = torch.cat([h1, x], dim=-1)  # skip connection
        h2, (h2_n, c2_n) = self.lstm2(x2, prev2)

        h2 = self.dropout2(h2)
        x3 = torch.cat([h1, h2], dim=-1)  # skip connection
        h3, (h3_n, c3_n) = self.lstm3(x3, prev3)

        h3 = self.dropout3(h3)
        x4 = torch.cat([h2, h3], dim=-1)  # skip connection
        h4, (h4_n, c4_n) = self.lstm4(x4, prev4)

        h4 = self.dropout4(h4)
        h = torch.cat([h3, h4], dim=-1)  # skip connection
        params = self.linear1(h)

        mog_params = params.narrow(-1, 0, params.size()[-1] - 1)
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho = mog_params.chunk(6, dim=-1)
        weights = F.softmax(pre_weights, dim=-1)
        eps = 1e-6
        rho = torch.clamp(self.tanh(pre_rho), min=-1+eps, max=1-eps)
        end = F.sigmoid(params.narrow(-1, params.size()[-1] - 1, 1))

        return end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, (h1_n, c1_n), (h2_n, c2_n), (h3_n, c3_n), (h4_n, c4_n)
