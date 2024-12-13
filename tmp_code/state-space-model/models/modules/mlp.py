from torch import nn 
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        config, 
        layer_idx,
        conv1d_config=None,
        **kwargs
    ):
        d_model = config.intermediate_size
        super().__init__()
        in_features, out_features = 2* d_model, d_model
        hidden_features = d_model * 2
        # self.return_residual = return_residual

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = F.silu
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x1,x2):
        # import pdb;pdb.set_trace()
        x = torch.concat([x1,x2], dim=-1)
        # x = x1 + x2
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y 


class GLU(nn.Module):
    def __init__(
        self,
        config, 
        layer_idx,
        conv1d_config=None,
        **kwargs
    ):
        super().__init__()
        d_model = config.intermediate_size
        in_features, out_features = d_model, d_model
        hidden_features = d_model * 2
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.activation = F.silu


    def forward(self, dt, gate):
        x = self.fc1(dt) * self.activation(self.fc2(gate))
        y = self.fc3(x)
        return y