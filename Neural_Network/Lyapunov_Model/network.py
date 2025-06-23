import torch
import torch.nn as nn

class LyapunovNet(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(hidden_layer_size, out_features = 1)

        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)
        nn.init.constant_(self.l1.bias, 0.9)
        nn.init.constant_(self.l2.bias, 0.9)
        nn.init.constant_(self.l3.bias, 0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi_theta = self.l3(self.tanh2(self.l2(self.tanh1(self.l1(x)))))
        phi0_theta = self.l3(self.tanh2(self.l2(self.tanh1(self.l1(torch.zeros_like(x))))))
        Vx = torch.square(phi_theta - phi0_theta) + self.alpha * torch.norm(x)
        return Vx