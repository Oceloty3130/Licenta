import torch
import torch.nn as nn
from .custom_state_function import state_fcn


def ell_hat_loss_individual_state(model: nn.Module, x: torch.Tensor, expressions: list[str], gamma: float = 0.01) -> torch.Tensor:
    # Asigură că x are dimensiuni [1, order] și poate primi gradient
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.clone().detach().requires_grad_(True)

    output = model(x)
    # Calculăm gradientul output față de x
    grads = torch.autograd.grad(output, x, retain_graph=True, create_graph=True)[0]
    # Înlăturăm dimensiunea batch pentru dot product
    grads = grads.view(-1)

    # Evaluăm funcția de stare f(x)
    f_val = state_fcn(expressions, x)
    f_val = f_val.view(-1)

    # Dot product între gradienți și f(x) + termen de regularizare
    comp = torch.dot(grads, f_val) + gamma * torch.norm(x)
    return comp


def custom_lyap_loss(x_min: float, x_max: float, step: float, model: nn.Module, expressions: list[str], gamma: float = 0.01) -> torch.Tensor:
    """
    Estimează loss-ul integrând pe grid-ul
    [x_min, x_max]^n cu pasul step.
    """
    device = next(model.parameters()).device
    order = len(expressions)

    # Construim grid-ul de puncte
    xs = torch.arange(x_min, x_max + 1e-9, step, device=device)
    grid = torch.cartesian_prod(*[xs for _ in range(order)])

    loss = torch.tensor(0.0, device=device)

    for x_pt in grid:
        # Conversie sigură din tensor existent și shape [1, order]
        x_pt = x_pt.clone().detach().view(1, -1).to(dtype=torch.float32, device=device).requires_grad_(True)
        comp = ell_hat_loss_individual_state(model, x_pt, expressions, gamma)
        loss = loss + torch.square(torch.relu(comp))

    # Returnăm media pierderilor pe toate punctele
    return loss / grid.size(0)
