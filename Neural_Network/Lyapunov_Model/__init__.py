import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

import Custom_State_Function as cstf
import Security as sy

class LyapunovNet(nn.Module):
    def __init__(self, input_size, hidden_layer_size, alpha):
        super(LyapunovNet, self).__init__()
        self.input_size = input_size
        self.alpha = alpha

        self.hidden_layer_size = hidden_layer_size
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.tanh1 = nn.Tanh()
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.tanh2 = nn.Tanh()
        self.l3 = nn.Linear(hidden_layer_size, out_features=1)

        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

        nn.init.constant_(self.l1.bias, 0.9)
        nn.init.constant_(self.l2.bias, 0.9)
        nn.init.constant_(self.l3.bias, 0.9)

    def forward(self, x):
        x1 = self.l1(x)
        than1 = self.tanh1(x1)
        x2 = self.l2(than1)
        than2 = self.tanh2(x2)
        x3 = self.l3(than2)
        phi_theta = x3

        x10 = self.l1(torch.zeros_like(x))
        than10 = self.tanh1(x10)
        x20 = self.l2(than10)
        than20 = self.tanh2(x20)
        x30 = self.l3(than20)
        phi0_theta = x30

        Vx = torch.square(phi_theta - phi0_theta) + self.alpha * torch.norm(x)
        return Vx


# def state_fcn(x):
#     dx1 = -math.sin(x[1])
#     dx = (x[0] - 1) * math.cos(x[1]) - 6.42 * torch.sin(x[1]) + 0.15
#     return torch.tensor([dx1, dx])

def interpret_function(expr_str, x):
    dx = cstf.custom_function(expr_str)
    return dx(x[0],x[1])

def state_fcn(expr_str1, expr_str2, x):
    if expr_str1 is None and expr_str2 is None:
        return 0

    if expr_str1 is None:
        dx = interpret_function(expr_str2, x)
        return dx

    if expr_str2 is None:
        dx  = interpret_function(expr_str1, x)
        return dx

    dx = interpret_function(expr_str1, x)
    dx1 = interpret_function(expr_str2, x)
    return torch.tensor([dx, dx1])


def ell_hat_loss_individual_state(model, input_tensor, gamma=0.01):
    output = model(input_tensor)
    gradients = torch.autograd.grad(output, input_tensor, retain_graph=True, create_graph=True)[0]
    loss_for_input_tensor = gradients * state_fcn(expr_str1, expr_str2, input_tensor) + gamma * torch.norm(input_tensor)
    return loss_for_input_tensor


def custom_lyap_loss(x1_min, x1_max, dx1, model, gamma=0.01):
    loss = torch.tensor([0], dtype=torch.float32)

    num_points = 0
    x1 = x1_min
    while x1 < x1_max:
        x1_2 = x1_min
        while x1_2 < x1_max:
            x1_tensor = torch.tensor([x1,x1_2], dtype=torch.float32, requires_grad=True)
            component = ell_hat_loss_individual_state(model, x1_tensor, gamma)
            tensor_relu = torch.relu(component)
            value = 0
            tensor_relu = tensor_relu[(tensor_relu != value)]
            if len(tensor_relu) == 0:
                tensor_relu = torch.tensor([0], dtype=torch.float32, requires_grad=True)
            loss += torch.square(tensor_relu.sum())
            x1_2 += dx1
            num_points += 1
        x1 += dx1

    loss = loss/num_points
    return loss


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.01
input_size = 2
hidden_layer_size = 10
alpha = 0.1
x1_min = -1
x1_max = 1
dx1 = 1e-2

model = LyapunovNet(input_size, hidden_layer_size, alpha).to(device)

if os.path.exists("lyapunov_model.pt"):
    model.load_state_dict(torch.load("lyapunov_model.pt", map_location=device))
    model.eval()
    print("Model loaded.")
else:
    print("Training a new model.")

if sy.verify_model_integrity_encrypted(model):
    print("Model integrity OK.")
else:
    print("WARNING: Model integrity compromised!")

expr_str1 = input("Please enter the expression string: ")
expr_str2 = input("Please enter the expression string: ")

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
loss = 1
epoch = 0

while epoch < 200:
    optimizer.zero_grad()
    loss = custom_lyap_loss(x1_min, x1_max, dx1, model)
    loss.backward(retain_graph=True)
    optimizer.step()
    epoch += 1
    print(f"Epoch {epoch} - Loss = {loss.item()}")

    if epoch % 10 == 0:
        # Save model with encrypted hash
        sy.save_model_with_encrypted_hash(model)
        sy.save_obfuscated_model(model)

        # Check model integrity later
        valid = sy.verify_model_integrity_encrypted(model)
        print("Encrypted model integrity verified:", valid)

    # Visualization
    x1_vals = np.linspace(x1_min, x1_max, 100)
    x2_vals = np.linspace(x1_min, x1_max, 100)
    V_vals = np.zeros((100, 100))
    DV_vals = np.zeros((100, 100))

    for i, x1 in enumerate(x1_vals):
        for j, x2 in enumerate(x2_vals):
            x_sample = torch.tensor([x1, x2], dtype=torch.float32, requires_grad=True)
            output = model(x_sample)
            gradients = torch.autograd.grad(outputs=output, inputs=x_sample, retain_graph=True, create_graph=True)[0]
            f_val = state_fcn(expr_str1, expr_str2, x_sample)
            dot = torch.dot(f_val, gradients)

            V_vals[i, j] = output.item()
            DV_vals[i, j] = dot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x1_vals, x2_vals)
    Z = np.reshape(DV_vals, X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('V(x)')
    plt.show()