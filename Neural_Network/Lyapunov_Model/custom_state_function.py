import sympy as sp
import torch
from functools import lru_cache
from .custom_function import *

_OPERATORS = {
    'sin': custom_sin,
    'cos': custom_cos,
    'tanh': custom_tanh,
    'cosh': custom_cosh,
    'sinh': custom_sinh,
    '*' : custom_mul,
    '**': custom_pow,
    '/': custom_div,
}

@lru_cache(maxsize=None)
def custom_function(expr_str: str, order: int):
    """
    Primește un șir de forma "x1 + x2**2" și un număr de variabile `order`.
    Returnează o funcție Python f(x_vec).
    """
    # Normalizează operatorul ^ în Python
    expr_str = expr_str.replace('^', '**')

    # Construiește simbolurile x1…xN
    symbols = sp.symbols(' '.join(f'x{i + 1}' for i in range(order)))

    # Parsea expresia
    expr = sp.sympify(expr_str)

    # Lambdify cu operatori personalizați + numpy fallback
    func = sp.lambdify(symbols, expr, modules=[_OPERATORS, 'numpy'])

    def f(x_vec):
        if len(x_vec) != order:
            raise ValueError(f"Expected {order} variables, got {len(x_vec)}")
        return func(*x_vec)

    return f


def interpret_function(expr_str: str, x: torch.Tensor) -> float:
    """
    Primește o expresie de forma "dx1/dt = x2 - x1" și tensorul x.
    Construiește și evaluează funcția sympy doar o singură dată per expresie.
    """
    # Determină numărul de variabile din x
    order = x.numel() if x.dim() == 1 else x.size(-1)
    func = custom_function(expr_str, order)
    return func(x.flatten().tolist())


def state_fcn(expressions: list[str], x: torch.Tensor) -> torch.Tensor:
    """
    Construiește vectorul f(x) evaluând fiecare expresie din listă.
    - expressions: ["dx1/dt = ...", "dx2/dt = ...", ...]
    - x: tensor de dim [order] sau [1, order]
    Returnează un tensor shape [order].
    """
    x_vec = x.flatten()
    vals = [interpret_function(expr, x_vec) for expr in expressions]
    return torch.tensor(vals, dtype=x.dtype, device=x.device)
