import sympy as sp
import Custom_Function as cf

x1, x2 = sp.symbols('x1 x2')

def custom_function(expr_str):
    expr = sp.sympify(expr_str)
    function = sp.lambdify((x1, x2), expr, modules=[{'sin': cf.custom_sin()},
                                                    {'cos': cf.custom_cos()},
                                                    {'tanh': cf.custom_tanh()},
                                                    {'cosh': cf.custom_cosh()},
                                                    {'sinh': cf.custom_sinh()},'numpy'])

    return function