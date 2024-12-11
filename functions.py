import numpy as np
import matplotlib.pyplot as plt

# Função Quadrática Simples
def func_quadratic(x):
    x1, x2 = x
    return x1**2 + x2**2  # Mínimo em (0,0)

# Função de Rosenbrock
def func_rosenbrock(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2  # Mínimo em (1,1)

# Função Ackley
def func_ackley(x):
    x1, x2 = x
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

# Função Griewank
def func_griewank(x):
    x1, x2 = x
    return 1 + (x1**2 + x2**2) / 4000 - np.cos(x1) * np.cos(x2 / np.sqrt(2))

# Função de Schwefel
def func_schwefel(x):
    x1, x2 = x
    return 418.9829 * 2 - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))

# Função de Himmelblau
def func_himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Função de Michalewicz
def func_michalewicz(x):
    x1, x2 = x
    return -np.sin(x1) * np.sin(x1**2 / np.pi)**20 - np.sin(x2) * np.sin(2 * x2**2 / np.pi)**20

# Função de Levy
def func_levy(x):
    x1, x2 = x
    term1 = np.sin(np.pi * (1 + (x1 - 1) / 4))**2
    term2 = (x1 - 1)**2 * (1 + np.sin(np.pi * (1 + (x1 - 1) / 4))**2)
    term3 = (x2 - 1)**2 * (1 + np.sin(np.pi * (1 + (x2 - 1) / 4))**2)
    return term1 + term2 + term3