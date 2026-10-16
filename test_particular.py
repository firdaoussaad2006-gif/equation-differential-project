import sympy as sp
from particular_solver import solve_particular

x = sp.symbols("x")

# Equation: y'' + y' + y = x^2 * exp(-x)
f = x**2 * sp.exp(-x)

# Calculer la solution particuliere
yp = solve_particular(1, 1, 1, f)

print("Solution particuliere :", yp)

# Verifier la solution
verification = sp.simplify(
    sp.diff(yp, x, 2)
    + sp.diff(yp, x)
    + yp
    - f
)

print("Verification :", verification)