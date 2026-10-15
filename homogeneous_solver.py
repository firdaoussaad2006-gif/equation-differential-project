import sympy as sp

# Variable independante
x = sp.Symbol("x", real=True)

# Constantes arbitraires
C1, C2 = sp.symbols("C1 C2", real=True)


def solve_homogeneous(a, b, c):
    """Resout a*y'' + b*y' + c*y = 0."""

    # Verifier les coefficients
    try:
        a, b, c = [sp.sympify(v) for v in (a, b, c)]
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise ValueError("Coefficients invalides.") from exc

    for nom, valeur in (("a", a), ("b", b), ("c", c)):
        if valeur.is_number is not True or valeur.is_real is not True:
            raise ValueError(f"{nom} doit etre un nombre reel.")

    if a == 0:
        raise ValueError("a doit etre different de zero.")

    # Calcul du discriminant
    delta = sp.simplify(b**2 - 4*a*c)

    # Cas 1 : deux racines reelles distinctes
    if delta.is_positive:
        r1 = (-b + sp.sqrt(delta)) / (2*a)
        r2 = (-b - sp.sqrt(delta)) / (2*a)
        y = C1*sp.exp(r1*x) + C2*sp.exp(r2*x)

    # Cas 2 : racine double
    elif delta.is_zero:
        r = -b / (2*a)
        y = (C1 + C2*x)*sp.exp(r*x)

    # Cas 3 : racines complexes conjuguees
    elif delta.is_negative:
        alpha = -b / (2*a)
        beta = sp.sqrt(-delta) / (2*sp.Abs(a))
        y = sp.exp(alpha*x)*(
            C1*sp.cos(beta*x) + C2*sp.sin(beta*x)
        )

    else:
        raise ValueError("Signe du discriminant indetermine.")

    return sp.simplify(y)
