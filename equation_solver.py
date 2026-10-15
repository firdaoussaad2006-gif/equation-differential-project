
import sympy as sp

# Variable symbolique commune
x = sp.Symbol("x", real=True)

# Constantes d'integration
C1, C2 = sp.symbols("C1 C2")


def solve_equation(a, b, c, f, yh, yp,
                   x0=None, y0=None, v0=None):
    """
    Combine yh et yp, applique les conditions
    initiales et verifie la solution.
    """

    # 1. Validation des donnees
    a, b, c = map(sp.sympify, (a, b, c))
    f = sp.sympify(f)
    yh = sp.sympify(yh)
    yp = sp.sympify(yp)

    if a == 0:
        raise ValueError("Le coefficient 'a' doit etre non nul.")

    # 2. Solution generale
    y_general = sp.simplify(yh + yp)
    y_final = y_general

    # 3. Gestion des conditions initiales
    valeurs = (x0, y0, v0)
    conditions_presentes = any(v is not None for v in valeurs)

    if conditions_presentes:
        if not all(v is not None for v in valeurs):
            raise ValueError(
                "Il faut fournir x0, y0 et v0 ensemble."
            )

        x0, y0, v0 = map(sp.sympify, valeurs)

        # Construction du systeme
        equations = [
            sp.Eq(y_general.subs(x, x0), y0),
            sp.Eq(
                sp.diff(y_general, x).subs(x, x0),
                v0
            )
        ]

        # Resolution pour C1 et C2
        solutions = sp.solve(
            equations, (C1, C2), dict=True
        )

        if len(solutions) != 1:
            raise ValueError(
                "Conditions initiales incompatibles "
                "ou insuffisantes."
            )

        constantes = solutions[0]

        if C1 not in constantes or C2 not in constantes:
            raise ValueError(
                "Impossible de determiner C1 et C2."
            )

        y_final = sp.simplify(
            y_general.subs(constantes)
        )

    # 4. Verification de l'equation
    residu = sp.simplify(
        a * sp.diff(y_final, x, 2)
        + b * sp.diff(y_final, x)
        + c * y_final
        - f
    )

    # 5. Verification des conditions initiales
    conditions_valides = None

    if conditions_presentes:
        conditions_valides = (
            sp.simplify(y_final.subs(x, x0) - y0) == 0
            and
            sp.simplify(
                sp.diff(y_final, x).subs(x, x0) - v0
            ) == 0
        )

    # 6. Retourner les resultats
    return {
        "yh": yh,
        "yp": yp,
        "general_solution": y_general,
        "solution": y_final,
        "constants": constantes if conditions_presentes else None,
        "residual": residu,
        "verified": residu == 0,
        "initial_conditions_verified": conditions_valides
    }
