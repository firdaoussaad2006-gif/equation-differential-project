
import sympy as sp

x = sp.symbols("x")


def solve_particular(a, b, c, f):
    a, b, c, f = map(sp.sympify, (a, b, c, f))

    if a == 0:
        raise ValueError("a doit etre different de zero.")

    # Resoudre chaque terme de f(x)
    solutions = []

    for terme in sp.Add.make_args(sp.expand(f)):
        yp = resoudre_un_terme(a, b, c, terme)
        solutions.append(yp)

    resultat = sp.simplify(sum(solutions, sp.Integer(0)))

    # Verifier la solution
    erreur = sp.simplify(
        a * sp.diff(resultat, x, 2)
        + b * sp.diff(resultat, x)
        + c * resultat - f
    )

    if erreur != 0:
        raise ValueError("La verification a echoue.")

    return resultat


def resoudre_un_terme(a, b, c, f):

    # Cas 1 : polynome
    if not f.has(sp.exp, sp.sin, sp.cos):
        if not f.is_polynomial(x):
            raise ValueError("Fonction non prise en charge.")

        degre = sp.Poly(f, x).degree()

        for k in range(3):
            coeffs = sp.symbols(f"A0:{degre + k + 1}")
            yp = sum(coeffs[i] * x**i
                     for i in range(degre + k + 1))

            solution = calculer(a, b, c, f, yp, coeffs)

            if solution is not None:
                return solution

    # Cas 2 : exponentielle exp(lambda*x)
    elif f.has(sp.exp) and not f.has(sp.sin, sp.cos):
        exp_f = list(f.atoms(sp.exp))

        if len(exp_f) != 1:
            raise ValueError("Exponentielle non prise en charge.")

        e = exp_f[0]
        lam = sp.diff(e.args[0], x)
        P = sp.simplify(f / e)

        if (sp.simplify(e.args[0] - lam*x) != 0
                or not P.is_polynomial(x)):
            raise ValueError("Exponentielle non prise en charge.")

        degre = sp.Poly(P, x).degree()

        for k in range(3):
            coeffs = sp.symbols(f"A0:{degre + 1}")
            polynome = sum(coeffs[i] * x**i
                           for i in range(degre + 1))

            yp = x**k * e * polynome
            solution = calculer(
                a, b, c, f, yp, coeffs, facteur=e
            )

            if solution is not None:
                return solution

    # Cas 3 : sinus ou cosinus
    elif not f.has(sp.exp):
        fonctions = list(f.atoms(sp.sin)) + list(f.atoms(sp.cos))

        if len(fonctions) != 1:
            raise ValueError("Sinus ou cosinus non pris en charge.")

        fonction = fonctions[0]
        angle = fonction.args[0]
        omega = sp.diff(angle, x)
        amplitude = sp.simplify(f / fonction)

        if (sp.simplify(angle - omega*x) != 0
                or omega == 0 or amplitude.has(x)):
            raise ValueError("Fonction trigonometrique non prise en charge.")

        A, B = sp.symbols("A B")

        for k in range(3):
            yp = x**k * (
                A * sp.cos(omega*x) + B * sp.sin(omega*x)
            )

            solution = calculer(
                a, b, c, f, yp, (A, B), omega=omega
            )

            if solution is not None:
                return solution

    raise ValueError("Aucune solution particuliere trouvee.")


def calculer(a, b, c, f, yp, coeffs, facteur=None, omega=None):
    # Calculer l'equation avec le candidat
    erreur = sp.expand(
        a * sp.diff(yp, x, 2)
        + b * sp.diff(yp, x)
        + c * yp - f
    )

    # Simplifier le facteur exponentiel commun
    if facteur is not None:
        erreur = sp.simplify(erreur / facteur)

    # Transformer les sinus et cosinus en symboles
    C, S = sp.symbols("C S")

    if omega is not None:
        erreur = sp.expand_trig(erreur)
        erreur = sp.expand(erreur).xreplace({
            sp.cos(omega*x): C,
            sp.sin(omega*x): S
        })

        variables = (x, C, S)
    else:
        variables = (x,)

    # Extraire les coefficients
    try:
        polynome = sp.Poly(erreur, *variables)
        equations = polynome.coeffs()
    except sp.PolynomialError:
        equations = [erreur]

    # Resoudre les coefficients inconnus
    solutions = sp.solve(equations, coeffs, dict=True)

    for valeurs in solutions:
        candidat = sp.simplify(yp.subs(valeurs))

        verification = sp.simplify(
            a * sp.diff(candidat, x, 2)
            + b * sp.diff(candidat, x)
            + c * candidat - f
        )

        if verification == 0:
            return candidat

    return None
