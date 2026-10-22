import hashlib

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp

from equation_solver import solve_equation, C1, C2
from homogeneous_solver import solve_homogeneous
from particular_solver import solve_particular

x = sp.Symbol("x")
x_reel = sp.Symbol("x", real=True)

st.set_page_config(
    page_title="Équations différentielles | Solveur",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Style visuel ----------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: Inter, sans-serif; }
      .stApp { background: #F4F7FB; }
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
      [data-testid="stSidebar"] { background: #F8FAFD; border-right: 1px solid #DCE5EF; }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
      [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF; border: 1px solid #DDE6F0; border-radius: 15px;
        padding: .15rem .15rem .25rem .15rem; box-shadow: 0 3px 12px rgba(15, 40, 70, .035);
      }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #334155; font-size: .82rem; font-weight: 650; margin-bottom: .25rem;
      }
      [data-testid="stSidebar"] input {
        background: #FBFDFF !important; border: 1px solid #D4DFEA !important;
        border-radius: 9px !important; min-height: 2.55rem;
      }
      [data-testid="stSidebar"] input:focus {
        border-color: #1687B8 !important; box-shadow: 0 0 0 2px rgba(22, 135, 184, .12) !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #FBFDFF; border-color: #D4DFEA; border-radius: 9px; min-height: 2.55rem;
      }
      .input-panel-title { color: #102A43; font-size: 1.3rem; font-weight: 800; margin-bottom: .15rem; }
      .input-panel-subtitle { color: #64748B; font-size: .83rem; line-height: 1.45; margin-bottom: .9rem; }
      .input-section { color: #1769AA; font-weight: 800; font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; }
      .field-note { color: #718096; font-size: .75rem; line-height: 1.4; margin-top: -.25rem; }
      .sidebar-progress { background: #EAF2F9; border-radius: 10px; padding: .65rem .75rem; color: #33516D; font-size: .82rem; margin: .5rem 0 .9rem 0; }
      .hero {
        padding: 1.65rem 1.8rem; border-radius: 22px;
        background: linear-gradient(115deg, #102A43 0%, #1769AA 68%, #0EA5A4 120%);
        color: white; margin-bottom: 1.1rem;
        box-shadow: 0 12px 32px rgba(16,42,67,.12);
      }
      .hero h1 { color: white; font-size: clamp(1.65rem, 3vw, 2.25rem); margin: 0 0 .45rem 0; font-weight: 800; }
      .hero p { color: #E7F4FF; margin: 0; font-size: 1rem; }
      .eyebrow { color: #D3F7F4; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 800; margin-bottom: .55rem; }
      .section-label { color: #1769AA; font-weight: 800; letter-spacing: .04em; font-size: .82rem; text-transform: uppercase; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF; border: 1px solid #E1E8F0; border-radius: 17px;
      }
      div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #1769AA, #0E8D9A); color: white;
        border: 0; border-radius: 11px; min-height: 3rem; font-weight: 750;
      }
      div.stButton > button[kind="primary"]:hover { border: 0; filter: brightness(.95); }
      [data-testid="stMetric"] {
        background: white; border: 1px solid #E1E8F0; border-radius: 15px; padding: .85rem 1rem;
      }
      .equation-caption { color: #64748B; font-size: .88rem; margin-bottom: .35rem; }
      .preview-card { background: #FFFFFF; border: 1px solid #DCE7F2; border-left: 5px solid #0EA5A4; border-radius: 16px; padding: 1rem 1.2rem; margin: .5rem 0 1.2rem 0; }
      .hint { color: #64748B; font-size: .88rem; }
      .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
      .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Outil mathématique · ordre 2</div>
      <h1>∿ Solveur d’équations différentielles</h1>
      <p>Configurez votre équation, obtenez une solution symbolique et explorez son comportement graphique.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def convertir_symbole_x(expression):
    expression = sp.sympify(expression)
    substitutions = {}
    for symbole in expression.free_symbols:
        if symbole.name == "x" and symbole != x_reel:
            substitutions[symbole] = x_reel
        elif symbole.name == "C1" and symbole != C1:
            substitutions[symbole] = C1
        elif symbole.name == "C2" and symbole != C2:
            substitutions[symbole] = C2
    return expression.xreplace(substitutions)


def build_function(kind, params):
    if kind == "Exponentielle":
        return sp.sympify(params["A"]) * sp.exp(sp.sympify(params["lambda"]) * x)
    if kind == "Trigonométrique":
        alpha = sp.sympify(params["alpha"])
        beta = sp.sympify(params["beta"])
        omega = sp.sympify(params["omega"])
        return alpha * sp.sin(omega * x) + beta * sp.cos(omega * x)
    if kind == "Polynôme":
        return (
            sp.sympify(params["a2"]) * x**2
            + sp.sympify(params["a1"]) * x
            + sp.sympify(params["a0"])
        )
    raise ValueError("Type de fonction non reconnu.")


def equation_signature(values):
    """Signature des paramètres pour éviter d'afficher des résultats périmés."""
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def solve_problem(a_text, b_text, c_text, kind, params, x0_text, y0_text,
                  v0_text, xmin_text, xmax_text):
    try:
        a, b, c = map(sp.sympify, (a_text, b_text, c_text))
    except (sp.SympifyError, TypeError) as exc:
        raise ValueError("Les coefficients a, b et c doivent être des expressions numériques valides.") from exc

    if not all(v.is_number and v.is_real for v in (a, b, c)):
        raise ValueError("a, b et c doivent être des nombres réels.")
    if a == 0:
        raise ValueError("Le coefficient a doit être non nul pour une équation du second ordre.")

    f_interface = build_function(kind, params)
    yh = solve_homogeneous(a, b, c)

    # Le solveur particulier prend en charge un terme trigonométrique par appel.
    if kind == "Trigonométrique":
        alpha = sp.sympify(params["alpha"])
        beta = sp.sympify(params["beta"])
        omega = sp.sympify(params["omega"])
        morceaux = []
        if alpha != 0:
            morceaux.append(solve_particular(a, b, c, alpha * sp.sin(omega * x)))
        if beta != 0:
            morceaux.append(solve_particular(a, b, c, beta * sp.cos(omega * x)))
        yp = sp.simplify(sum(morceaux, sp.Integer(0)))
    else:
        yp = solve_particular(a, b, c, f_interface)

    f = convertir_symbole_x(f_interface)
    yh = convertir_symbole_x(yh)
    yp = convertir_symbole_x(yp)

    if not all(str(v).strip() for v in (x0_text, y0_text, v0_text)):
        raise ValueError("Remplissez x₀, y(x₀) et y′(x₀).")
    try:
        x0, y0, v0 = map(sp.sympify, (x0_text, y0_text, v0_text))
        xmin, xmax = float(xmin_text), float(xmax_text)
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise ValueError("Vérifiez les conditions initiales et les bornes du graphique.") from exc

    result = solve_equation(a, b, c, f, yh, yp, x0=x0, y0=y0, v0=v0)
    solution = sp.simplify(result["solution"])
    free_constants = [s for s in solution.free_symbols if s.name in ("C1", "C2")]
    if free_constants:
        raise ValueError("C₁ et C₂ n’ont pas été déterminées. Vérifiez les conditions initiales.")
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
        raise ValueError("La borne de début doit être inférieure à la borne de fin.")

    constants = result.get("constants") or {}
    return {
        "a": a, "b": b, "c": c, "f": f, "result": result,
        "solution": solution, "c1": constants.get(C1), "c2": constants.get(C2),
        "xmin": xmin, "xmax": xmax,
    }


# ---------- Panneau de saisie (hors formulaire pour conserver l’aperçu en direct) ----------
with st.sidebar:
    st.markdown('<div class="input-panel-title">Votre problème</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-panel-subtitle">Renseignez les paramètres ci-dessous. L’équation se met à jour automatiquement.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-progress">✦ <b>4 étapes simples</b><br/>Coefficients → Fonction → Conditions → Calcul</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="input-section">01 &nbsp; Coefficients de l’équation</p>', unsafe_allow_html=True)
        st.caption("Pour : ay″ + by′ + cy = f(x)")
        col_a, col_b, col_c = st.columns(3, gap="small")
        a_text = col_a.text_input("a", "1", key="coef_a", help="Coefficient de y″ ; il doit être différent de zéro.")
        b_text = col_b.text_input("b", "-3", key="coef_b", help="Coefficient de y′.")
        c_text = col_c.text_input("c", "2", key="coef_c", help="Coefficient de y.")

    with st.container(border=True):
        st.markdown('<p class="input-section">02 &nbsp; Second membre f(x)</p>', unsafe_allow_html=True)
        kind = st.selectbox("Forme de la fonction", ["Exponentielle", "Trigonométrique", "Polynôme"], key="function_kind")
        params = {}
        if kind == "Exponentielle":
            p1, p2 = st.columns(2, gap="small")
            params["A"] = p1.text_input("Amplitude A", "1", key="exp_A", help="Facteur devant l’exponentielle.")
            params["lambda"] = p2.text_input("Exposant λ", "1", key="exp_lambda", help="Paramètre dans exp(λx).")
            st.markdown('<div class="field-note">Exemple : f(x) = A e<sup>λx</sup></div>', unsafe_allow_html=True)
        elif kind == "Trigonométrique":
            p1, p2 = st.columns(2, gap="small")
            params["alpha"] = p1.text_input("Coefficient α · sin", "1", key="trig_alpha")
            params["beta"] = p2.text_input("Coefficient β · cos", "1", key="trig_beta")
            params["omega"] = st.text_input("Fréquence ω", "1", key="trig_omega", help="Fréquence angulaire des fonctions sinus et cosinus.")
            st.markdown('<div class="field-note">f(x) = α sin(ωx) + β cos(ωx)</div>', unsafe_allow_html=True)
        else:
            p1, p2 = st.columns(2, gap="small")
            params["a2"] = p1.text_input("Coefficient x²", "1", key="poly_a2")
            params["a1"] = p2.text_input("Coefficient x", "0", key="poly_a1")
            params["a0"] = st.text_input("Terme constant", "1", key="poly_a0")
            st.markdown('<div class="field-note">f(x) = a₂x² + a₁x + a₀</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="input-section">03 &nbsp; Conditions initiales</p>', unsafe_allow_html=True)
        st.caption("Valeurs connues au point x₀")
        x0_text = st.text_input("Point initial x₀", "0", key="initial_x0")
        p1, p2 = st.columns(2, gap="small")
        y0_text = p1.text_input("Valeur y(x₀)", "1", key="initial_y0")
        v0_text = p2.text_input("Dérivée y′(x₀)", "0", key="initial_v0")

    with st.container(border=True):
        st.markdown('<p class="input-section">04 &nbsp; Intervalle du graphique</p>', unsafe_allow_html=True)
        st.caption("Choisissez la plage de valeurs de x à afficher.")
        p1, p2 = st.columns(2, gap="small")
        xmin_text = p1.text_input("Début de x", "-5", key="graph_xmin")
        xmax_text = p2.text_input("Fin de x", "5", key="graph_xmax")

    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    calculate = st.button("✦  Calculer la solution", type="primary", use_container_width=True)
    st.caption("Vous pouvez saisir des nombres ou des expressions comme `1/2` et `pi`.")

# ---------- Équation visible et mise à jour en direct ----------
try:
    a_preview, b_preview, c_preview = [sp.sympify(v) for v in (a_text, b_text, c_text)]
    f_preview = build_function(kind, params)
    equation_preview = (
        sp.latex(a_preview) + r"\,y''(x) + " + sp.latex(b_preview) + r"\,y'(x) + "
        + sp.latex(c_preview) + r"\,y(x) = " + sp.latex(f_preview)
    )
    preview_valid = True
except Exception:
    equation_preview = r"a\,y''(x)+b\,y'(x)+c\,y(x)=f(x)"
    preview_valid = False

with st.container(border=True):
    st.markdown('<div class="equation-caption">VOTRE ÉQUATION DIFFÉRENTIELLE · APERÇU EN DIRECT</div>', unsafe_allow_html=True)
    st.latex(equation_preview)
    if preview_valid:
        st.caption("L’aperçu est actualisé automatiquement lorsque vous modifiez les paramètres.")
    else:
        st.warning("Certaines valeurs ne sont pas encore valides. Corrigez-les dans le panneau de gauche pour afficher l’équation complète.")

current_values = (
    a_text, b_text, c_text, kind, tuple(sorted(params.items())),
    x0_text, y0_text, v0_text, xmin_text, xmax_text,
)
current_signature = equation_signature(current_values)

if calculate:
    try:
        st.session_state["solution_data"] = solve_problem(
            a_text, b_text, c_text, kind, params, x0_text, y0_text,
            v0_text, xmin_text, xmax_text
        )
        st.session_state["solution_signature"] = current_signature
        st.session_state.pop("solve_error", None)
    except Exception as exc:
        st.session_state["solve_error"] = str(exc)
        st.session_state.pop("solution_data", None)
        st.session_state.pop("solution_signature", None)

if st.session_state.get("solve_error"):
    st.error(f"Impossible de calculer la solution : {st.session_state['solve_error']}")

data = st.session_state.get("solution_data")
if data and st.session_state.get("solution_signature") != current_signature:
    st.info("Les paramètres ont changé. Cliquez sur **Calculer la solution** pour mettre à jour les résultats.")
    data = None

if not data:
    st.markdown("### Bienvenue dans votre espace de résolution")
    intro1, intro2, intro3 = st.columns(3, gap="medium")
    with intro1:
        with st.container(border=True):
            st.markdown("#### ① Définir")
            st.write("Saisissez les coefficients **a**, **b**, **c** et choisissez le second membre **f(x)**.")
    with intro2:
        with st.container(border=True):
            st.markdown("#### ② Résoudre")
            st.write("Ajoutez les conditions initiales, puis lancez le calcul symbolique.")
    with intro3:
        with st.container(border=True):
            st.markdown("#### ③ Explorer")
            st.write("Visualisez la solution, les étapes mathématiques et la vérification.")

if data:
    result = data["result"]
    st.markdown("## Résultats de résolution")
    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        with st.container(border=True):
            st.metric("Constante C₁", sp.sstr(data["c1"]))
    with m2:
        with st.container(border=True):
            st.metric("Constante C₂", sp.sstr(data["c2"]))
    with m3:
        with st.container(border=True):
            st.metric("Vérification", "Réussie ✓" if result["verified"] else "À examiner")

    with st.container(border=True):
        st.markdown("### Solution finale")
        st.latex(r"y(x) = " + sp.latex(data["solution"]))

    tab_graph, tab_steps, tab_check = st.tabs(["📈 Graphique", "∑ Étapes mathématiques", "✓ Vérification"])

    with tab_graph:
        st.markdown("#### Représentation de la solution")
        xx = np.linspace(data["xmin"], data["xmax"], 600)
        try:
            fn = sp.lambdify(x_reel, data["solution"], modules=["numpy"])
            yy = np.asarray(fn(xx))
            if yy.ndim == 0:
                yy = np.full_like(xx, float(yy), dtype=float)
            yy = np.real_if_close(yy)
            if np.iscomplexobj(yy):
                raise ValueError("La solution produit des valeurs complexes sur l’intervalle choisi.")
            yy = np.asarray(yy, dtype=float)
            fig, ax = plt.subplots(figsize=(11, 5.2))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#FBFCFE")
            ax.plot(xx, yy, linewidth=2.8, color="#1769AA", label=r"$y(x)$")
            ax.axhline(0, color="#98A2B3", linewidth=.8)
            ax.axvline(0, color="#98A2B3", linewidth=.8)
            ax.grid(True, alpha=.22, linestyle="--")
            ax.set_xlabel("x", fontsize=11)
            ax.set_ylabel("y(x)", fontsize=11)
            ax.set_title("Solution de l’équation différentielle", loc="left", weight="bold", pad=14)
            ax.legend(frameon=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"La solution a été calculée, mais le graphique n’a pas pu être tracé : {exc}")

    with tab_steps:
        st.markdown("#### Décomposition de la solution")
        with st.container(border=True):
            st.markdown("**1. Second membre**")
            st.latex(r"f(x) = " + sp.latex(data["f"]))
        with st.container(border=True):
            st.markdown("**2. Solution homogène**")
            st.latex(r"y_h(x) = " + sp.latex(result["yh"]))
        with st.container(border=True):
            st.markdown("**3. Solution particulière**")
            st.latex(r"y_p(x) = " + sp.latex(result["yp"]))
        with st.container(border=True):
            st.markdown("**4. Solution générale**")
            st.latex(r"y_g(x) = " + sp.latex(result["general_solution"]))
        with st.container(border=True):
            st.markdown("**5. Application des conditions initiales**")
            st.write(f"Conditions imposées : y({x0_text}) = {y0_text} et y′({x0_text}) = {v0_text}.")
            st.latex(r"y(x) = " + sp.latex(data["solution"]))

    with tab_check:
        st.markdown("#### Vérification mathématique")
        check1, check2 = st.columns(2)
        with check1:
            st.markdown("**Résidu de l’équation**")
            st.latex(sp.latex(result["residual"]))
        with check2:
            st.markdown("**Statut des vérifications**")
            st.write("Équation différentielle :", "✓ Vérifiée" if result["verified"] else "⚠ Non vérifiée")
            st.write(
                "Conditions initiales :",
                "✓ Vérifiées" if result["initial_conditions_verified"] else "⚠ Non vérifiées",
            )
        if result["verified"] and result["initial_conditions_verified"]:
            st.success("La solution satisfait l’équation différentielle et les conditions initiales.")
        else:
            st.warning("Au moins une vérification a échoué. Examinez le résidu et les conditions initiales.")

    with st.expander("Afficher le résumé textuel complet"):
        st.code(
            f"Équation : {sp.sstr(data['a'])}·y''(x) + {sp.sstr(data['b'])}·y'(x) "
            f"+ {sp.sstr(data['c'])}·y(x) = f(x)\n\n"
            f"Second membre : f(x) = {sp.sstr(data['f'])}\n\n"
            f"Solution homogène :\n{sp.sstr(result['yh'])}\n\n"
            f"Solution particulière :\n{sp.sstr(result['yp'])}\n\n"
            f"Solution générale :\ny(x) = {sp.sstr(result['general_solution'])}\n\n"
            f"Solution finale :\ny(x) = {sp.sstr(data['solution'])}\n\n"
            f"Résidu : {sp.sstr(result['residual'])}\n"
            f"Équation vérifiée : {result['verified']}\n"
            f"Conditions initiales vérifiées : {result['initial_conditions_verified']}",
            language="text",
        )
