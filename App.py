import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
import tempfile
import pandas as pd

# ---------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisuMath - Método da Bisseção",
    page_icon="📐",
    layout="wide"
)

# Estilização CSS limpa
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; color: #1E3A8A; font-weight: bold; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FUNÇÕES AUXILIARES DE FORMATAÇÃO E CÁLCULO
# ---------------------------------------------------------
def fmt(val, dec=6):
    """Formata valores numéricos para o padrão brasileiro (vírgula)."""
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        return f"{val:.{dec}f}".replace(".", ",")
    return str(val)

def parse_function(expr_str):
    x = sp.Symbol('x')
    expr_clean = expr_str.replace("^", "**").replace("sen", "sin")
    expr_sympy = sp.sympify(expr_clean)
    f_num = sp.lambdify(x, expr_sympy, modules=["numpy", "math"])
    return f_num, expr_sympy

def check_domain_issues(f, a, b):
    test_points = np.linspace(a, b, 100)
    try:
        vals = f(test_points)
        if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
            return True
    except Exception:
        return True
    return False

# ---------------------------------------------------------
# BARRA LATERAL (1/4 DA TELA)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações VisuMath")
    
    if st.button("🔄 Restaurar Padrões", use_container_width=True):
        st.session_state["func_input"] = "x**2 - 2"
        st.session_state["a_input"] = 1.0
        st.session_state["b_input"] = 2.0
        st.session_state["tol_input"] = 0.0001
        st.session_state["mode_iter"] = "Automático"
        st.session_state["manual_iter"] = 10
        st.session_state["viz_mode"] = "Slider Interativo (Estilo Colab)"
        st.rerun()

    st.markdown("---")
    st.subheader("🚀 Exemplos Rápidos (1-Clique)")
    
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("1. Cúbica", use_container_width=True):
            st.session_state["func_input"] = "x**3 - x - 2"
            st.session_state["a_input"] = 1.0
            st.session_state["b_input"] = 2.0
        if st.button("3. Logarítmica", use_container_width=True):
            st.session_state["func_input"] = "log(x) - 1"
            st.session_state["a_input"] = 2.0
            st.session_state["b_input"] = 3.0
        if st.button("5. Exponencial", use_container_width=True):
            st.session_state["func_input"] = "exp(x) - 3*x"
            st.session_state["a_input"] = 1.0
            st.session_state["b_input"] = 2.0

    with col_ex2:
        if st.button("2. Parábola", use_container_width=True):
            st.session_state["func_input"] = "x**2 - 4"
            st.session_state["a_input"] = 0.0
            st.session_state["b_input"] = 3.0
        if st.button("4. Trigonométrica", use_container_width=True):
            st.session_state["func_input"] = "sin(x) - 0.5"
            st.session_state["a_input"] = 0.0
            st.session_state["b_input"] = 1.5

    st.markdown("---")
    st.subheader("📥 Parâmetros de Entrada")
    
    func_str = st.text_input(
        "Função f(x):", 
        value=st.session_state.get("func_input", "x**2 - 2"),
        key="func_input"
    )
    
    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        a_val = st.number_input("Início (a):", value=float(st.session_state.get("a_input", 1.0)), format="%.4f", key="a_input")
    with col_ab2:
        b_val = st.number_input("Fim (b):", value=float(st.session_state.get("b_input", 2.0)), format="%.4f", key="b_input")
        
    tol_val = st.select_slider(
        "Tolerância (ε):",
        options=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        value=st.session_state.get("tol_input", 0.0001),
        format_func=lambda x: f"{x:.7f}".rstrip('0').rstrip('.').replace(".", ","),
        key="tol_input"
    )
    
    mode_iter = st.radio(
        "Limite de Iterações:",
        options=["Automático", "Manual"],
        index=0 if st.session_state.get("mode_iter", "Automático") == "Automático" else 1,
        key="mode_iter"
    )
    
    if mode_iter == "Manual":
        max_iter_user = st.slider(
            "Máximo de passos:", 
            min_value=1, 
            max_value=40, 
            value=int(st.session_state.get("manual_iter", 10)),
            key="manual_iter"
        )
    else:
        max_iter_user = 40
        
    viz_mode = st.radio(
        "Modo de Visualização:",
        options=["Slider Interativo (Estilo Colab)", "Animação em GIF"],
        key="viz_mode"
    )
    
    if viz_mode == "Animação em GIF":
        fps_val = st.slider("Velocidade (FPS):", min_value=1, max_value=5, value=2)

    btn_calcular = st.button("🚀 CALCULAR BISSEÇÃO", type="primary", use_container_width=True)

# ---------------------------------------------------------
# ÁREA PRINCIPAL (3/4 DA TELA)
# ---------------------------------------------------------
if not btn_calcular and "calculated" not in st.session_state:
    st.markdown('<div class="main-title">📐 VisuMath - Método da Bisseção</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Plataforma Interativa para Análise Numérica e Visualização do Teorema de Bolzano</div>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info("""
        ### 🎯 Objetivos
        - Visualizar dinamicamente a convergência do Método da Bisseção.
        - Analisar o comportamento do intervalo $[a, b]$ e do ponto médio $c$ a cada iteração.
        - Explorar tabelas de convergência com rigor acadêmico.
        """)
    with col_info2:
        st.warning("""
        ### 💡 Orientações de Uso
        1. Selecione um **Exemplo Rápido** na barra lateral ou digite sua função.
        2. Defina o intervalo $[a, b]$ inicial garantindo $f(a) \cdot f(b) < 0$.
        3. Ajuste a tolerância e clique em **CALCULAR BISSEÇÃO**.
        """)
        
    st.markdown("---")
    with st.expander("📖 **Fundamentação Teórica & Origem do Método**", expanded=True):
        st.markdown("""
        #### O Teorema do Valor Intermediário (Teorema de Bolzano)
        Se uma função real $f(x)$ for contínua em $[a, b]$ e $f(a) \cdot f(b) < 0$, existe pelo menos um $c \in (a, b)$ tal que $f(c) = 0$.
        """)

else:
    st.session_state["calculated"] = True
    
    try:
        f, sympy_expr = parse_function(func_str)
        fa_test = f(a_val)
        fb_test = f(b_val)
    except Exception as e:
        st.error(f"❌ Erro ao interpretar a função $f(x)$. Detalhes: {e}")
        st.stop()
        
    if ("log" in func_str or "ln" in func_str) and (a_val <= 0 or b_val <= 0):
        st.error("⚠️ **Alerta de Domínio:** A função logarítmica exige $x > 0$. Ajuste o intervalo inicial.")
        st.stop()
        
    if check_domain_issues(f, a_val, b_val):
        st.error("⚠️ O intervalo contém valores fora do domínio válido da função.")
        st.stop()

    if fa_test * fb_test >= 0:
        st.error(f"⚠️ **Violação de Bolzano:** $f(a) \cdot f(b) \ge 0$. Os sinais nas extremidades devem ser opostos.")
        st.stop()

    # Execução do Algoritmo
    iterations_data = []
    a, b = a_val, b_val
    target_tol = tol_val
    max_steps = max_iter_user if mode_iter == "Manual" else 40
    
    for k in range(1, max_steps + 1):
        c = (a + b) / 2.0
        fa, fb, fc = f(a), f(b), f(c)
        width = abs(b - a)
        
        subst = "a ← c" if fa * fc > 0 else "b ← c"
        iterations_data.append({
            "k": k, "a": a, "b": b, "c": c,
            "fa": fa, "fb": fb, "fc": fc,
            "width": width, "subst": subst
        })
        
        if width < target_tol or abs(fc) < 1e-12:
            break
            
        if fa * fc < 0:
            b = c
        else:
            a = c

    # ---------------------------------------------------------
    # VISUALIZAÇÃO GRÁFICA AVANÇADA (PADRÃO COLAB)
    # ---------------------------------------------------------
    st.markdown(f"### 📊 Análise Gráfica Dinâmica (Estilo Colab): ${sp.latex(sympy_expr)}$")
    
    def render_colab_style_plot(idx):
        fig, ax = plt.subplots(figsize=(10, 5))
        data = iterations_data[idx]
        ca, cb, cc = data["a"], data["b"], data["c"]
        cfa, cfb, cfc = data["fa"], data["fb"], data["fc"]
        
        # Domínio adaptativo suave ao redor do intervalo ativo
        span = max(abs(cb - ca), 0.1)
        x_vals = np.linspace(ca - span * 1.5, cb + span * 1.5, 500)
        y_vals = f(x_vals)
        
        # Curva principal
        ax.plot(x_vals, y_vals, color="#1E3A8A", lw=2.5, label="$f(x)$")
        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
        
        # Preenchimento translúcido da área de busca ativa [a, b]
        ax.axvspan(ca, cb, color="#3B82F6", alpha=0.15, label="Intervalo de Busca $[a, b]$")
        
        # Linhas verticais limpas
        ax.axvline(ca, color="#EF4444", ls=":", lw=1.8, label=f"a = {fmt(ca, 4)}")
        ax.axvline(cb, color="#8B5CF6", ls=":", lw=1.8, label=f"b = {fmt(cb, 4)}")
        ax.axvline(cc, color="#0EA5E9", ls="--", lw=2, label=f"c (Ponto Médio) = {fmt(cc, 4)}")
        
        # Pontos exatos sobre a curva
        ax.plot(ca, cfa, 'o', color="#EF4444", markersize=8)
        ax.plot(cb, cfb, 'o', color="#8B5CF6", markersize=8)
        ax.plot(cc, cfc, 'o', color="#0EA5E9", markersize=10, zorder=5)
        
        # Anotações de texto no gráfico
        ax.annotate(f"f(a) {'(+)' if cfa>0 else '(-)'}", (ca, cfa), textcoords="offset points", xytext=(-15, 10), ha='center', fontweight='bold', color="#EF4444")
        ax.annotate(f"f(b) {'(+)' if cfb>0 else '(-)'}", (cb, cfb), textcoords="offset points", xytext=(15, 10), ha='center', fontweight='bold', color="#8B5CF6")
        ax.annotate(f"f(c) {'(+)' if cfc>0 else '(-)'}", (cc, cfc), textcoords="offset points", xytext=(0, 15), ha='center', fontweight='bold', color="#0EA5E9")

        # HUD flutuante interno estilo painel de laboratório
        hud = (
            f"Passo: {data['k']} | Intervalo: [{fmt(ca, 4)}, {fmt(cb, 4)}]\n"
            f"c = {fmt(cc, 5)} | f(c) = {fmt(cfc, 6)}\n"
            f"Amplitude (b-a) = {fmt(data['width'], 6)}"
        )
        ax.text(0.03, 0.93, hud, transform=ax.transAxes, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#94A3B8', alpha=0.9))

        ax.set_title(f"Iteração {data['k']} — Convergência do Método da Bisseção", fontsize=12, fontweight='bold', pad=12)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.5)
        return fig

    if viz_mode == "Slider Interativo (Estilo Colab)":
        step_sel = st.slider("🔍 Deslize para acompanhar a evolução passo a passo:", 1, len(iterations_data), len(iterations_data))
        fig_plot = render_colab_style_plot(step_sel - 1)
        st.pyplot(fig_plot)
        plt.close(fig_plot)
    else:
        with st.spinner("Gerando animação GIF..."):
            fig, ax = plt.subplots(figsize=(10, 5))
            def update(i):
                ax.clear()
                # Reutiliza a lógica de frame para o gif
                data = iterations_data[i]
                ca, cb, cc = data["a"], data["b"], data["c"]
                cfa, cfb, cfc = data["fa"], data["fb"], data["fc"]
                span = max(abs(cb - ca), 0.1)
                x_vals = np.linspace(ca - span * 1.5, cb + span * 1.5, 400)
                ax.plot(x_vals, f(x_vals), color="#1E3A8A", lw=2)
                ax.axhline(0, color="black", ls="--")
                ax.axvspan(ca, cb, color="#3B82F6", alpha=0.15)
                ax.axvline(ca, color="#EF4444", ls=":")
                ax.axvline(cb, color="#8B5CF6", ls=":")
                ax.axvline(cc, color="#0EA5E9", ls="--")
                ax.plot(cc, cfc, 'bo', markersize=8)
                ax.set_title(f"Iteração {data['k']}")
                ax.grid(True, alpha=0.5)

            anim = animation.FuncAnimation(fig, update, frames=len(iterations_data), interval=1000//fps_val)
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                anim.save(tmp.name, writer='pillow', fps=fps_val)
                st.image(tmp.name)
            plt.close(fig)

    # ---------------------------------------------------------
    # TABELA DE BISSEÇÕES ACUMULADA (CORRIGIDA E SEM BUGS)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Tabela de Bisseções Acumulada")
    
    table_rows = []
    for row in iterations_data:
        is_tol_met = row["width"] < target_tol
        comp_str = f"{fmt(row['width'])} < {fmt(target_tol)}" if is_tol_met else f"{fmt(row['width'])} > {fmt(target_tol)}"
        table_rows.append({
            "k": row["k"],
            "a": fmt(row["a"]),
            "b": fmt(row["b"]),
            "c = (a+b)/2": fmt(row["c"]),
            "f(a)": fmt(row["fa"]),
            "f(b)": fmt(row["fb"]),
            "f(c)": fmt(row["fc"]),
            "|b - a|": fmt(row["width"]),
            "Critério ε": comp_str,
            "Substituição": row["subst"]
        })
    
    df_results = pd.DataFrame(table_rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Indicador de Redução
    initial_width = abs(b_val - a_val)
    final_width = iterations_data[-1]["width"]
    reduction_pct = (1.0 - (final_width / initial_width)) * 100.0
    
    st.metric(
        label="📉 Porcentagem de Redução do Intervalo Acumulada",
        value=f"{reduction_pct:.4f}%".replace(".", ","),
        delta=f"De {fmt(initial_width, 4)} para {fmt(final_width, 6)}"
    )

    # ---------------------------------------------------------
    # RELATÓRIO & CONCLUSÃO FORMAL (LIMPO E SEM POLUIÇÃO)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📝 Relatório & Conclusão Formal")
    
    final_c = iterations_data[-1]["c"]
    final_fc = iterations_data[-1]["fc"]
    total_steps = len(iterations_data)
    
    col_rel1, col_rel2 = st.columns(2)
    with col_rel1:
        st.markdown(f"""
        - **Status da Tolerância ($\varepsilon$):** {'✅ Atingida' if final_width < target_tol else '❌ Não Atingida'}
        - **Iterações Executadas:** **{total_steps}**
        """)
    with col_rel2:
        st.markdown(f"""
        - **Raiz Aproximada ($c$):** `{fmt(final_c, 7)}`
        - **Erro Residual $f(c)$:** `{fmt(final_fc, 8)}`
        - **Amplitude Final:** `{fmt(final_width, 8)}`
        """)
        
    if total_steps >= 20:
        st.warning("⚠️ **Nota:** O algoritmo ultrapassou 20 iterações, indicando que métodos abertos poderiam convergir com maior velocidade.")

    st.success(f"""
    **Conclusão Formal:**
    Pelo Método da Bisseção, a raiz aproximada da função $f(x) = {sp.latex(sympy_expr)}$ no intervalo inicial $[{fmt(a_val)}, {fmt(b_val)}]$ é **$x \\approx {fmt(final_c, 6)}$**, obtida após **{total_steps} iterações**, garantindo um erro residual de $f(c) = {fmt(final_fc, 8)}$ e precisão de intervalo de ${fmt(final_width, 7)}$.
    """)
