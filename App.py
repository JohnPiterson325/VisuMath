import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
import tempfile
import os

# ---------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisuMath - Método da Bisseção",
    page_icon="📐",
    layout="wide"
)

# Estilização CSS customizada para cabeçalhos e tabelas
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; color: #1E3A8A; font-weight: bold; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .stTable { font-size: 0.95rem; }
    .highlight-blue { background-color: #E0F2FE; font-weight: bold; }
    .highlight-green { background-color: #DCFCE7; color: #166534; font-weight: bold; }
    .highlight-red { background-color: #FEE2E2; color: #991B1B; font-weight: bold; }
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
    """Converte a string da função em uma função Python operável com Numpy."""
    x = sp.Symbol('x')
    expr_clean = expr_str.replace("^", "**").replace("sen", "sin")
    expr_sympy = sp.sympify(expr_clean)
    f_num = sp.lambdify(x, expr_sympy, modules=["numpy", "math"])
    return f_num, expr_sympy

def check_domain_issues(f, a, b):
    """Verifica se há pontos fora do domínio no intervalo (ex: ln(x) para x <= 0)."""
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
    
    # 1. Botão Restaurar Padrões
    if st.button("🔄 Restaurar Padrões", use_container_width=True):
        st.session_state["func_input"] = "x**2 - 2"
        st.session_state["a_input"] = 1.0
        st.session_state["b_input"] = 2.0
        st.session_state["tol_input"] = 0.0001
        st.session_state["mode_iter"] = "Automático"
        st.session_state["manual_iter"] = 10
        st.session_state["viz_mode"] = "Animação em Vídeo"
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
        options=["Animação em Vídeo", "Slider Interativo Passo a Passo"],
        key="viz_mode"
    )
    
    if viz_mode == "Animação em Vídeo":
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
        - Explorar tabelas de convergência com o rigor acadêmico padrão.
        """)
    with col_info2:
        st.warning("""
        ### 💡 Orientações de Uso
        1. Selecione um **Exemplo Rápido** na barra lateral ou digite sua função.
        2. Defina o intervalo $[a, b]$ inicial garantindo que $f(a) \cdot f(b) < 0$.
        3. Ajuste a tolerância desejada $\varepsilon$ e escolha o modo de iteração.
        4. Clique em **CALCULAR BISSEÇÃO** para gerar a análise gráfica completa.
        """)
        
    st.markdown("---")
    with st.expander("📖 **Fundamentação Teórica & Origem do Método**", expanded=True):
        st.markdown("""
        #### O Teorema do Valor Intermediário (Teorema de Bolzano)
        Se uma função real $f(x)$ for contínua em um intervalo fechado $[a, b]$ e $f(a) \cdot f(b) < 0$, então existe pelo menos um número real $c \in (a, b)$ tal que $f(c) = 0$. Em termos simples: se a função muda de sinal entre os extremos, ela obrigatoriamente cruza o eixo horizontal.

        #### Origem da "Bisseção"
        O termo **Bisseção** vem do latim (*bis* = dois, *sectio* = corte). Trata-se de um método numérico de busca por intervalo que divide sistematicamente o intervalo ativo ao meio a cada iteração ($c = \frac{a+b}{2}$), descartando a metade que não contém a raiz e mantendo a propriedade da troca de sinal.
        """)

else:
    st.session_state["calculated"] = True
    
    try:
        f, sympy_expr = parse_function(func_str)
        fa_test = f(a_val)
        fb_test = f(b_val)
    except Exception as e:
        st.error(f"❌ Erro ao interpretar a função $f(x)$. Verifique a sintaxe matemática. Detalhes: {e}")
        st.stop()
        
    if ("log" in func_str or "ln" in func_str) and (a_val <= 0 or b_val <= 0):
        st.error("⚠️ **Alerta de Domínio:** A função logarítmica só é definida para valores estritamente positivos ($x > 0$). Ajuste o intervalo inicial $[a, b]$ para números maiores que zero.")
        st.stop()
        
    if check_domain_issues(f, a_val, b_val):
        st.error("⚠️ O intervalo selecionado contém valores fora do domínio válido da função ou descontinuidades.")
        st.stop()

    if fa_test * fb_test >= 0:
        st.error(f"""
        ⚠️ **Violação do Teorema de Bolzano:** $f(a) \cdot f(b) \ge 0$.
        - $f(a) = f({fmt(a_val)}) = {fmt(fa_test)}$
        - $f(b) = f({fmt(b_val)}) = {fmt(fb_test)}$
        
        O Método da Bisseção exige que os sinais nas extremidades do intervalo sejam opostos ($f(a) \cdot f(b) < 0$). Ajuste o intervalo inicial.
        """)
        st.stop()

    iterations_data = []
    a, b = a_val, b_val
    target_tol = tol_val
    max_steps = max_iter_user if mode_iter == "Manual" else 40
    
    n_theoretical = int(np.ceil(np.log2(abs(b_val - a_val) / target_tol))) if target_tol > 0 else 0

    stop_reason = ""
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
            stop_reason = "Tolerância atingida com sucesso."
            break
            
        if fa * fc < 0:
            b = c
        else:
            a = c
    else:
        stop_reason = "Número máximo de iterações atingido."

    st.markdown(f"### 📊 Análise Gráfica Dinâmica: ${sp.latex(sympy_expr)}$")
    
    def generate_frame(idx, ax):
        ax.clear()
        data = iterations_data[idx]
        ca, cb, cc = data["a"], data["b"], data["c"]
        cfa, cfb, cfc = data["fa"], data["fb"], data["fc"]
        
        margin = max(abs(cb - ca) * 0.8, 0.05)
        x_min, x_max = ca - margin, cb + margin
        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = f(x_vals)
        
        ax.plot(x_vals, y_vals, color="#1E3A8A", lw=2.5, label="$f(x)$")
        ax.axhline(0, color="black", lw=1, ls="--")
        
        ax.axvspan(ca, cb, color="#E0F2FE", alpha=0.5, label="Intervalo Ativo $[a, b]$")
        
        ax.axvline(ca, color="#DC2626", ls=":", lw=2, label=f"a = {fmt(ca, 4)}")
        ax.axvline(cb, color="#9333EA", ls=":", lw=2, label=f"b = {fmt(cb, 4)}")
        ax.axvline(cc, color="#2563EB", ls="--", lw=2, label=f"c = {fmt(cc, 4)}")
        
        ax.plot(ca, cfa, 'ro', markersize=7)
        ax.plot(cb, cfb, 'mo', markersize=7)
        ax.plot(cc, cfc, 'bo', markersize=8)
        
        ax.text(ca, cfa, f"  f(a) {'+' if cfa>0 else '-'}", color="#DC2626", fontweight="bold")
        ax.text(cb, cfb, f"  f(b) {'+' if cfb>0 else '-'}", color="#9333EA", fontweight="bold")
        ax.text(cc, cfc, f"  f(c) {'+' if cfc>0 else '-'}", color="#2563EB", fontweight="bold")
        
        hud_text = (
            f"Iteração: {data['k']}\n"
            f"a = {fmt(ca, 5)}  |  b = {fmt(cb, 5)}\n"
            f"c = (a+b)/2 = {fmt(cc, 5)}\n"
            f"f(c) = {fmt(cfc, 6)}\n"
            f"Tamanho (b-a) = {fmt(data['width'], 6)}"
        )
        ax.text(0.02, 0.95, hud_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#CBD5E1'),
                fontsize=9, fontfamily='monospace')
                
        ax.set_title(f"Método da Bisseção - Passo {data['k']} de {len(iterations_data)}", fontsize=11, fontweight='bold')
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if viz_mode == "Animação em Vídeo":
        with st.spinner("Renderizando animação interativa..."):
            anim = animation.FuncAnimation(
                fig, 
                generate_frame, 
                frames=len(iterations_data), 
                fargs=(ax,), 
                interval=int(1000 // fps_val)
            )
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                anim.save(tmpfile.name, writer='pillow', fps=fps_val)
                st.image(tmpfile.name)
            plt.close(fig)
    else:
        step_selected = st.slider("Selecione a Iteração:", 1, len(iterations_data), len(iterations_data))
        generate_frame(step_selected - 1, ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("### 📋 Tabela de Bisseções Acumulada")
    
    table_html = f"""
    <table style="width:100%; border-collapse: collapse; text-align: center; font-size: 0.9rem;">
        <thead>
            <tr style="background-color: #1E3A8A; color: white;">
                <th style="padding: 8px;">k</th>
                <th style="padding: 8px;">a</th>
                <th style="padding: 8px;">b</th>
                <th style="padding: 8px; background-color: #0284C7;">c = (a+b)/2</th>
                <th style="padding: 8px;">f(a)</th>
                <th style="padding: 8px;">f(b)</th>
                <th style="padding: 8px; background-color: #0284C7;">f(c)</th>
                <th style="padding: 8px;">|b - a|</th>
                <th style="padding: 8px;">|b - a| &lt; ε ({fmt(target_tol)})</th>
                <th style="padding: 8px;">Substituição</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for row in iterations_data:
        is_tol_met = row["width"] < target_tol
        tol_cmp = f"{fmt(row['width'])} &lt; {fmt(target_tol)}" if is_tol_met else f"{fmt(row['width'])} &gt; {fmt(target_tol)}"
        
        fa_color = "highlight-green" if row["fa"] > 0 else "highlight-red"
        fb_color = "highlight-green" if row["fb"] > 0 else "highlight-red"
        fc_color = "highlight-green" if row["fc"] > 0 else "highlight-red"
        
        table_html += f"""
        <tr style="border-bottom: 1px solid #E2E8F0;">
            <td style="padding: 6px; font-weight: bold;">{row['k']}</td>
            <td style="padding: 6px;">{fmt(row['a'])}</td>
            <td style="padding: 6px;">{fmt(row['b'])}</td>
            <td style="padding: 6px;" class="highlight-blue">{fmt(row['c'])}</td>
            <td style="padding: 6px;" class="{fa_color}">{fmt(row['fa'])}</td>
            <td style="padding: 6px;" class="{fb_color}">{fmt(row['fb'])}</td>
            <td style="padding: 6px;" class="{fc_color}">{fmt(row['fc'])}</td>
            <td style="padding: 6px;">{fmt(row['width'])}</td>
            <td style="padding: 6px;">{tol_cmp}</td>
            <td style="padding: 6px; font-style: italic;">{row['subst']}</td>
        </tr>
        """
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    initial_width = abs(b_val - a_val)
    final_width = iterations_data[-1]["width"]
    reduction_pct = (1.0 - (final_width / initial_width)) * 100.0
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(
        label="📉 Porcentagem de Redução do Intervalo Acumulada",
        value=f"{reduction_pct:.4f}%".replace(".", ","),
        delta=f"De {fmt(initial_width, 4)} para {fmt(final_width, 6)}"
    )

    st.markdown("---")
    st.markdown("### 📝 Relatório & Conclusão Formal")
    
    final_c = iterations_data[-1]["c"]
    final_fc = iterations_data[-1]["fc"]
    total_steps = len(iterations_data)
    
    col_rel1, col_rel2 = st.columns(2)
    with col_rel1:
        st.markdown(f"""
        **Status da Tolerância (ε):** {'✅ Atingida' if final_width < target_tol else '❌ Não Atingida'}\n
        **Iterações Executadas:** {total_steps} passo(s)\n
        **Teto / Estimativa de Iterações:** 
        - Iterações configuradas: **{max_steps}**
        - Iterações estimadas (fórmula teórico-analítica): **{n_theoretical}**
        """)
    
    with col_rel2:
        st.markdown(f"""
        **Raiz Aproximada ($c$):** `{fmt(final_c, 7)}`\n
        **Valor da Função no Ponto Exato $f(c)$:** `{fmt(final_fc, 8)}`\n
        **Amplitude Final do Intervalo:** `{fmt(final_width, 8)}`
        """)
        
    if total_steps >= 20:
        st.warning("⚠️ **Recomendação Científica:** O algoritmo ultrapassou 20 iterações. Devido à convergência linear do Método da Bisseção, recomenda-se considerar métodos de ordem superior (como Newton-Raphson ou Secante) para maior eficiência computacional.")

    if mode_iter == "Manual" and final_width >= target_tol:
        st.info(f"💡 **Nota Pedagógica:** Você limitou a execução a **{max_steps} iterações**, mas eram necessárias teoricamente **{n_theoretical} iterações** para satisfazer a tolerância de {fmt(target_tol)}. Por isso, a tolerância não foi atingida.")

    st.success(f"""
    **Conclusão Formal:**
    Pelo Método da Bisseção, a raiz aproximada da função $f(x) = {sp.latex(sympy_expr)}$ no intervalo inicial $[{fmt(a_val)}, {fmt(b_val)}]$ é **$x \\approx {fmt(final_c, 6)}$**, obtida após **{total_steps} iterações**, garantindo um erro residual de $f(c) = {fmt(final_fc, 8)}$ e precisão de intervalo de ${fmt(final_width, 7)}$.
    """)
