import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
import tempfile
import os

# Configuração da página do VisuMath
st.set_page_config(page_title="VisuMath - Animação da Bisseção", layout="wide")

st.title("📐 VisuMath: Máquina de Bisseção Visual")
st.markdown("Ferramenta interativa avançada para o ensino e análise do Método da Bisseção.")

# ==========================================
# BARRA LATERAL (PARÂMETROS)
# ==========================================
st.sidebar.header("⚙️ Parâmetros do Método")

funcao_str = st.sidebar.text_input("Função f(x) (ex: x**3 - x - 2):", "x**3 - x - 2")
a_input = st.sidebar.number_input("Início do intervalo (a):", value=1.0, format="%.4f")
b_input = st.sidebar.number_input("Fim do intervalo (b):", value=2.0, format="%.4f")

epsilon = st.sidebar.selectbox(
    "Precisão / Tolerância (ε):",
    options=[1e-2, 1e-3, 1e-4, 1e-5],
    index=2,
    format_func=lambda x: f"{x}"
)

modo_visualizacao = st.sidebar.radio(
    "Escolha o Modo de Visualização:",
    options=["Vídeo Animado Fluido", "Modo Interativo (Slider por Passo)"]
)

velocidade_fps = st.sidebar.slider("Velocidade da Animação (FPS - Modo 1):", min_value=1, max_value=5, value=2)

if st.button("🚀 Executar VisuMath", type="primary"):
    x_sym = sp.Symbol('x')
    try:
        # Tratamento seguro da expressão matemática
        expr = sp.sympify(funcao_str)
        f = sp.lambdify(x_sym, expr, "numpy")
        
        a_inicial, b_inicial = float(a_input), float(b_input)
        
        if f(a_inicial) * f(b_inicial) > 0:
            st.warning("⚠️ AVISO: f(a) e f(b) possuem o mesmo sinal. O método pode não convergir (Teorema de Bolzano).")

        # ==========================================
        # PRÉ-PROCESSAMENTO DAS ITERAÇÕES
        # ==========================================
        historico = []
        a, b = a_inicial, b_inicial
        limite_loop = 25
        
        for i in range(limite_loop):
            c = (a + b) / 2.0
            fa, fb, fc = f(a), f(b), f(c)
            
            if fa * fc < 0:
                a_next, b_next = a, c
                lado_descartado = 'direito'
            else:
                a_next, b_next = c, b
                lado_descartado = 'esquerdo'
                
            tamanho_atual = abs(b - a)
            
            sinal_fa = "+" if fa > 0 else ("-" if fa < 0 else "0")
            sinal_fb = "+" if fb > 0 else ("-" if fb < 0 else "0")
            sinal_fc = "+" if fc > 0 else ("-" if fc < 0 else "0")
            
            str_precisao = f"{tamanho_atual:.4f} < {epsilon}" if tamanho_atual < epsilon else f"{tamanho_atual:.4f} > {epsilon}"
            
            historico.append({
                'iter': i + 1,
                'a': a,
                'b': b,
                'c': c,
                'sinal_fa': sinal_fa,
                'sinal_fc': sinal_fc,
                'sinal_fb': sinal_fb,
                'precisao_str': str_precisao,
                'a_next': a_next,
                'b_next': b_next,
                'lado_descartado': lado_descartado,
                'tamanho': tamanho_atual,
                'fc': fc
            })
            
            if tamanho_atual < epsilon or abs(fc) < 1e-12:
                break
                
            a, b = a_next, b_next

        num_iteracoes = len(historico)
        st.success(f"✓ Executadas {num_iteracoes} iteração(ões) com sucesso!")

        # ==========================================
        # MODO 1: VÍDEO ANIMADO (GIF)
        # ==========================================
        if modo_visualizacao == "Vídeo Animado Fluido":
            st.subheader("📹 Animação Passo a Passo")
            
            FRAMES_PER_ITER = 20
            total_frames = num_iteracoes * FRAMES_PER_ITER
            
            fig, ax_plot = plt.subplots(figsize=(9, 5), dpi=100)
            x_global = np.linspace(a_inicial - 0.5, b_inicial + 0.5, 400)
            y_global = f(x_global)
            
            # Elementos gráficos dinâmicos para a animação
            line_func, = ax_plot.plot(x_global, y_global, color='#1f77b4', lw=2, label=f"f(x)")
            ax_plot.axhline(0, color='gray', linestyle='--', alpha=0.7)
            
            line_a = ax_plot.axvline(a_inicial, color='red', lw=2, ls='--', label="Limite 'a'")
            line_b = ax_plot.axvline(b_inicial, color='purple', lw=2, ls='--', label="Limite 'b'")
            point_c, = ax_plot.plot([], [], 'o', color='orange', markersize=8, label="Ponto médio c")
            fill_container = [ax_plot.axvspan(a_inicial, b_inicial, color='#17becf', alpha=0.15, label="Intervalo Ativo")]

            def update(frame):
                iter_idx = min(frame // FRAMES_PER_ITER, num_iteracoes - 1)
                d = historico[iter_idx]
                
                line_a.set_xdata([d['a'], d['a']])
                line_b.set_xdata([d['b'], d['b']])
                point_c.set_data([d['c']], [d['fc']])
                
                fill_container[0].remove()
                fill_container[0] = ax_plot.axvspan(d['a'], d['b'], color='orange', alpha=0.2)
                
                ax_plot.set_xlim(d['a'] - 0.5, d['b'] + 0.5)
                ymin, ymax = np.min(y_global), np.max(y_global)
                ax_plot.set_ylim(ymin - 1, ymax + 1)
                ax_plot.set_title(f"Iteração {d['iter']} | Intervalo: [{d['a']:.4f}, {d['b']:.4f}] | c = {d['c']:.4f}", fontsize=10, fontweight='bold')
                return line_a, line_b, point_c, fill_container[0]

            ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/velocidade_fps, repeat=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmpfile:
                ani.save(tmpfile.name, writer='pillow', fps=velocidade_fps)
                gif_path = tmpfile.name
                
            st.image(gif_path, use_container_width=True)
            plt.close(fig)
            os.remove(gif_path)

        # ==========================================
        # MODO 2: SLIDER INTERATIVO POR PASSO
        # ==========================================
        else:
            st.subheader("🎛️ Modo Interativo por Passo")
            passo_selecionado = st.slider("Selecione a Iteração:", min_value=1, max_value=num_iteracoes, value=1)
            
            d = historico[passo_selecionado - 1]
            
            fig, ax_p = plt.subplots(figsize=(9, 4.5), dpi=100)
            x_vals = np.linspace(a_inicial - 0.5, b_inicial + 0.5, 300)
            
            ax_p.plot(x_vals, f(x_vals), color='#1f77b4', lw=2.5, label="f(x)")
            ax_p.axhline(0, color='gray', linestyle='--', alpha=0.7)
            ax_p.axvline(d['a'], color='red', lw=2, ls='--', label="a")
            ax_p.axvline(d['b'], color='purple', lw=2, ls='--', label="b")
            ax_p.plot(d['c'], d['fc'], 'o', color='orange', markersize=9, label="Ponto médio c")
            ax_p.axvspan(d['a'], d['b'], color='#17becf', alpha=0.2, label="Intervalo Ativo")
            
            ax_p.set_xlim(d['a'] - 0.5, d['b'] + 0.5)
            ax_p.set_title(f"Iteração {d['iter']} — Intervalo [{d['a']:.4f}, {d['b']:.4f}]", fontweight='bold')
            ax_p.grid(True, alpha=0.3)
            ax_p.legend(loc='upper right', fontsize=8)
            
            st.pyplot(fig)
            plt.close(fig)

        # ==========================================
        # TABELA DE ITERAÇÕES ACUMULADA
        # ==========================================
        st.subheader("📊 Tabela de Bisseções Acumulada")
        tabela_dados = []
        for d in historico:
            tabela_dados.append({
                "N° Iter.": d['iter'],
                "a": round(d['a'], 4),
                "b": round(d['b'], 4),
                "c": round(d['c'], 4),
                "f(a) sinal": d['sinal_fa'],
                "f(b) sinal": d['sinal_fb'],
                "f(c) sinal": d['sinal_fc'],
                "Precisão (b - a)": d['precisao_str']
            })
        st.dataframe(tabela_dados, use_container_width=True)

        # ==========================================
        # RELATÓRIO FINAL SIMPLIFICADO
        # ==========================================
        ultima = historico[-1]
        st.markdown("---")
        st.subheader("📋 Relatório e Conclusão")
        st.info(f"""
        * **Função Analisada:** $f(x) = {funcao_str}$
        * **Intervalo Final Encontrado:** $[{ultima['a']:.4f}, {ultima['b']:.4f}]$
        * **Raiz Aproximada ($c$):** **{ultima['c']:.4f}**
        * **Total de Iterações:** {len(historico)}
        """)

    except Exception as e:
        st.error(f"Erro ao processar a função ou os dados. Detalhe técnico: {e}")
