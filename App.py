import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
import tempfile
import os

# Configuração da página do VisuMath
st.set_page_config(page_title="VisuMath - Animação da Bisseção", layout="wide")

st.title("📐 VisuMath: Animação do Método da Bisseção")
st.markdown("Veja o método da bisseção em ação com a convergência do intervalo em tempo real.")

# Barra lateral
st.sidebar.header("Parâmetros do Método")
funcao_str = st.sidebar.text_input("Digite a função f(x):", "x**3 - x - 2")
a_input = st.sidebar.number_input("Início do intervalo (a):", value=1.0)
b_input = st.sidebar.number_input("Fim do intervalo (b):", value=2.0)
tolerancia = st.sidebar.select_slider(
    "Tolerância (ε):", 
    options=[1e-1, 1e-2, 1e-3, 1e-4], 
    value=1e-2,
    format_func=lambda x: f"{x:.4f}"
)
fps_input = st.sidebar.slider("Velocidade da Animação (FPS):", min_value=1, max_value=5, value=2)

if st.button("🎬 Gerar Animação do Método", type="primary"):
    x = sp.Symbol('x')
    try:
        f_expr = sp.sympify(funcao_str)
        f = sp.lambdify(x, f_expr, 'numpy')
        
        a_orig, b_orig = float(a_input), float(b_input)
        
        # Validação do Teorema de Bolzano
        if f(a_orig) * f(b_orig) >= 0:
            st.error("Erro: f(a) e f(b) devem ter sinais opostos (Teorema de Bolzano).")
        else:
            # Registrar o histórico de cada passo
            historico = []
            a, b = a_orig, b_orig
            fa, fb = f(a), f(b)
            
            for i in range(1, 25):
                c = (a + b) / 2.0
                fc = f(c)
                tamanho = b - a
                
                historico.append({
                    'iter': i,
                    'a': a,
                    'b': b,
                    'c': c,
                    'fc': fc,
                    'tamanho': tamanho
                })
                
                if abs(tamanho) < tolerancia or abs(fc) < 1e-12:
                    break
                    
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc

            st.success(f"Concluído! Raiz aproximada c ≈ **{historico[-1]['c']:.4f}** obtida em {len(historico)} iterações.")

            # Layout: Vídeo da Animação à esquerda, Tabela à direita
            col_video, col_tabela = st.columns([1.3, 1])
            
            with col_video:
                st.subheader("📹 Animação Passo a Passo")
                
                # Configurando a figura do Matplotlib
                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
                
                x_vals = np.linspace(a_orig - 0.5, b_orig + 0.5, 400)
                y_vals = f(x_vals)
                
                # Elementos gráficos fixos e móveis
                line_func, = ax.plot(x_vals, y_vals, label=f"f(x) = {funcao_str}", color="#1f77b4", linewidth=2)
                ax.axhline(0, color='black', linewidth=1, linestyle='--')
                
                # Usando lista para armazenar o objeto do preenchimento de forma mutável
                fill_container = [ax.axvspan(a_orig, b_orig, color='orange', alpha=0.2, label="Intervalo [a, b]")]
                line_a = ax.axvline(a_orig, color='red', linestyle='--', linewidth=1.5, label="a")
                line_b = ax.axvline(b_orig, color='blue', linestyle='--', linewidth=1.5, label="b")
                point_c, = ax.plot([], [], 'go', markersize=8, label="Ponto médio (c)")
                
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.legend(loc="upper left", fontsize="small")

                # Função de atualização frame por frame
                def update(frame):
                    d = historico[frame]
                    
                    # Atualiza os limites visuais das linhas verticais
                    line_a.set_xdata([d['a'], d['a']])
                    line_b.set_xdata([d['b'], d['b']])
                    point_c.set_data([d['c']], [d['fc']])
                    
                    # Remove o preenchimento antigo e desenha o novo intervalo atualizado
                    fill_container[0].remove()
                    fill_container[0] = ax.axvspan(d['a'], d['b'], color='orange', alpha=0.25)
                    
                    ax.set_title(f"Iteração {d['iter']}: Intervalo = [{d['a']:.4f}, {d['b']:.4f}] | c = {d['c']:.4f}", fontsize=11, fontweight='bold')
                    return line_a, line_b, point_c, fill_container[0]

                # Criando a animação com Matplotlib
                ani = animation.FuncAnimation(
                    fig, update, frames=len(historico), interval=1000/fps_input, repeat=True
                )
                
                # Salvando como GIF animado temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmpfile:
                    ani.save(tmpfile.name, writer='pillow', fps=fps_input)
                    gif_path = tmpfile.name
                
                # Exibindo o GIF no Streamlit
                st.image(gif_path, caption="Animação contínua da busca da raiz", use_container_width=True)
                plt.close(fig)
                
                # Apagando o arquivo temporário
                os.remove(gif_path)

            with col_tabela:
                st.subheader("📊 Tabela de Iterações")
                tabela_dados = []
                for d in historico:
                    tabela_dados.append({
                        "Iter.": d['iter'],
                        "a": round(d['a'], 4),
                        "b": round(d['b'], 4),
                        "c": round(d['c'], 4),
                        "f(c)": f"{d['fc']:.4e}",
                        "b - a": round(d['tamanho'], 4)
                    })
                st.dataframe(tabela_dados, height=380)

    except Exception as e:
        st.error(f"Sintaxe inválida ou erro na função. Detalhe: {e}")
