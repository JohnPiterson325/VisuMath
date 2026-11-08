import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Configuração da página do VisuMath
st.set_page_config(page_title="VisuMath - Método da Bisseção", layout="wide")

st.title("📐 VisuMath: Visualização Interativa do Método da Bisseção")
st.markdown("Ferramenta desenvolvida para o ensino e compreensão de Cálculo Numérico.")

# Barra lateral para os parâmetros de entrada
st.sidebar.header("Parâmetros do Método")
funcao_str = st.sidebar.text_input("Digite a função f(x):", "x**3 - x - 2")
a_input = st.sidebar.number_input("Início do intervalo (a):", value=1.0)
b_input = st.sidebar.number_input("Fim do intervalo (b):", value=2.0)
tolerancia = st.sidebar.selectbox("Tolerância (ε):", [1e-1, 1e-2, 1e-3, 1e-4], index=1)

# Botão para executar o cálculo
if st.button("Executar Bisseção"):
    x = sp.Symbol('x')
    try:
        f_expr = sp.sympify(funcao_str)
        f = sp.lambdify(x, f_expr, 'numpy')
        
        a = float(a_input)
        b = float(b_input)
        
        # Validação inicial do Teorema de Bolzano
        if f(a) * f(b) >= 0:
            st.error("Erro: A função deve ter sinais opostos nos extremos do intervalo [a, b] (Teorema de Bolzano).")
        else:
            iteracoes = []
            fa = f(a)
            fb = f(b)
            
            # Executando o método da bisseção iterativamente
            for i in range(1, 20):
                c = (a + b) / 2
                fc = f(c)
                tamanho = b - a
                
                iteracoes.append({
                    "Iteração": i,
                    "a": round(a, 4),
                    "b": round(b, 4),
                    "c (Ponto Médio)": round(c, 4),
                    "f(a)": "+" if fa > 0 else "-",
                    "f(b)": "+" if fb > 0 else "-",
                    "f(c)": "+" if fc > 0 else "-",
                    "Tamanho (b-a)": round(tamanho, 5)
                })
                
                if abs(tamanho) < tolerancia or abs(fc) < 1e-12:
                    break
                
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc

            st.success(f"Raiz aproximada encontrada: **{c:.4f}** após {len(iteracoes)} iterações.")

            # Layout em duas colunas: Gráfico e Tabela
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.subheader("Gráfico da Função e Intervalo Final")
                fig, ax = plt.subplots(figsize=(6, 4))
                x_vals = np.linspace(a_input - 0.5, b_input + 0.5, 400)
                y_vals = f(x_vals)
                
                ax.plot(x_vals, y_vals, label=f"f(x) = {funcao_str}", color="blue")
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
                ax.axvline(c, color='green', linestyle=':', label=f"Raiz c ≈ {c:.4f}")
                ax.set_xlim(a - 0.2, b + 0.2)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()
                st.pyplot(fig)
                
            with col2:
                st.subheader("Tabela de Iterações Acumuladas")
                st.dataframe(iteracoes)

    except Exception as e:
        st.error(f"Erro ao processar a função. Verifique a sintaxe. Detalhe: {e}")
