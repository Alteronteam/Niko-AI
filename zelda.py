#oh yess baby
import streamlit as st
import pandas as pd
import numpy as np
import time

# Título da aplicação
st.title("Gráfico de Barras Dinâmico com Streamlit e Pandas")

# Cria um placeholder (contêiner vazio) para o gráfico
chart_placeholder = st.empty()

# Simulação de atualização em tempo real
for i in range(100):
    # Gera dados aleatórios usando NumPy e Pandas
    # Cria um DataFrame com 3 colunas ('a', 'b', 'c') e dados aleatórios
    # O shape dos dados muda ligeiramente a cada iteração para mostrar a dinamicidade
    data = pd.DataFrame(
        np.random.randn(10, 3), # Gera 10 linhas e 3 colunas de números aleatórios
        columns=['a', 'b', 'c']
    )
    
    # Atualiza o placeholder com o novo gráfico de barras
    with chart_placeholder:
        st.markdown(f"**Iteração {i+1}**")
        st.bar_chart(data)
        
    # Espera 0.5 segundos antes da próxima atualização para simular o tempo real
    time.sleep(1)

st.success("Demonstração concluída!")