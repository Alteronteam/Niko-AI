import streamlit as st
import pandas as pd
import time
import numpy as np # Usado apenas para simular a leitura do sensor

st.title("૮ ˶ᵔ ᵕ ᵔ˶ ა")

# Cria um placeholder vazio onde o gráfico será atualizado, tipo onde ele vai caber pq é meio coisas
chart_placeholder = st.empty()


# Definimos o número total de iterações para a demonstração
MAX_ITERATIONS = 50 

for i in range(1, MAX_ITERATIONS + 1):
    
  
    foguinho = np.random.uniform(5.0, 30.0) 
    prantinha = np.random.uniform(5.0, 30.0) 
   

    
    data_to_plot = pd.DataFrame({
        'Categoria': ['prantinha', 'ˆ𐃷ˆ'], 
        'Valor Atual': [foguinho, prantinha] 
    })
    
    # B. Atualiza o placeholder com o novo gráfico e informações
    with chart_placeholder:
        # Exibe os valores atuais
        st.markdown(
            f"**Atualização {i}** | **area devastada** `{foguinho:.2f}` | **area coberta** `{prantinha:.2f}`"
        )
        
        # O bar_chart utiliza as colunas 'Categoria' (eixo X) e 'Valor Atual' (eixo Y)
        # O gráfico será redesenhado do zero a cada iteração
        st.bar_chart(
            data_to_plot,
            x='Categoria',
            y='Valor Atual',
            use_container_width=True
        )
        
    # Espera 1 segundo (ajuste conforme a frequência de atualização desejada)
    time.sleep(1)

st.success("vacas exterminadoras premium para mais tempo")