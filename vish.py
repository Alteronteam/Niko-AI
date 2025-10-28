from keras.models import load_model
import cv2  
import numpy as np
import streamlit as st
import pandas as pd
import time
# Não precisamos mais do time aqui, mas vamos manter o import caso queira usar.


# --- Configurações Iniciais ---
np.set_printoptions(suppress=True)
st.set_page_config(layout="wide") 

# Carrega Modelo e Rótulos
# Verifique se os arquivos .h5 e .txt estão no mesmo diretório
try:
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    clean_class_names = [name[2:].strip() for name in class_names]
except Exception as e:
    st.error(f"Erro ao carregar modelo ou rótulos: {e}. Verifique se os arquivos 'keras_Model.h5' e 'labels.txt' estão presentes.")
    st.stop() # Para o script se houver erro de arquivo

# --- Streamlit UI ---
st.title("quiemador de gpu")

# Cria colunas para a organização
col1, col2 = st.columns([1, 1])
image_placeholder = col1.empty() # Para a imagem processada (Webcam)
text_placeholder = col2.empty() # Para o texto de predição

# Placeholder para o gráfico de barras
chart_placeholder = st.empty()


# --- Configuração da Câmera ---
# Tente diferentes índices (0, 1, -1) se a câmera padrão (0) não funcionar.
# Adicione cv2.CAP_DSHOW no Windows para maior estabilidade.
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if cv2.CAP_DSHOW in locals() else 0)

if not camera.isOpened():
    st.error("Erro: Não foi possível abrir a câmera. Tente outro índice (ex: cv2.VideoCapture(1)).")
    st.stop()


# --- LOOP PRINCIPAL DE PROCESSAMENTO ---

# Use um contador para limitar o número de loops se for necessário, ou deixe 'while True'
try:
    while True:

        # 1. CAPTURA
        ret, image = camera.read()
        time.sleep(0.5)

        # 2. PRÉ-PROCESSAMENTO (Conforme o seu código)
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        model_input_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
        model_input_image = (model_input_image / 127.5) - 1

        # 3. PREDIÇÃO
        prediction = model.predict(model_input_image, verbose=0)
        
        # O array de predições é o que usaremos para o gráfico
        scores = prediction[0]
        index = np.argmax(scores)
        class_name = clean_class_names[index]
        confidence_score = scores[index]

        # 4. ATUALIZAÇÃO DO STREAMLIT (Gráfico)
        
        # Crie o DataFrame de comparação de Confiança
        data_to_plot = pd.DataFrame({
            'Classe': clean_class_names, 
            'Confiança (%)': scores * 100
        }).sort_values(by='Confiança (%)', ascending=False)
        
        with chart_placeholder:
            st.subheader("Confiança de Todas as Classes")
            st.bar_chart(
                data_to_plot,
                x='Classe',
                y='Confiança (%)',
                use_container_width=True
            )

        # 5. ATUALIZAÇÃO DO STREAMLIT (Imagem e Texto)

        # Remove o cv2.cvtColor se a imagem estiver com a cor correta, mas BGR para RGB é padrão
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_placeholder.image(rgb_image, caption="Câmera em Tempo Real", use_column_width='always')
        
        text_placeholder.markdown(
            f"""
            ### Resultado da Classificação
            ---
            **Classe Predita:** <span style='font-size: 24px; color: green;'>**{class_name}**</span>
            **Confiança:** **{confidence_score * 100:.2f}%**
            """, unsafe_allow_html=True
        )

        # Não usamos cv2.waitKey() ou input de teclado em Streamlit
        # O loop continua rodando o mais rápido possível

# Bloco final (finally) para garantir que a câmera seja liberada
finally:
    camera.release()
    cv2.destroyAllWindows()