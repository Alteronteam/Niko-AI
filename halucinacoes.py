from keras.models import load_model
import cv2  
import numpy as np
import streamlit as st
import pandas as pd
import time




#coisas do streamlit 
np.set_printoptions(suppress=True)
st.set_page_config(layout="wide") 


try:
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    clean_class_names = [name[2:].strip() for name in class_names]
except Exception as e:
    st.error(f"Erro ao carregar modelo ou rótulos: {e}. Verifique se os arquivos 'keras_Model.h5' e 'labels.txt' estão presentes.")
    st.stop() # Para o script se houver erro vai que

#titulo
st.title("IA para identificar problemas em pastos")

# Cria colunas para a organização
col1, col2 = st.columns([1, 1])
image_placeholder = col1.empty() # Para a imagem processada (Webcam)
text_placeholder = col2.empty() # Para o texto de predição


chart_placeholder = st.empty()



camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if cv2.CAP_DSHOW in locals() else 0)





try:
    while True:

        # 1. tira a foto
        ret, image = camera.read()
        #time.sleep(0.5)

        # 2. pre processamento roubado do cam.py
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        model_input_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
        model_input_image = (model_input_image / 127.5) - 1

        # 3.roubado do cam.py sla o que dfaz mas se tirar explode =/
        prediction = model.predict(model_input_image, verbose=0)
        
        # O array do gráfico
        scores = prediction[0]
        index = np.argmax(scores)
        class_name = clean_class_names[index]
        confidence_score = scores[index]

        # 4. atualiza Gráfico em si
        
        # Crie o DataFrame de comparação de chances
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

        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_placeholder.image(rgb_image, caption="Câmera em Tempo Real", use_container_width='always')
        
        text_placeholder.markdown(
            f"""
            ### Resultado da Classificação
            ---
            o terreno está  <span style='font-size: 24px; color: green;'>**{class_name}**</span>
            **Chance:** **{confidence_score * 100:.2f}%**
            """, unsafe_allow_html=True
            
        )
    

        


finally:
    camera.release()
    cv2.destroyAllWindows()