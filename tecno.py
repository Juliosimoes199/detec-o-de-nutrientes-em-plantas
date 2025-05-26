import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Configurações da página
st.set_page_config(
    page_title="Detecção de Deficiência em Plantações",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Barra lateral para configurações e informações
with st.sidebar:
    st.title("⚙️ Configurações")
    st.markdown("Aplicação para detecção de deficiências de nutrientes em cultivos usando Visão Computacional.")
    st.markdown("---")
    st.subheader("Nutrientes capazes de detectar")
    st.markdown("Cálcio")
    st.markdown("Magnésio")
    st.markdown("Nitrogénio")
    st.markdown("Fósforo")
    st.markdown("Potássio")
    st.markdown("---")
    st.subheader("Instruções")
    st.markdown("1. Faça o upload de uma imagem clara de um cultivo.")
    st.markdown("2. Clique no botão 'Detectar Deficiências'.")
    st.markdown("3. Os resultados da detecção serão exibidos abaixo.")
    st.markdown("---")
    st.subheader("Sobre o Modelo")
    st.markdown("Modelo YOLOv11 treinado para identificar deficiências de nutrientes (NPK) em plantações.")
    st.markdown("---")
    st.info("Desenvolvido pela Tecno Society.")

# Conteúdo principal
st.title("🥬 Detecção de Deficiência de Nutrientes em Cultivos de Alface")
st.markdown("---")

# Carregue o modelo treinado (isso será feito apenas uma vez)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Widget para fazer o upload da imagem
uploaded_file = st.file_uploader("Faça o upload de uma imagem de um cultivo...", type=["jpg", "jpeg", "png"])

st.markdown("---")

if uploaded_file is not None:
    # Layout em colunas para a imagem carregada e o botão
    col1, col2 = st.columns([1, 1])

    with col1:
        try:
            # Exibe a imagem carregada
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem Carregada", use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao abrir a imagem: {e}")

    with col2:
        detect_button = st.button("Detectar Deficiências")

    # Botão para realizar a detecção (centralizado abaixo da imagem)
    if detect_button:
        if uploaded_file is not None:
            with st.spinner("Analisando a imagem..."):
                try:
                    # Realiza as previsões na imagem carregada
                    image = Image.open(uploaded_file)
                    results = model.predict(image)

                    st.subheader("Resultados da Detecção:")
                    if results and results[0].boxes:
                        result = results[0]
                        boxes = result.boxes
                        classes = boxes.cls
                        confidences = boxes.conf
                        xyxy = boxes.xyxy

                        st.write("Classes detectadas:", classes.cpu().numpy())
                        st.write("Pontuações de confiança:", confidences.cpu().numpy())
                        st.write("Coordenadas das caixas delimitadoras:", xyxy.cpu().numpy())

                        # Exibe a imagem com as caixas delimitadoras
                        annotated_image = results[0].plot()
                        st.image(annotated_image, caption="Imagem com Detecções", use_container_width=True)
                    else:
                        st.info("Nenhuma deficiência de nutriente detectada nesta imagem.")
                except Exception as e:
                    st.error(f"Erro durante a detecção: {e}")
            st.markdown("---")
