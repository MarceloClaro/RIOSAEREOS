import streamlit as st
from PIL import Image
import numpy as np

# Função fictícia para simular previsão de evapotranspiração
def predict_evapotranspiration(image, altura, diametro, copa, lai):
    """
    Simula a previsão de evapotranspiração baseada em uma imagem e dados físicos.
    """
    # Simulação: ajustar essa função para integrar com um modelo real
    evapotranspiracao = (float(altura) * 0.5 + float(diametro) * 0.3 + float(copa) * 0.1 + float(lai) * 0.2) * 10
    return round(evapotranspiracao, 2)

# Configuração do título
st.title("Estimativa de Evapotranspiração por CNN")

# Carregamento da imagem
st.header("Carregar Imagem da Espécie Arbórea ou Arbustiva")
uploaded_image = st.file_uploader("Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagem Carregada", use_column_width=True)

# Entrada de variáveis físicas
st.header("Insira as Variáveis Físicas")
altura = st.text_input("Altura (m):", "0")
diametro = st.text_input("Diâmetro do Tronco (cm):", "0")
copa = st.text_input("Área da Copa (m²):", "0")
lai = st.text_input("LAI (Índice de Área Foliar):", "0")

# Botão para calcular evapotranspiração
if st.button("Calcular Evapotranspiração"):
    if uploaded_image is not None and all([altura, diametro, copa, lai]):
        try:
            # Converte as variáveis para uso no modelo
            altura = float(altura)
            diametro = float(diametro)
            copa = float(copa)
            lai = float(lai)
            
            # Simulação de previsão com base no modelo
            evapotranspiracao = predict_evapotranspiration(image, altura, diametro, copa, lai)
            
            # Exibe o resultado
            st.success(f"Evapotranspiração estimada: {evapotranspiracao} litros/dia")
        except ValueError:
            st.error("Por favor, insira valores numéricos válidos para as variáveis físicas.")
    else:
        st.error("Certifique-se de carregar a imagem e preencher todas as variáveis.")
