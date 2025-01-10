try:
    import streamlit as st
    from PIL import Image
    import numpy as np
except ModuleNotFoundError as e:
    print("Streamlit or PIL is not installed. Ensure the correct environment is used or install the required packages.")
    raise e

# Função para calcular a área foliar total
def calculate_area_foliar_total(folhas_data, galhos):
    """
    Calcula a área foliar total com base nas dimensões das folhas e no número de galhos.
    """
    total_area = 0
    for largura, comprimento in folhas_data:
        try:
            largura = float(largura)
            comprimento = float(comprimento)
            total_area += (largura * comprimento) * galhos  # Área de uma folha * número de galhos
        except ValueError:
            continue
    return total_area

# Ajuste no cálculo do LAI para usar a área foliar total
def calculate_lai(area_foliar_total, area_copa):
    """
    Calcula o LAI com base na área foliar total e na área da copa.
    """
    try:
        lai = area_foliar_total / float(area_copa)  # LAI baseado na área foliar total
        return round(lai, 2)
    except ZeroDivisionError:
        return 0.0

# Função fictícia para simular previsão de evapotranspiração
def predict_evapotranspiration(image, altura, diametro, copa, lai):
    """
    Simula a previsão de evapotranspiração baseada em uma imagem e dados físicos.
    """
    evapotranspiracao = (float(altura) * 0.5 + float(diametro) * 0.3 + float(copa) * 0.1 + float(lai) * 0.2) * 10
    return round(evapotranspiracao, 2)

# Configuração do título
st.title("Estimativa de Evapotranspiração por CNN")

# Carregamento da imagem
st.header("Carregar Imagem da Espécie Arbórea ou Arbustiva")
uploaded_image = st.file_uploader("Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagem Carregada", use_container_width=True)

# Entrada de variáveis físicas
st.header("Insira as Variáveis Físicas")
num_especies = st.number_input("Quantidade de Espécimes:", min_value=1, step=1, value=1)

especies_data = []
for i in range(num_especies):
    st.subheader(f"Espécime {i + 1}")
    altura = st.text_input(f"Altura (m) - Espécime {i + 1}:", "0")
    diametro = st.text_input(f"Diâmetro do Tronco (cm) - Espécime {i + 1}:", "0")
    copa = st.text_input(f"Área da Copa (m²) - Espécime {i + 1}:", "0")
    galhos = st.number_input(f"Quantidade de Galhos - Espécime {i + 1}:", min_value=1, step=1, value=1)

    folhas_data = []
    for j in range(galhos):
        st.text(f"Dimensão Foliar por Galho {j + 1} - Espécime {i + 1}")
        largura_folha = st.text_input(f"Largura da Folha (cm) - Galho {j + 1} - Espécime {i + 1}:", "0")
        comprimento_folha = st.text_input(f"Comprimento da Folha (cm) - Galho {j + 1} - Espécime {i + 1}:", "0")
        folhas_data.append((largura_folha, comprimento_folha))

    # Cálculo da área foliar total
    area_foliar_total = calculate_area_foliar_total(folhas_data, galhos)

    # Recalcular o LAI com base na área foliar total
    if area_foliar_total > 0 and copa != "0":
        lai = calculate_lai(area_foliar_total, copa)
        st.text(f"LAI Calculado para o Espécime {i + 1}: {lai}")
    else:
        lai = "0"

    especies_data.append((altura, diametro, copa, lai, galhos, folhas_data))

# Botão para calcular evapotranspiração
if st.button("Calcular Evapotranspiração"):
    if uploaded_image is not None and all(all(var != "0" for var in especie[:4]) for especie in especies_data):
        try:
            resultados = []
            for i, (altura, diametro, copa, lai, galhos, folhas_data) in enumerate(especies_data):
                altura = float(altura)
                diametro = float(diametro)
                copa = float(copa)
                lai = float(lai)

                # Simulação de previsão com base no modelo
                evapotranspiracao = predict_evapotranspiration(image, altura, diametro, copa, lai)
                resultados.append(f"Espécime {i + 1}: {evapotranspiracao} litros/dia")

            # Exibe os resultados
            for resultado in resultados:
                st.success(resultado)
        except ValueError:
            st.error("Por favor, insira valores numéricos válidos para todas as variáveis físicas.")
    else:
        st.error("Certifique-se de carregar a imagem e preencher todas as variáveis para cada espécime.")
