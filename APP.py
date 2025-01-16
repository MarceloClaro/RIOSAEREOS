import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------------
# 1. Armazenamento em session_state para persistência
# ---------------------------------------------------------------
if "resultados" not in st.session_state:
    st.session_state.resultados = []  # Evapotranspirações (modelo)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ---------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------
def calculate_area_foliar_total(folhas_data, galhos):
    """
    Calcula a área foliar total com base nas dimensões (largura x comprimento)
    de cada folha e no número de galhos do espécime.
    """
    total_area = 0.0
    for largura_str, comprimento_str in folhas_data:
        try:
            largura = float(largura_str)
            comprimento = float(comprimento_str)
            total_area += (largura * comprimento) * galhos
        except ValueError:
            # Se houver erro na conversão, ignora aquela folha
            continue
    return total_area

def calculate_lai(area_foliar_total, area_copa):
    """
    Calcula o LAI (Leaf Area Index) = Área Foliar Total / Área da Copa.
    Retorna 0.0 em caso de erro ou divisão por zero.
    """
    try:
        area_copa_val = float(area_copa)
        lai = area_foliar_total / area_copa_val
        return round(lai, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0

def predict_evapotranspiration(image, altura, diametro, copa, lai):
    """
    Simula a previsão de evapotranspiração (litros/dia) usando coeficientes
    ilustrativos. A imagem pode futuramente ser usada em um modelo de CNN real.
    """
    et = (altura * 0.5 + diametro * 0.3 + copa * 0.1 + lai * 0.2) * 10
    return round(et, 2)

# ---------------------------------------------------------------
# 3. Cabeçalho e título
# ---------------------------------------------------------------
st.title("Estimativa de Evapotranspiração por CNN (Versão Ajustada)")

# ---------------------------------------------------------------
# 4. Carregar imagem
# ---------------------------------------------------------------
st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")

uploaded_file = st.file_uploader("Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_container_width=True)

# ---------------------------------------------------------------
# 5. Dados dos espécimes
# ---------------------------------------------------------------
st.header("2) Insira as Variáveis Físicas dos Espécimes")

num_especies = st.number_input("Quantidade de Espécimes:", min_value=1, step=1, value=1)

# Lista para armazenar dados temporariamente
especies_data = []
for i in range(num_especies):
    st.subheader(f"Espécime {i+1}")
    altura = st.text_input(f"Altura (m) - Espécime {i+1}:", "0")
    diametro = st.text_input(f"Diâmetro do Tronco (cm) - Espécime {i+1}:", "0")
    copa = st.text_input(f"Área da Copa (m²) - Espécime {i+1}:", "0")
    galhos = st.number_input(f"Quantidade de Galhos - Espécime {i+1}:", min_value=1, step=1, value=1)

    folhas_data = []
    for j in range(galhos):
        st.markdown(f"**Galho {j+1}** - Espécime {i+1}")
        largura_folha = st.text_input(f"Largura da Folha (cm) - Galho {j+1} - Espécime {i+1}:", "0")
        comprimento_folha = st.text_input(f"Comprimento da Folha (cm) - Galho {j+1} - Espécime {i+1}:", "0")
        folhas_data.append((largura_folha, comprimento_folha))

    especies_data.append((altura, diametro, copa, galhos, folhas_data))

# ---------------------------------------------------------------
# 6. Botão: Calcular evapotranspiração (modelo)
# ---------------------------------------------------------------
st.header("3) Cálculo da Evapotranspiração (Modelo)")

if st.button("Calcular Evapotranspiração"):
    # Zera lista de resultados para recalcular
    st.session_state.resultados = []

    # Verifica se há imagem carregada
    if st.session_state.uploaded_image is None:
        st.error("É necessário carregar uma imagem antes de calcular.")
    else:
        # Calcula para cada espécime
        for i, (altura_str, diametro_str, copa_str, galhos, folhas_data) in enumerate(especies_data):
            try:
                altura_val = float(altura_str)
                diametro_val = float(diametro_str)
                copa_val = float(copa_str)

                # Área foliar total
                aft = calculate_area_foliar_total(folhas_data, galhos)
                # LAI
                lai_val = calculate_lai(aft, copa_val)
                # Modelo
                et_val = predict_evapotranspiration(
                    st.session_state.uploaded_image,
                    altura_val,
                    diametro_val,
                    copa_val,
                    lai_val
                )

                st.session_state.resultados.append(et_val)
                st.success(f"Espécime {i+1}: {et_val} litros/dia (modelo)")

            except ValueError:
                st.error(f"Espécime {i+1}: Por favor, insira valores numéricos válidos.")
                # Se der erro, interrompe
                break

# ---------------------------------------------------------------
# 7. Contraprova Experimental
# ---------------------------------------------------------------
st.header("4) Contraprova Experimental com Múltiplas Medições")

num_experimentos = st.number_input(
    "Quantidade de medições experimentais para cada Espécime:",
    min_value=1, step=1, value=1
)

contraprovas = {}
for i in range(num_especies):
    st.subheader(f"Espécime {i+1} - Valores Experimentais (mL)")
    valores_experimentais = []
    for j in range(num_experimentos):
        val = st.text_input(f"Medição {j+1} (mL) - Espécime {i+1}:", "0")
        valores_experimentais.append(val)
    contraprovas[i] = valores_experimentais

tempo_coleta_horas = st.number_input("Tempo (horas) de coleta para cada medição:", min_value=1, step=1, value=24)

if st.button("Comparar com a Contraprova"):
    # Verifica se temos resultados para todos os espécimes
    if len(st.session_state.resultados) == num_especies:
        # Para cada espécime
        for i in range(num_especies):
            st.markdown(f"---\n**Espécime {i+1}:**")

            try:
                # Converte strings para float
                valores_exp_float = [float(x) for x in contraprovas[i]]
                # Converte mL -> L e ajusta para 24h (litros/dia)
                evap_exps = []
                for vol_mL in valores_exp_float:
                    vol_L = vol_mL / 1000.0
                    vol_L_dia = vol_L / (tempo_coleta_horas / 24.0)
                    evap_exps.append(vol_L_dia)

                # Exibe medições convertidas
                st.write("Medições (litros/dia):", [f"{v:.2f}" for v in evap_exps])

                media_experimental = np.mean(evap_exps)
                et_modelo = st.session_state.resultados[i]

                st.write(f"Média experimental: {media_experimental:.2f} litros/dia")
                st.write(f"Valor previsto pelo modelo: {et_modelo:.2f} litros/dia")

                # Verifica se há pelo menos 2 valores diferentes
                valores_unicos = set(evap_exps)
                if len(evap_exps) < 2 or len(valores_unicos) < 2:
                    # Se há apenas 1 medição ou todas medições iguais
                    st.warning(
                        "Não é possível realizar o teste t com uma única medição ou valores idênticos. "
                        "O teste exige pelo menos 2 valores distintos."
                    )
                    # Como alternativa, exibimos a diferença absoluta:
                    diferenca_abs = abs(media_experimental - et_modelo)
                    st.write(f"Diferença (modelo x experimento): {diferenca_abs:.2f} litros/dia")
                else:
                    # Pode fazer o teste t normalmente
                    t_stat, p_value = stats.ttest_1samp(evap_exps, et_modelo)

                    st.write(f"T-estatística: {t_stat:.4f}")
                    st.write(f"P-valor: {p_value:.6f}")

                    alpha = 0.05
                    if p_value < alpha:
                        st.error("Diferença estatisticamente significativa (p < 0.05).")
                    else:
                        st.info("Diferença não é estatisticamente significativa (p ≥ 0.05).")

            except ValueError:
                st.error(f"Espécime {i+1}: Insira valores experimentais válidos (números).")

    else:
        st.warning("É necessário primeiro calcular a evapotranspiração pelo modelo para todos os espécimes.")

# ---------------------------------------------------------------
# 8. Seção Explicativa Expandida
# ---------------------------------------------------------------
with st.expander("Explicação Técnica e Interpretação Detalhada"):
    st.markdown("""
    ## 1. Cálculo da Área Foliar Total (AFT)
    A **Área Foliar Total** é obtida ao somar a área de cada folha 
    (largura × comprimento) multiplicada pelo número de galhos.
    """)
    st.latex(r'''
    \text{AFT} = \sum_{i=1}^{n} (\text{largura}_i \times \text{comprimento}_i)\times \text{galhos}
    ''')
    st.markdown("""
    ## 2. Índice de Área Foliar (LAI)
    O **LAI** é a razão entre a AFT e a área de projeção da copa:
    """)
    st.latex(r'''
    \text{LAI} = \frac{\text{AFT}}{\text{Área da Copa}}
    ''')
    st.markdown("""
    ## 3. Evapotranspiração (Modelo)
    Exemplo de equação (coeficientes ilustrativos):
    """)
    st.latex(r'''
    \text{ET (litros/dia)} = 
    [0.5 \times \text{Altura (m)} + 0.3 \times \text{Diâmetro (cm)} 
    + 0.1 \times \text{Área de Copa (m²)} + 0.2 \times \text{LAI}] \times 10
    ''')
    st.markdown("""
    Ajuste os coeficientes conforme o seu modelo real ou estudo específico.

    ## 4. Contraprova Experimental
    - O volume coletado (mL) é convertido para litros (dividindo por 1000).
    - Ajustamos para 24 horas, dividindo pela razão (tempo_coleta_horas / 24).

    ## 5. Teste de Hipótese (Teste t de Student - 1 amostra)
    - **Hipótese nula (H0):** A média experimental é igual ao valor do modelo.
    - **Hipótese alternativa (H1):** A média experimental difere do valor do modelo.
    - Se p-valor < 0.05 (nível de significância α): rejeita-se H0, indicando diferença significativa.

    ### Por que T-estatística e P-valor podem ser NaN?
    - Se houver **apenas 1 medição** ou **todas medições forem idênticas**, a variância é zero, impossibilitando o cálculo do teste t.
    - Nesse caso, exibimos uma mensagem indicando ser inviável realizar estatística inferencial com apenas 1 valor 
      (ou valores iguais).

    ## 6. Diferença Absoluta
    Quando não é possível fazer o teste t (poucos valores), exibimos a diferença absoluta 
    entre a média experimental e o modelo (litros/dia). Esse número **não** indica significância estatística, 
    mas pelo menos informa quão distante está o valor medido do previsto.

    ## 7. Boas Práticas Finais
    - Coletar 2 ou mais medições **em diferentes condições** para estimar a variabilidade da contraprova.
    - Validar as variáveis (altura, diâmetro, etc.) dentro de faixas plausíveis para evitar inputs irreais.
    - Considerar dados climáticos (umidade, radiação, temperatura) para maior realismo.
    - Se desejar usar CNN real, treine um modelo usando imagens + variáveis para prever evapotranspiração.
    """)

