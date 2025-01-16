import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------------
# 1. Armazenamento em session_state
# ---------------------------------------------------------------
# Inicializa variáveis em session_state para persistência
if "resultados" not in st.session_state:
    st.session_state.resultados = []  # Lista de evapotranspirações calculadas
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ---------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------
def calculate_area_foliar_total(folhas_data, galhos):
    """
    Calcula a área foliar total com base nas dimensões (largura x comprimento)
    de cada folha e no número de galhos do espécime.

    Parâmetros:
    -----------
    folhas_data : list of tuples
        Lista de tuplas (largura_str, comprimento_str) para cada galho.
    galhos : int
        Número de galhos do espécime.

    Retorna:
    --------
    float
        Área foliar total.
    """
    total_area = 0.0
    for largura_str, comprimento_str in folhas_data:
        try:
            largura = float(largura_str)
            comprimento = float(comprimento_str)
            total_area += (largura * comprimento) * galhos
        except ValueError:
            # Se algum valor não puder ser convertido, ignoramos a folha.
            continue
    return total_area

def calculate_lai(area_foliar_total, area_copa):
    """
    Calcula o LAI (Leaf Area Index) = Área Foliar Total / Área da Copa.

    Parâmetros:
    -----------
    area_foliar_total : float
        Área foliar total.
    area_copa : float
        Área da copa (m²).

    Retorna:
    --------
    float
        Valor do LAI (arredondado em 2 casas decimais).
        Retorna 0.0 se ocorrer zero division ou erro de conversão.
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
    ilustrativos. A imagem é recebida como parâmetro apenas para fins
    de compatibilidade com um hipotético modelo de CNN.

    Parâmetros:
    -----------
    image : PIL.Image
        Imagem da espécie (não usada diretamente aqui).
    altura : float
        Altura do espécime (em metros).
    diametro : float
        Diâmetro do tronco (em centímetros).
    copa : float
        Área da copa (em m²).
    lai : float
        Índice de área foliar (sem unidade).

    Retorna:
    --------
    float
        Evapotranspiração simulada, em litros/dia.
    """
    # Exemplo de fórmula (coeficientes são fictícios):
    # Evapotranspiração = (0.5*altura + 0.3*diametro + 0.1*copa + 0.2*lai)*10
    et = (altura * 0.5 + diametro * 0.3 + copa * 0.1 + lai * 0.2) * 10
    return round(et, 2)

# ---------------------------------------------------------------
# 3. Cabeçalho e título do app
# ---------------------------------------------------------------
st.title("Estimativa de Evapotranspiração por CNN (Versão Ajustada)")

# ---------------------------------------------------------------
# 4. Carregar imagem
# ---------------------------------------------------------------
st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")

uploaded_file = st.file_uploader("Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    # Armazena a imagem em session_state para persistir
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_container_width=True)

# ---------------------------------------------------------------
# 5. Entrada de dados dos espécimes
# ---------------------------------------------------------------
st.header("2) Insira as Variáveis Físicas dos Espécimes")

num_especies = st.number_input("Quantidade de Espécimes:", min_value=1, step=1, value=1)

# Lista temporária para coletar informações de cada espécime
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

    # Armazena as informações para processamento posterior
    especies_data.append((altura, diametro, copa, galhos, folhas_data))

# ---------------------------------------------------------------
# 6. Botão para calcular evapotranspiração do modelo
# ---------------------------------------------------------------
st.header("3) Cálculo da Evapotranspiração (Modelo)")

if st.button("Calcular Evapotranspiração"):
    # Zera a lista de resultados para recalcular
    st.session_state.resultados = []

    # Verifica se há imagem carregada
    if st.session_state.uploaded_image is None:
        st.error("É necessário carregar uma imagem antes de calcular.")
    else:
        # Para cada espécime
        for i, (altura_str, diametro_str, copa_str, galhos, folhas_data) in enumerate(especies_data):
            try:
                altura_val = float(altura_str)
                diametro_val = float(diametro_str)
                copa_val = float(copa_str)

                # Cálculo da área foliar total
                aft = calculate_area_foliar_total(folhas_data, galhos)
                # Cálculo do LAI
                lai_val = calculate_lai(aft, copa_val)
                # Evapotranspiração do modelo
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
                # Se der erro, interrompe a execução do laço
                break

# ---------------------------------------------------------------
# 7. Contraprova Experimental (múltiplas medições)
# ---------------------------------------------------------------
st.header("4) Contraprova Experimental com Múltiplas Medições")

num_experimentos = st.number_input(
    "Quantidade de medições experimentais para cada Espécime:",
    min_value=1, step=1, value=1
)

# Coleta dos valores experimentais (em mL)
contraprovas = {}
for i in range(num_especies):
    st.subheader(f"Espécime {i+1} - Valores Experimentais (mL)")
    valores_experimentais = []
    for j in range(num_experimentos):
        val = st.text_input(f"Medição {j+1} (mL) - Espécime {i+1}:", "0")
        valores_experimentais.append(val)
    contraprovas[i] = valores_experimentais

tempo_coleta_horas = st.number_input("Tempo (horas) de coleta para cada medição:", min_value=1, step=1, value=24)

# Botão para comparar com a contraprova
if st.button("Comparar com a Contraprova"):
    if len(st.session_state.resultados) == num_especies:
        # Para cada espécime, faz o teste t de Student (1 amostra)
        for i in range(num_especies):
            st.markdown(f"---\n**Espécime {i+1}:**")

            try:
                # Converte strings para float
                valores_exp_float = [float(x) for x in contraprovas[i]]
                # Converte mL -> litros e ajusta para 24h (litros/dia)
                evap_exps = []
                for vol_mL in valores_exp_float:
                    vol_L = vol_mL / 1000.0  # mL para litros
                    vol_L_dia = vol_L / (tempo_coleta_horas / 24.0)
                    evap_exps.append(vol_L_dia)

                st.write("Medições (litros/dia):", [f"{v:.2f}" for v in evap_exps])

                media_experimental = np.mean(evap_exps)
                et_modelo = st.session_state.resultados[i]

                st.write(f"Média experimental: {media_experimental:.2f} litros/dia")
                st.write(f"Valor previsto pelo modelo: {et_modelo:.2f} litros/dia")

                # Teste t: comparando a lista evap_exps com a média do modelo
                t_stat, p_value = stats.ttest_1samp(evap_exps, et_modelo)

                st.write(f"T-estatística: {t_stat}")
                st.write(f"P-valor: {p_value}")

                alpha = 0.05
                # Verifica se resultado é numérico ou NaN
                if np.isnan(t_stat) or np.isnan(p_value):
                    st.warning("T-estatística ou P-valor retornaram 'NaN' — veja explicação técnica!")
                else:
                    # Interpretação do resultado
                    if p_value < alpha:
                        st.error("Diferença estatisticamente significativa (p < 0.05).")
                    else:
                        st.info("Diferença não é estatisticamente significativa (p ≥ 0.05).")

            except ValueError:
                st.error(f"Espécime {i+1}: Insira valores experimentais válidos (números).")

    else:
        st.warning("É necessário primeiro calcular a evapotranspiração pelo modelo para todos os espécimes.")

# ---------------------------------------------------------------
# 8. Seção Explicativa (expandida)
# ---------------------------------------------------------------
with st.expander("Explicação Técnica e Interpretação Detalhada"):
    st.markdown("""
    ## 1) Cálculo da Área Foliar Total (AFT)
    A **Área Foliar Total (AFT)** é a soma da área de cada folha, multiplicada pelo número total de galhos do espécime.
    Admitindo que cada folha em um galho tem a mesma dimensão (largura × comprimento), temos:
    """)
    st.latex(r'''
    \text{AFT} = \sum_{i=1}^{n} (\text{largura}_i \times \text{comprimento}_i)\times \text{galhos}
    ''')
    st.markdown("""
    - Caso haja variação entre folhas de um mesmo galho, pode-se ajustar o código para individualizar cada folha.

    ## 2) Índice de Área Foliar (LAI)
    O **LAI (Leaf Area Index)** é a razão entre a área foliar total (AFT) e a área de projeção do dossel (copa):
    """)
    st.latex(r'''
    \text{LAI} = \frac{\text{AFT}}{\text{Área da Copa}}
    ''')
    st.markdown("""
    - Se a área da copa for muito pequena (ou zero), teremos problemas de divisão por zero, resultando em LAI = 0.0.

    ## 3) Cálculo de Evapotranspiração (Modelo)
    Usamos coeficientes **exemplificativos**:
    """)
    st.latex(r'''
    \text{ET (litros/dia)} = 
    [0.5 \times \text{Altura (m)} + 0.3 \times \text{Diâmetro (cm)} 
    + 0.1 \times \text{Área de Copa (m²)} + 0.2 \times \text{LAI}] \times 10
    ''')
    st.markdown("""
    > **Obs.:** Ajuste esses coeficientes segundo o seu **modelo real** ou conforme experimentos validados.

    ## 4) Contraprova Experimental
    - **Coleta de água** (mL) em certo intervalo (horas).
    - Conversão de mL para litros (dividir por 1000).
    - Ajuste para 24h: multiplica (ou divide) considerando `tempo_coleta_horas`.  
      Ex.: `litros_dia = (vol_mL / 1000) / (tempo_coleta_horas / 24)`

    ## 5) Teste de Hipótese (Teste t de Student - 1 amostra)
    - **Hipótese Nula (H0):** a média experimental **não** difere do valor previsto pelo modelo.
    - **Hipótese Alternativa (H1):** a média experimental **difere** significativamente do valor previsto.
    - Estatística do teste (t) e p-valor:
      - Se `p_value < α` (ex.: 0.05), **rejeitamos** H0.
      - Caso contrário, **não rejeitamos** H0.

    ### Caso T-estatística: NaN e P-valor: NaN
    Se o teste t retornar valores `NaN`, isso ocorre principalmente por:
    1. **Amostra com tamanho 1 ou 0** – Não há variabilidade suficiente para o teste t.
    2. **Todos os valores experimentais são idênticos** – A variância é zero, resultando em divisão por zero no cálculo.
    3. **Valores inválidos** – Se houver `inf` ou já for `NaN` em alguma medição.
    
    > Quando `NaN` ocorre, significa que **não** há como calcular a significância estatística com esse conjunto de dados.  
    > Soluções possíveis:  
    > - Coletar mais medições (para ter pelo menos 2 ou mais valores diferentes).  
    > - Verificar se o tipo de teste usado é adequado, ou se é preciso outro teste estatístico.

    ## 6) Potenciais Melhores Práticas
    - Incorporar dados climáticos (umidade, radiação solar, temperatura) para um modelo de evapotranspiração mais acurado.
    - Ajustar ou substituir a fórmula atual por um **modelo baseado em CNN** (Rede Neural Convolucional) devidamente treinado.
    - Validar intervalos de valores (ex.: altura entre 0,5m e 100m, etc.) para evitar inputs irreais.
    - Fornecer **múltiplas medições** para cada espécime ao longo de diferentes dias/horas, melhorando a robustez da contraprova.
    """)

