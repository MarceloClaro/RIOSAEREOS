import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats

# ----------------------------------------------------------------------
# 1) ARMAZENAMENTO DE RESULTADOS: session_state
# ----------------------------------------------------------------------
# Garante que a lista de resultados e as imagens fiquem salvas na sessão
if "resultados" not in st.session_state:
    st.session_state.resultados = []  # evapotranspiração (modelo)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ----------------------------------------------------------------------
# 2) FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------
def calculate_area_foliar_total(folhas_data, galhos):
    """
    Calcula a área foliar total com base nas dimensões (largura x comprimento)
    de cada folha e no número de galhos do espécime.
    
    folhas_data : list of tuples (largura_str, comprimento_str)
    galhos : int
    return : float (área foliar total)
    """
    total_area = 0.0
    for largura_str, comprimento_str in folhas_data:
        try:
            largura = float(largura_str)
            comprimento = float(comprimento_str)
            total_area += (largura * comprimento) * galhos
        except ValueError:
            # Se não conseguiu converter algum valor, ignora aquela folha
            continue
    return total_area

def calculate_lai(area_foliar_total, area_copa):
    """
    LAI (Leaf Area Index) = Área Foliar Total / Área da Copa
    Se ocorrer ZeroDivisionError ou ValueError, retorna 0.0
    """
    try:
        area_copa_val = float(area_copa)
        lai = area_foliar_total / area_copa_val
        return round(lai, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0

def predict_evapotranspiration(image, altura, diametro, copa, lai):
    """
    Função genérica (dummy) para prever a evapotranspiração (litros/dia),
    usando coeficientes ilustrativos. A 'image' está aqui apenas como parâmetro,
    caso futuramente utilize CNN ou outra técnica que dependa da imagem.
    """
    # altura (m), diametro (cm), copa (m²), lai (sem unidade)
    # Apenas um exemplo ilustrativo:
    evapotranspiracao = (altura * 0.5 + diametro * 0.3 + copa * 0.1 + lai * 0.2) * 10
    return round(evapotranspiracao, 2)

# ----------------------------------------------------------------------
# 3) CABEÇALHO E TÍTULO
# ----------------------------------------------------------------------
st.title("Estimativa de Evapotranspiração por CNN (Versão Ajustada)")

# ----------------------------------------------------------------------
# 4) CARREGAR IMAGEM
# ----------------------------------------------------------------------
st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")

uploaded_file = st.file_uploader("Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    # Armazena a imagem em session_state
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_container_width=True)

# ----------------------------------------------------------------------
# 5) DADOS DOS ESPÉCIMES
# ----------------------------------------------------------------------
st.header("2) Insira os Dados Físicos dos Espécimes")

num_especies = st.number_input("Quantidade de Espécimes:", min_value=1, step=1, value=1)

# Vamos criar uma lista temporária para coletar informações de cada espécime
especies_data = []

for i in range(num_especies):
    st.subheader(f"Espécime {i+1}")
    altura = st.text_input(f"Altura (m) - Espécime {i+1}:", "0")
    diametro = st.text_input(f"Diâmetro do Tronco (cm) - Espécime {i+1}:", "0")
    copa = st.text_input(f"Área da Copa (m²) - Espécime {i+1}:", "0")
    galhos = st.number_input(f"Quantidade de Galhos - Espécime {i+1}:", min_value=1, step=1, value=1)
    
    # Dimensões de folhas (largura, comprimento) para cada galho
    folhas_data = []
    for j in range(galhos):
        st.markdown(f"**Galho {j+1}** - Espécime {i+1}")
        largura_folha = st.text_input(f"Largura da Folha (cm) - Galho {j+1} - Espécime {i+1}:", "0")
        comprimento_folha = st.text_input(f"Comprimento da Folha (cm) - Galho {j+1} - Espécime {i+1}:", "0")
        folhas_data.append((largura_folha, comprimento_folha))

    # Armazena essas infos para uso posterior
    especies_data.append((altura, diametro, copa, galhos, folhas_data))

# ----------------------------------------------------------------------
# 6) BOTÃO PARA CALCULAR EVAPOTRANSPIRAÇÃO (MODELO)
# ----------------------------------------------------------------------
st.header("3) Cálculo da Evapotranspiração (Modelo)")

if st.button("Calcular Evapotranspiração"):
    # Zera a lista de resultados para refazer o cálculo do zero
    st.session_state.resultados = []
    
    # Verifica se há imagem carregada
    if st.session_state.uploaded_image is None:
        st.error("É necessário carregar uma imagem antes de calcular.")
    else:
        # Tenta calcular para cada espécime
        for i, (altura_str, diametro_str, copa_str, galhos, folhas_data) in enumerate(especies_data):
            try:
                altura_val = float(altura_str)
                diametro_val = float(diametro_str)
                copa_val = float(copa_str)
                
                # Calcula área foliar total
                aft = calculate_area_foliar_total(folhas_data, galhos)
                
                # Calcula LAI
                lai_val = calculate_lai(aft, copa_val)
                
                # Predição pelo modelo
                et_val = predict_evapotranspiration(
                    st.session_state.uploaded_image,
                    altura_val,
                    diametro_val,
                    copa_val,
                    lai_val
                )
                
                # Armazena o resultado no session_state
                st.session_state.resultados.append(et_val)
                
                st.success(f"Espécime {i+1}: {et_val} litros/dia (modelo)")
            except ValueError:
                st.error(f"Espécime {i+1}: Por favor, insira valores numéricos válidos.")
                # Se der erro em qualquer espécime, interrompe
                break

# ----------------------------------------------------------------------
# 7) CONTRAPROVA EXPERIMENTAL (MÚLTIPLAS MEDIÇÕES)
# ----------------------------------------------------------------------
st.header("4) Contraprova Experimental com Múltiplas Medições")

num_experimentos = st.number_input(
    "Quantidade de medições experimentais para cada Espécime:", 
    min_value=1, step=1, value=1
)

# Coleta dos valores experimentais
contraprovas = {}

for i in range(num_especies):
    st.subheader(f"Espécime {i+1} - Valores Experimentais (mL) por medição")
    valores_experimentais = []
    for j in range(num_experimentos):
        val = st.text_input(f"Medição {j+1} (mL) - Espécime {i+1}:", "0")
        valores_experimentais.append(val)
    contraprovas[i] = valores_experimentais

tempo_coleta_horas = st.number_input("Tempo (horas) de coleta para cada medição:", min_value=1, step=1, value=24)

if st.button("Comparar com a Contraprova"):
    # Verifica se temos resultados de todos os espécimes
    if len(st.session_state.resultados) == num_especies:
        # Para cada espécime, faz o teste t
        for i in range(num_especies):
            st.markdown(f"---\n**Espécime {i+1}:**")
            
            try:
                # Converte valores de string para float
                valores_experimentais_float = [float(x) for x in contraprovas[i]]
                # Converte mL -> L, e ajusta para 24h
                evap_exps = []
                for vol_mL in valores_experimentais_float:
                    vol_L = vol_mL / 1000.0  # mL para litros
                    vol_L_dia = vol_L / (tempo_coleta_horas / 24.0)
                    evap_exps.append(vol_L_dia)
                
                # Mostra medições convertidas
                st.write("Medições (litros/dia):", [f"{v:.2f}" for v in evap_exps])
                
                media_experimental = np.mean(evap_exps)
                et_modelo = st.session_state.resultados[i]
                
                st.write(f"Média experimental: {media_experimental:.2f} litros/dia")
                st.write(f"Valor previsto pelo modelo: {et_modelo:.2f} litros/dia")
                
                # Teste t: comparando a lista evap_exps com a média do modelo (et_modelo)
                t_stat, p_value = stats.ttest_1samp(evap_exps, et_modelo)
                
                st.write(f"T-estatística: {t_stat:.2f}")
                st.write(f"P-valor: {p_value:.4f}")
                
                alpha = 0.05
                if p_value < alpha:
                    st.error("A diferença é estatisticamente significativa (p < 0.05).")
                else:
                    st.info("A diferença não é estatisticamente significativa (p >= 0.05).")
            
            except ValueError:
                st.error(f"Espécime {i+1}: Insira valores experimentais válidos (números).")
            
    else:
        st.warning("É necessário primeiro calcular a evapotranspiração pelo modelo para todos os espécimes.")

# ----------------------------------------------------------------------
# 8) SEÇÃO EXPLICATIVA (EXPANDER)
# ----------------------------------------------------------------------
with st.expander("Explicação Técnica - Clique para Expandir"):
    st.markdown("""
    ### 1. Cálculo da Área Foliar Total (AFT)
    A **Área Foliar Total** é calculada somando-se a área de cada folha (largura × comprimento)
    multiplicada pela quantidade de galhos, assumindo que cada galho tem folhas de dimensões semelhantes.
    """)
    st.latex(r'''
    \text{AFT} = \sum_{i=1}^{n} (\text{Largura}_i \times \text{Comprimento}_i)\times \text{galhos}
    ''')
    
    st.markdown("""
    ### 2. Cálculo do Índice de Área Foliar (LAI)
    O **LAI** (Leaf Area Index) é a razão entre a área foliar total (AFT) e a área de projeção do dossel (copa):
    """)
    st.latex(r'''
    \text{LAI} = \frac{\text{AFT}}{\text{Área da Copa}}
    ''')
    
    st.markdown("""
    ### 3. Evapotranspiração (Modelo)
    Exemplo de formulação simples (coeficientes ilustrativos):
    """)
    st.latex(r'''
    \text{ET}(\text{litros/dia}) = 
    [0.5 \times \text{Altura} + 0.3 \times \text{Diâmetro} 
    + 0.1 \times \text{Área de Copa} + 0.2 \times \text{LAI}] \times 10
    ''')
    
    st.markdown("""
    ### 4. Contraprova Experimental
    - Convertemos **mL** para **litros** (dividindo por 1000).
    - Para extrapolar para **litros/dia**, dividimos o volume coletado pelo tempo (em horas)
      e então multiplicamos por 24, ou seja, `volume_L / (tempo_horas/24)`.
    
    ### 5. Teste de Hipótese
    - **Teste t de Student** (1 amostra): Compara a média experimental à estimativa do modelo.
    - **Hipótese nula (H0):** a média experimental é igual (ou não difere significativamente) do valor do modelo.
    - Se o **p-valor < 0.05**, rejeitamos H0, concluindo que há diferença significativa.
    
    ### 6. Possíveis Ajustes
    - Incorporar fatores climáticos (temperatura, radiação solar, umidade relativa) para um modelo mais realista.
    - Ajustar os coeficientes ou usar um modelo de Machine Learning treinado com dados reais.
    - Validar melhor os intervalos de valores (ex.: altura mínima e máxima possíveis).
    """)

