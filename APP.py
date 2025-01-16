import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats

# ======================================
# Funções auxiliares
# ======================================
def calculate_area_foliar_total(folhas_data, galhos):
    """
    Calcula a área foliar total com base nas dimensões das folhas (largura x comprimento) 
    e no número de galhos do espécime.
    
    Parâmetros:
    -----------
    folhas_data : list of tuples
        Lista de tuplas (largura, comprimento) para cada galho.
    galhos : int
        Quantidade de galhos (galhos).

    Retorna:
    --------
    float
        Área foliar total.
    """
    total_area = 0
    for largura, comprimento in folhas_data:
        try:
            largura = float(largura)
            comprimento = float(comprimento)
            # Área de uma folha * número de galhos
            total_area += (largura * comprimento) * galhos
        except ValueError:
            continue
    return total_area

def calculate_lai(area_foliar_total, area_copa):
    """
    Calcula o LAI (Leaf Area Index) com base na área foliar total e na área da copa.

    Parâmetros:
    -----------
    area_foliar_total : float
        Área foliar total (em cm², m², etc.).
    area_copa : float
        Área da copa (m²).

    Retorna:
    --------
    float
        Valor do LAI arredondado em 2 casas decimais.
    """
    try:
        lai = area_foliar_total / float(area_copa)  # LAI baseado na área foliar total
        return round(lai, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0

def predict_evapotranspiration(image, altura, diametro, copa, lai):
    """
    Simula a previsão de evapotranspiração (em litros/dia) baseada em:
    - altura (m)
    - diâmetro do tronco (cm)
    - área de copa (m²)
    - LAI (índice de área foliar)
    - e a imagem (embora aqui, seja utilizada apenas como dummy).
    
    Parâmetros:
    -----------
    image : PIL.Image
        Imagem da espécie (atualmente não usada nos cálculos).
    altura : float
        Altura do espécime em metros.
    diametro : float
        Diâmetro do tronco em centímetros.
    copa : float
        Área da copa em m².
    lai : float
        Índice de área foliar (LAI).

    Retorna:
    --------
    float
        Evapotranspiração simulada em litros/dia.
    """
    # Exemplo de cálculo (coeficientes meramente ilustrativos):
    evapotranspiracao = (float(altura) * 0.5 + float(diametro) * 0.3 + float(copa) * 0.1 + float(lai) * 0.2) * 10
    return round(evapotranspiracao, 2)

# ======================================
# Interface com Streamlit
# ======================================
st.title("Estimativa de Evapotranspiração por CNN (Expandido)")

# Carregamento da imagem
st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")
uploaded_image = st.file_uploader("Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagem Carregada", use_container_width=True)
else:
    image = None

# Entrada de variáveis físicas
st.header("2) Insira as Variáveis Físicas dos Espécimes")

num_especies = st.number_input("Quantidade de Espécimes:", min_value=1, step=1, value=1)
especies_data = []

for i in range(num_especies):
    st.subheader(f"Espécime {i + 1}")
    
    altura = st.text_input(f"Altura (m) - Espécime {i + 1}:", "0")
    diametro = st.text_input(f"Diâmetro do Tronco (cm) - Espécime {i + 1}:", "0")
    copa = st.text_input(f"Área da Copa (m²) - Espécime {i + 1}:", "0")
    galhos = st.number_input(f"Quantidade de Galhos - Espécime {i + 1}:", min_value=1, step=1, value=1)
    
    # Dimensões de folhas por galho
    folhas_data = []
    for j in range(galhos):
        st.markdown(f"**Dimensão Foliar para o Galho {j + 1}** - Espécime {i + 1}")
        largura_folha = st.text_input(f"Largura da Folha (cm) - Galho {j + 1} - Espécime {i + 1}:", "0")
        comprimento_folha = st.text_input(f"Comprimento da Folha (cm) - Galho {j + 1} - Espécime {i + 1}:", "0")
        folhas_data.append((largura_folha, comprimento_folha))
    
    # Cálculo da área foliar total
    area_foliar_total = calculate_area_foliar_total(folhas_data, galhos)
    
    # Cálculo do LAI (índice de área foliar)
    if area_foliar_total > 0 and copa != "0":
        lai = calculate_lai(area_foliar_total, copa)
        st.text(f"LAI Calculado para o Espécime {i + 1}: {lai}")
    else:
        lai = 0
    
    especies_data.append((altura, diametro, copa, lai, galhos, folhas_data))

# Botão para calcular evapotranspiração
st.header("3) Cálculo de Evapotranspiração (Modelo)")
resultados = []
if st.button("Calcular Evapotranspiração"):
    # Verifica se todas as variáveis foram preenchidas corretamente
    if image is not None:
        try:
            for i, (altura, diametro, copa, lai, galhos, folhas_data) in enumerate(especies_data):
                altura_val = float(altura)
                diametro_val = float(diametro)
                copa_val = float(copa)
                lai_val = float(lai)
                
                evapotranspiracao = predict_evapotranspiration(
                    image,
                    altura_val,
                    diametro_val,
                    copa_val,
                    lai_val
                )
                
                resultados.append(evapotranspiracao)
                st.success(f"Espécime {i + 1}: {evapotranspiracao} litros/dia (modelo)")
        except ValueError:
            st.error("Por favor, insira valores numéricos válidos para todas as variáveis físicas.")
    else:
        st.error("É necessário carregar uma imagem antes de calcular.")

# ======================================
# Contraprova experimental (medições)
# ======================================
st.header("4) Contraprova Experimental")
"""
Para tornar o teste de hipóteses mais robusto, você pode inserir **várias medições** 
experimentais (contraprova) para cada espécime. Dessa forma, o teste t pode avaliar 
de fato se a média das medições experimentais difere do valor previsto pelo modelo.
"""

num_experimentos = st.number_input("Quantidade de valores experimentais de contraprova para cada Espécime:", 
                                   min_value=1, step=1, value=1)

# Dicionário para armazenar dados experimentais (volume em mL / dia ou algo similar)
contraprovas = {}

for i in range(num_especies):
    st.subheader(f"Espécime {i+1} - Valores Experimentais")
    valores_experimentais = []
    for j in range(num_experimentos):
        val = st.text_input(f"Volume coletado (mL) na medição {j+1} do Espécime {i+1}:", "0")
        valores_experimentais.append(val)
    contraprovas[i] = valores_experimentais

# Duração total em horas para cada medição
tempo_coleta = st.number_input("Tempo (horas) para cada medição experimental:", min_value=1, step=1, value=24)

if st.button("Comparar com a Contraprova"):
    if len(resultados) == num_especies:
        # Para cada espécime, comparamos
        for i, evap_modelo in enumerate(resultados):
            st.write("---")
            st.write(f"**Espécime {i+1}:**")
            try:
                # Converte as strings de volume para float
                valores_experimentais = [float(v) for v in contraprovas[i]]
                
                # Calcula evapotranspiração experimental (litros/dia) para cada medição
                # volume_coletado (mL) -> litros = volume_coletado/1000
                # Se cada medição foi feita ao longo de "tempo_coleta" horas,
                # ajustamos para 24 horas se quisermos "litros/dia".
                
                evap_exps = []
                for volume_mL in valores_experimentais:
                    evap_L = volume_mL / 1000  # converte para litros
                    # Ajuste para litros/dia
                    evap_L_day = evap_L / (tempo_coleta / 24.0) 
                    evap_exps.append(evap_L_day)
                
                # Apresenta as medições experimentais já convertidas
                st.write(f"Medições experimentais (litros/dia): {['{:.2f}'.format(e) for e in evap_exps]}")

                # Faz o teste de hipótese (t de Student) comparando a média das medições experimentais
                # com o valor previsto pelo modelo (evap_modelo).
                media_experimental = np.mean(evap_exps)
                st.write(f"Média experimental: {media_experimental:.2f} litros/dia")
                st.write(f"Valor previsto pelo modelo: {evap_modelo:.2f} litros/dia")
                
                # Teste t: H0 = não há diferença entre a média experimental e o valor do modelo
                t_stat, p_value = stats.ttest_1samp(evap_exps, evap_modelo)

                alpha = 0.05  # nível de significância
                st.write(f"T-estatística: {t_stat:.2f}")
                st.write(f"P-valor: {p_value:.4f}")
                
                if p_value < alpha:
                    st.error("A diferença é estatisticamente significativa (Rejeita H0).")
                else:
                    st.info("A diferença não é estatisticamente significativa (Não rejeita H0).")

            except ValueError:
                st.error(f"Por favor, insira valores numéricos válidos para o Espécime {i+1}.")
    else:
        st.warning("É necessário primeiro calcular a evapotranspiração pelo modelo para todos os espécimes.")

# ======================================
# Seção Explicativa
# ======================================
with st.expander("Explicação Técnica (Clique para expandir)"):
    st.markdown("""
    **1. Cálculo da Área Foliar Total (AFT)**  
    A AFT é a soma das áreas de cada folha multiplicada pelo número de galhos, 
    assumindo que cada galho tem folhas de dimensões semelhantes:
    """)
    st.latex(r'''
    \text{AFT} = \sum_{i=1}^{n} (\text{largura}_i \times \text{comprimento}_i) \times \text{galhos}
    ''')
    
    st.markdown("""
    **2. Índice de Área Foliar (LAI)**  
    O LAI (Leaf Area Index) é a razão entre a área foliar total (AFT) e a 
    área de projeção do dossel/copa (AC):
    """)
    st.latex(r'''
    \text{LAI} = \frac{\text{AFT}}{\text{AC}}
    ''')
    
    st.markdown("""
    **3. Cálculo de Evapotranspiração (Modelo)**  
    Os valores (coeficientes) de 0.5, 0.3, 0.1 e 0.2 no código são meramente ilustrativos.  
    A fórmula poderia ser algo como:
    """)
    st.latex(r'''
    \text{ET (litros/dia)} = [0.5 \times \text{Altura} + 0.3 \times \text{Diâmetro} 
    + 0.1 \times \text{Área de Copa} + 0.2 \times \text{LAI}] \times 10
    ''')
    
    st.markdown("""
    **4. Contraprova Experimental**  
    - O volume de água coletado (mL) em determinado período (horas) é convertido para litros.  
    - Ajusta-se para as 24 horas do dia (litros/dia).  
    - É feita uma média de várias medições experimentais para aumentar a robustez estatística.
    
    **5. Teste de Hipótese (Teste t de Student)**  
    - **Hipótese nula (H0):** A média experimental NÃO difere do valor previsto pelo modelo.  
    - **Hipótese alternativa (H1):** Existe diferença significativa entre o valor experimental e o do modelo.  
    - Se o p-valor < α (normalmente 0,05), rejeita-se H0 e conclui-se haver diferença significativa.
    
    **Interpretação:**  
    - Um p-valor muito pequeno sugere alta evidência de que o modelo difere dos dados experimentais.  
    - Um p-valor alto sugere que o modelo pode estar de acordo com as medições experimentais, 
      dentro da margem de erro e da variabilidade observada.
    
    **6. Possíveis Melhorias Futuras**  
    - Utilizar modelos de Machine Learning (CNN, Random Forest, etc.) para estimar evapotranspiração.  
    - Adicionar validações avançadas para os dados de entrada (ex.: intervalos realistas para altura, diâmetro).  
    - Incorporar condições climáticas (temperatura, umidade, radiação solar) para aprimorar o modelo.
    """)

