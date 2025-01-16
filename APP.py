import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------------
# 1. Armazenamento em session_state para persistência
# ---------------------------------------------------------------
if "resultados" not in st.session_state:
    st.session_state.resultados = []  # Evapotranspirações (modelo)
if "historico" not in st.session_state:
    st.session_state.historico = []  # Histórico de resultados
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ---------------------------------------------------------------
# 2. Funções auxiliares
# ---------------------------------------------------------------
def calculate_area_foliar_total(folhas_data, galhos):
    total_area = 0.0
    for largura_str, comprimento_str in folhas_data:
        try:
            largura = float(largura_str)
            comprimento = float(comprimento_str)
            total_area += (largura * comprimento) * galhos
        except ValueError:
            continue
    return total_area

def calculate_lai(area_foliar_total, area_copa):
    try:
        area_copa_val = float(area_copa)
        lai = area_foliar_total / area_copa_val
        return round(lai, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0

def predict_evapotranspiration(image, altura, diametro, copa, lai):
    et = (altura * 0.5 + diametro * 0.3 + copa * 0.1 + lai * 0.2) * 10
    return round(et, 2)

# ---------------------------------------------------------------
# 3. Cabeçalho e título com ícone/emoji
# ---------------------------------------------------------------
st.title("🌱 Estimativa de Evapotranspiração por CNN (Versão Ajustada)")

# ---------------------------------------------------------------
# 4. Carregar imagem
# ---------------------------------------------------------------
st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")
uploaded_file = st.file_uploader("📷 Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_container_width=True)

# ---------------------------------------------------------------
# 5. Dados dos espécimes
# ---------------------------------------------------------------
st.header("2) Insira as Variáveis Físicas dos Espécimes")
num_especies = st.number_input("Quantidade de Espécimes:", min_value=1, step=1, value=1)
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
# 6. Cálculo da Evapotranspiração (Modelo)
# ---------------------------------------------------------------
st.header("3) Cálculo da Evapotranspiração (Modelo)")
if st.button("💧 Calcular Evapotranspiração"):
    st.session_state.resultados = []
    if st.session_state.uploaded_image is None:
        st.error("⚠️ É necessário carregar uma imagem antes de calcular.")
    else:
        for i, (altura_str, diametro_str, copa_str, galhos, folhas_data) in enumerate(especies_data):
            try:
                altura_val = float(altura_str)
                diametro_val = float(diametro_str)
                copa_val = float(copa_str)

                # Validação simples de intervalos (exemplo)
                if not (0.5 <= altura_val <= 100):
                    st.warning(f"⚠️ Altura do Espécime {i+1} fora do intervalo plausível (0,5m - 100m).\n"
                               f"**Interpretação:** Alturas fora deste intervalo podem indicar erros de entrada ou medições incorretas. Por favor, verifique os valores inseridos.")
                else:
                    st.success(f"Altura do Espécime {i+1} está dentro do intervalo plausível.")

                aft = calculate_area_foliar_total(folhas_data, galhos)
                lai_val = calculate_lai(aft, copa_val)
                et_val = predict_evapotranspiration(
                    st.session_state.uploaded_image,
                    altura_val,
                    diametro_val,
                    copa_val,
                    lai_val
                )
                st.session_state.resultados.append(et_val)
                st.write(f"🌿 Evapotranspiração estimada para o Espécime {i+1}: {et_val} litros/dia")
                st.write("""
                **Explicação:** Este valor mostra a evapotranspiração estimada para cada espécime, calculada com base no modelo.

                **Interpretação:** A evapotranspiração estimada indica a quantidade de água que é liberada pelas folhas do espécime para a atmosfera por dia, em litros. Se o valor for muito alto ou muito baixo em comparação com os outros espécimes, pode indicar a necessidade de ajustar os parâmetros do modelo ou os dados de entrada.
                """)

                # Gravação no histórico
                st.session_state.historico.append(f"Espécime {i+1}: {et_val} litros/dia")
                
            except ValueError:
                st.error(f"⚠️ Espécime {i+1}: Insira valores numéricos válidos.")
                break

# ---------------------------------------------------------------
# 7. Histórico e Limpeza do Histórico
# ---------------------------------------------------------------
st.sidebar.header("🔄 Histórico de Cálculos")
if st.session_state.historico:
    for record in st.session_state.historico:
        st.sidebar.write(record)
else:
    st.sidebar.write("Nenhum cálculo realizado ainda.")

if st.sidebar.button("🧹 Limpar Histórico"):
    st.session_state.historico.clear()
    st.sidebar.write("✅ Histórico limpo!")

# ---------------------------------------------------------------
# 8. Seção Explicativa Expandida com Fórmulas e Interpretações
# ---------------------------------------------------------------
with st.expander("🔍 Explicação Técnica e Interpretação Detalhada"):
    st.markdown("### Cálculos e Fórmulas")
    st.markdown("**Área Foliar Total (AFT):**")
    st.latex(r'''
    \text{AFT} = \sum_{i=1}^{n} (\text{largura}_i \times \text{comprimento}_i) \times \text{galhos}
    ''')
    st.markdown("**Índice de Área Foliar (LAI):**")
    st.latex(r'''
    \text{LAI} = \frac{\text{AFT}}{\text{Área da Copa}}
    ''')
    st.markdown("**Evapotranspiração (Modelo):**")
    st.latex(r'''
    \text{ET (litros/dia)} = 
    [0.5 \times \text{Altura (m)} + 0.3 \times \text{Diâmetro (cm)} 
    + 0.1 \times \text{Área de Copa (m²)} + 0.2 \times \text{LAI}] \times 10
    ''')
    
    st.markdown("""
    ## Testes Estatísticos
    - **Teste t de Student:** Compara a média de um conjunto de dados com um valor hipotético.
    - **Teste de Mann-Whitney:** Teste não paramétrico que compara distribuições; útil quando os dados não seguem distribuição normal.
    - **Teste de Wilcoxon:** Teste não paramétrico que compara medianas de amostras pareadas ou diferenças; útil para dados não normais.
    - **Teste de Sinal:** Teste não paramétrico simples baseado no sinal das diferenças entre observações e um valor hipotético.
    - **Diferença Absoluta:** Calcula a diferença direta entre a média experimental e o valor do modelo sem inferência estatística.

    Cada teste possui requisitos e interpretações específicas. Escolha o teste adequado com base 
    no tamanho da amostra, distribuição dos dados e tipo de hipótese a ser testada.

    ## Melhores Práticas Finais
    - Validar dados de entrada: ex. altura entre 0,5m e 100m, diâmetro em faixas plausíveis, etc.
    - Incorporar dados climáticos (temperatura, umidade, radiação solar) para melhorar a precisão do modelo de evapotranspiração.
    - Utilizar modelos avançados, como **CNNs**, treinados com dados reais para estimar a evapotranspiração.
    - Fornecer múltiplas medições para cada espécime em diferentes condições para aumentar a robustez da contraprova.
    """)
