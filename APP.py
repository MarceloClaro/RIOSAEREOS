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

def predict_evapotranspiration(image, altura, diametro, copa, lai, umidade, radiacao, temperatura):
    # Cálculo básico incorporando efeitos climáticos:
    et = (altura * 0.5 + diametro * 0.3 + copa * 0.1 + lai * 0.2) * 10
    et *= (1 + (radiacao/1000))      # Aumento com mais radiação solar
    et *= (1 - (umidade/100))        # Diminui com alta umidade
    et *= (1 + (temperatura/100))    # Aumenta com maior temperatura
    return round(et, 2)

# ---------------------------------------------------------------
# 3. Cabeçalho e título
# ---------------------------------------------------------------
st.title("Estimativa de Evapotranspiração por CNN (Versão Aprimorada)")

# ---------------------------------------------------------------
# 4. Carregar imagem
# ---------------------------------------------------------------
st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")
uploaded_file = st.file_uploader("Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file)
    st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_container_width=True)

# ---------------------------------------------------------------
# 5. Entrada de dados dos espécimes e clima
# ---------------------------------------------------------------
st.header("2) Insira as Variáveis Físicas dos Espécimes")

# Validação básica de intervalos
def validar_intervalo(valor, minimo, maximo, nome_var):
    if valor < minimo or valor > maximo:
        st.warning(f"O valor de {nome_var} ({valor}) está fora do intervalo recomendado [{minimo}, {maximo}].")

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

st.header("Variáveis Climáticas (Opcional)")
umidade = st.number_input("Umidade Relativa (%)", min_value=0.0, max_value=100.0, value=50.0)
radiacao = st.number_input("Radiação Solar (W/m²)", min_value=0.0, value=200.0)
temperatura = st.number_input("Temperatura (°C)", min_value=-50.0, max_value=60.0, value=25.0)

# ---------------------------------------------------------------
# 6. Cálculo da evapotranspiração (modelo)
# ---------------------------------------------------------------
st.header("3) Cálculo da Evapotranspiração (Modelo)")
if st.button("Calcular Evapotranspiração"):
    st.session_state.resultados = []
    if st.session_state.uploaded_image is None:
        st.error("É necessário carregar uma imagem antes de calcular.")
    else:
        for i, (altura_str, diametro_str, copa_str, galhos, folhas_data) in enumerate(especies_data):
            try:
                altura_val = float(altura_str)
                diametro_val = float(diametro_str)
                copa_val = float(copa_str)

                # Validar intervalos de valores
                validar_intervalo(altura_val, 0.5, 100, "altura")
                validar_intervalo(diametro_val, 1, 500, "diâmetro")
                validar_intervalo(copa_val, 0.1, 1000, "área da copa")

                aft = calculate_area_foliar_total(folhas_data, galhos)
                lai_val = calculate_lai(aft, copa_val)
                et_val = predict_evapotranspiration(
                    st.session_state.uploaded_image,
                    altura_val,
                    diametro_val,
                    copa_val,
                    lai_val,
                    umidade,
                    radiacao,
                    temperatura
                )

                st.session_state.resultados.append(et_val)
                st.success(f"Espécime {i+1}: {et_val} litros/dia (modelo)")

            except ValueError:
                st.error(f"Espécime {i+1}: Insira valores numéricos válidos.")
                break

# ---------------------------------------------------------------
# 7. Contraprova Experimental e Seleção de Teste Estatístico
# ---------------------------------------------------------------
st.header("4) Contraprova Experimental com Múltiplas Medições")
num_experimentos = st.number_input("Quantidade de medições experimentais para cada Espécime:", min_value=1, step=1, value=1)

contraprovas = {}
for i in range(num_especies):
    st.subheader(f"Espécime {i+1} - Valores Experimentais (mL)")
    valores_experimentais = []
    for j in range(num_experimentos):
        val = st.text_input(f"Medição {j+1} (mL) - Espécime {i+1}:", "0")
        valores_experimentais.append(val)
    contraprovas[i] = valores_experimentais

tempo_coleta_horas = st.number_input("Tempo (horas) de coleta para cada medição:", min_value=1, step=1, value=24)

st.subheader("Selecione o Teste Estatístico a Ser Realizado")
teste_selecionado = st.selectbox("Teste Estatístico", ["Teste t de Student", "Teste de Wilcoxon", "Teste de Sinal"])

if st.button("Comparar com a Contraprova"):
    if len(st.session_state.resultados) == num_especies:
        for i in range(num_especies):
            st.markdown(f"---\n**Espécime {i+1}:**")
            try:
                valores_exp_float = [float(x) for x in contraprovas[i]]
                evap_exps = []
                for vol_mL in valores_exp_float:
                    vol_L = vol_mL / 1000.0
                    vol_L_dia = vol_L / (tempo_coleta_horas / 24.0)
                    evap_exps.append(vol_L_dia)

                st.write("Medições (litros/dia):", [f"{v:.2f}" for v in evap_exps])
                media_experimental = np.mean(evap_exps)
                et_modelo = st.session_state.resultados[i]
                st.write(f"Média experimental: {media_experimental:.2f} litros/dia")
                st.write(f"Valor previsto pelo modelo: {et_modelo:.2f} litros/dia")

                valores_unicos = set(evap_exps)
                if len(evap_exps) < 2 or len(valores_unicos) < 2:
                    st.warning(
                        "Não é possível realizar o teste estatístico com uma única medição ou valores idênticos. "
                        "Forneça ao menos 2 valores distintos para análise estatística."
                    )
                    diferenca_abs = abs(media_experimental - et_modelo)
                    st.write(f"Diferença (modelo x experimento): {diferenca_abs:.2f} litros/dia")
                else:
                    # Seleção e execução do teste estatístico
                    if teste_selecionado == "Teste t de Student":
                        t_stat, p_value = stats.ttest_1samp(evap_exps, et_modelo)
                        st.write(f"Teste t -> T-estatística: {t_stat:.4f}, P-valor: {p_value:.6f}")
                        alpha = 0.05
                        if p_value < alpha:
                            st.error("Diferença estatisticamente significativa (Teste t).")
                        else:
                            st.info("Diferença não significativa (Teste t).")
                    elif teste_selecionado == "Teste de Wilcoxon":
                        try:
                            stat, p_value = stats.wilcoxon(np.array(evap_exps) - et_modelo)
                            st.write(f"Wilcoxon -> Estatística: {stat:.4f}, P-valor: {p_value:.6f}")
                            alpha = 0.05
                            if p_value < alpha:
                                st.error("Diferença estatisticamente significativa (Wilcoxon).")
                            else:
                                st.info("Diferença não significativa (Wilcoxon).")
                        except ValueError as e:
                            st.warning(f"Não foi possível realizar o Teste de Wilcoxon: {e}")
                    elif teste_selecionado == "Teste de Sinal":
                        # Implementação simples do teste de sinal
                        signs = [1 if x > et_modelo else 0 for x in evap_exps]
                        n = len(signs)
                        k = sum(signs)
                        p_value = stats.binom_test(k, n, 0.5)
                        st.write(f"Teste de Sinal -> Sucessos: {k} de {n}, P-valor: {p_value:.6f}")
                        alpha = 0.05
                        if p_value < alpha:
                            st.error("Diferença estatisticamente significativa (Teste de Sinal).")
                        else:
                            st.info("Diferença não significativa (Teste de Sinal).")

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
    ...
    ## 3. Evapotranspiração (Modelo)
    ...
    ## 4. Contraprova Experimental
    ...
    ## 5. Testes Estatísticos
    - **Teste t de Student**: Compara a média de um conjunto de dados com um valor hipotético.
    - **Teste de Wilcoxon**: Teste não paramétrico que compara medianas; útil para dados pareados ou não normais.
    - **Teste de Sinal**: Teste não paramétrico simples baseado em sinais das diferenças.
    - Cada teste possui requisitos e interpretações específicas.
    
    ### Escolha do Teste Estatístico
    - Verifique a **distribuição** e **tamanho da amostra** para escolher o teste adequado.
    - Se os dados são normalmente distribuídos e há variabilidade, use o Teste t.
    - Se a normalidade não pode ser assumida ou a amostra é pequena, considere o Teste de Wilcoxon ou o Teste de Sinal.
    
    ## 6. Diferença Absoluta
    ...
    
    ## 7. Boas Práticas Finais
    - Incorporar variáveis climáticas para maior acurácia.
    - Ajustar o modelo de evapotranspiração possivelmente com CNN.
    - Validar intervalos plausíveis para entradas.
    - Coletar múltiplas medições para robustez estatística.
    """)
