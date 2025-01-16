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

def predict_evapotranspiration(image, altura, diametro, copa, lai):
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

                # Validação simples de intervalos (exemplo)
                if not (0.5 <= altura_val <= 100):
                    st.warning(f"Altura do Espécime {i+1} fora do intervalo plausível (0,5m - 100m).\n"
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
                st.write(f"Evapotranspiração estimada para o Espécime {i+1}: {et_val} litros/dia")
                st.write("""
                **Explicação:** Este valor mostra a evapotranspiração estimada para cada espécime, calculada com base no modelo.

                **Interpretação:** A evapotranspiração estimada indica a quantidade de água que é liberada pelas folhas do espécime para a atmosfera por dia, em litros. Se o valor for muito alto ou muito baixo em comparação com os outros espécimes, pode indicar a necessidade de ajustar os parâmetros do modelo ou os dados de entrada.
                """)
            except ValueError:
                st.error(f"Espécime {i+1}: Insira valores numéricos válidos.")
                break

# ---------------------------------------------------------------
# 7. Contraprova Experimental
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

# ---------------------------------------------------------------
# 8. Escolha do Teste Estatístico e Comparação
# ---------------------------------------------------------------
st.header("5) Escolha o Teste Estatístico")
test_type = st.selectbox(
    "Escolha o teste estatístico para comparação:",
    ("Teste t de Student (1 amostra)",
     "Teste de Mann-Whitney",
     "Teste de Wilcoxon",
     "Teste de Sinal",
     "Diferença Absoluta")
)

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
                st.write("""
                **Explicação:** Estas são as medições experimentais de evapotranspiração convertidas para litros por dia.

                **Interpretação:** A evapotranspiração experimental reflete os valores observados diretamente. Comparar estas medições com as estimadas pelo modelo permite avaliar a precisão do modelo.
                """)
                media_experimental = np.mean(evap_exps)
                et_modelo = st.session_state.resultados[i]
                st.write(f"Média experimental: {media_experimental:.2f} litros/dia")
                st.write("""
                **Explicação:** Exibe a média das medições experimentais de evapotranspiração, que são coletadas diretamente.

                **Interpretação:** Esta média reflete o valor central das medições experimentais. Se a média experimental for semelhante ao valor previsto pelo modelo, isso sugere que o modelo está representando adequadamente a evapotranspiração. Caso contrário, pode ser necessário investigar as causas da discrepância.
                """)
                st.write(f"Valor previsto pelo modelo: {et_modelo:.2f} litros/dia")
                st.write("""
                **Explicação:** Exibe o valor previsto pelo modelo de evapotranspiração, calculado com base nas variáveis inseridas (altura, diâmetro, copa, LAI).

                **Interpretação:** Este é o valor que o modelo estima para a evapotranspiração do espécime. Comparar esse valor com as medições experimentais ajuda a avaliar a precisão do modelo. Uma diferença muito grande pode indicar que o modelo precisa ser aprimorado.
                """)

                valores_unicos = set(evap_exps)
                if len(evap_exps) < 2 or len(valores_unicos) < 2:
                    st.warning(
                        "Não é possível realizar o teste com uma única medição ou valores idênticos. "
                        "O teste exige pelo menos 2 valores distintos.\n"
                        "Recomenda-se coletar mais medições em diferentes condições para análises mais robustas."
                    )
                    diferenca_abs = abs(media_experimental - et_modelo)
                    st.write(f"Diferença (modelo x experimento): {diferenca_abs:.2f} litros/dia")
                    st.write("""
                    **Explicação:** Mostra a diferença absoluta entre o valor do modelo e o valor experimental.

                    **Interpretação:** A diferença absoluta fornece uma medida direta de quão distante o modelo está das medições experimentais. Se a diferença for pequena, isso indica que o modelo está ajustado corretamente; se for grande, pode ser necessário revisar o modelo ou os dados experimentais.
                    """)
                else:
                    p_value = None
                    if test_type == "Teste t de Student (1 amostra)":
                        stat, p_value = stats.ttest_1samp(evap_exps, et_modelo)
                        st.write(f"T-estatística: {stat:.4f}")
                        st.write("""
                        **Explicação:** A T-estatística quantifica a diferença entre a média experimental e o valor do modelo, normalizada pela variabilidade dos dados.

                        **Interpretação:** Quanto maior for o valor absoluto da T-estatística, mais significativa será a diferença entre a média experimental e o valor do modelo.
                        """)
                        st.write(f"P-valor: {p_value:.6f}")
                        st.write("""
                        **Explicação:** O P-valor indica a probabilidade de observarmos uma diferença tão extrema quanto a observada, assumindo que a hipótese nula seja verdadeira.

                        **Interpretação:** 
                        - **Se p < 0,05**: A diferença é estatisticamente significativa. Rejeitamos a hipótese nula de que não há diferença entre a média experimental e o valor do modelo.
                        - **Se p ≥ 0,05**: A diferença não é estatisticamente significativa. Não temos evidências suficientes para rejeitar a hipótese nula.
                        """)

                    elif test_type == "Teste de Mann-Whitney":
                        stat, p_value = stats.mannwhitneyu(evap_exps, [et_modelo]*len(evap_exps), alternative='two-sided')
                        st.write(f"Estatística U: {stat:.4f}")
                        st.write("""
                        **Explicação:** A Estatística U mede a diferença entre as distribuições dos dados experimentais e do modelo.

                        **Interpretação:** 
                        - **Valores U baixos** indicam uma grande diferença entre as distribuições.
                        - **Valores U altos** indicam uma menor diferença entre as distribuições.
                        """)
                        st.write(f"P-valor (Mann-Whitney): {p_value:.6f}")
                        st.write("""
                        **Explicação:** O P-valor determina se a diferença observada nas distribuições é significativa.

                        **Interpretação:** 
                        - **Se p < 0,05**: As distribuições são significativamente diferentes. Rejeitamos a hipótese nula de que as distribuições são idênticas.
                        - **Se p ≥ 0,05**: As distribuições não são significativamente diferentes. Não rejeitamos a hipótese nula.
                        """)

                    elif test_type == "Teste de Wilcoxon":
                        differences = np.array(evap_exps) - et_modelo
                        if np.all(differences == 0):
                            st.warning("Diferenças nulas impossibilitam o teste Wilcoxon.\n"
                                       "Isso ocorre quando todas as medições são idênticas ao valor do modelo.")
                        else:
                            try:
                                stat, p_value = stats.wilcoxon(differences)
                                st.write(f"Estatística W: {stat:.4f}")
                                st.write("""
                                **Explicação:** A Estatística W do Teste de Wilcoxon mede a soma das diferenças ordenadas das medições experimentais em relação ao modelo.

                                **Interpretação:** 
                                - **W alto**: Indica uma grande diferença entre as amostras pareadas.
                                - **W baixo**: Indica uma diferença menor.
                                """)
                                st.write(f"P-valor (Wilcoxon): {p_value:.6f}")
                                st.write("""
                                **Explicação:** O P-valor determina se a diferença observada nas medições pareadas é significativa.

                                **Interpretação:** 
                                - **Se p < 0,05**: A diferença é estatisticamente significativa. Rejeitamos a hipótese nula de que não há diferença nas medianas das amostras pareadas.
                                - **Se p ≥ 0,05**: A diferença não é estatisticamente significativa. Não rejeitamos a hipótese nula.
                                """)
                            except Exception as e:
                                st.error(f"Erro no teste de Wilcoxon: {e}")

                    elif test_type == "Teste de Sinal":
                        differences = np.array(evap_exps) - et_modelo
                        nonzero_diff = differences[differences != 0]
                        n = len(nonzero_diff)
                        if n == 0:
                            st.warning("Todos os valores experimentais são iguais ao valor do modelo.\n"
                                       "O teste de Sinal não pode ser aplicado quando todas as diferenças são nulas.")
                        else:
                            pos = np.sum(nonzero_diff > 0)
                            res = stats.binomtest(pos, n, 0.5)
                            st.write(f"Número de diferenças não-nulas: {n}")
                            st.write("""
                            **Explicação:** Este valor indica quantas das diferenças entre as medições experimentais e o modelo são positivas.

                            **Interpretação:** 
                            - **Maior número de sinais positivos**: Mais medições experimentais estão acima do modelo.
                            - **Maior número de sinais negativos**: Mais medições experimentais estão abaixo do modelo.
                            """)
                            st.write(f"Número de sinais positivos: {pos}")
                            st.write(f"P-valor (Teste de Sinal): {res.pvalue:.6f}")
                            st.write("""
                            **Explicação:** O P-valor determina se a proporção de sinais positivos é significativamente diferente de 0,5.

                            **Interpretação:** 
                            - **Se p < 0,05**: A proporção de sinais positivos é significativamente diferente de 0,5, indicando uma tendência estatística.
                            - **Se p ≥ 0,05**: Não há evidências suficientes para afirmar que a proporção de sinais positivos difere de 0,5.
                            """)

                    else:  # Diferença Absoluta
                        diferenca_abs = abs(media_experimental - et_modelo)
                        st.write(f"Diferença Absoluta (modelo x experimento): {diferenca_abs:.2f} litros/dia")
                        st.write("""
                        **Explicação:** Calcula a diferença direta entre o valor previsto pelo modelo e a média das medições experimentais.

                        **Interpretação:** 
                        - **Diferença pequena**: O modelo está ajustado corretamente às medições experimentais.
                        - **Diferença grande**: Pode ser necessário revisar o modelo ou os dados experimentais para melhorar a precisão.
                        """)

                    if p_value is not None:
                        alpha = 0.05
                        if p_value < alpha:
                            st.error("Diferença estatisticamente significativa (p < 0.05).")
                            st.write("""
                            **Interpretação:** A diferença entre o modelo e as medições experimentais é significativa. Isso sugere que o modelo não está representando adequadamente os dados experimentais e pode precisar de ajustes.
                            """)
                        else:
                            st.info("Diferença não é estatisticamente significativa (p ≥ 0.05).")
                            st.write("""
                            **Interpretação:** Não há diferença significativa entre o modelo e as medições experimentais. Isso sugere que o modelo está representando adequadamente a evapotranspiração observada.
                            """)

                except ValueError:
                    st.error(f"Espécime {i+1}: Insira valores experimentais válidos (números).")
    else:
        st.warning("É necessário primeiro calcular a evapotranspiração pelo modelo para todos os espécimes.")

# ---------------------------------------------------------------
# 9. Seção Explicativa Expandida com Fórmulas e Interpretações
# ---------------------------------------------------------------
with st.expander("Explicação Técnica e Interpretação Detalhada"):
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
    + 0.1 \times \text{Área da Copa (m²)} + 0.2 \times \text{LAI}] \times 10
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
