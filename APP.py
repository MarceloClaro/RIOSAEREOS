import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------------------------------------------------------------
# 1. Armazenamento em session_state para persistência
# ---------------------------------------------------------------
if "resultados" not in st.session_state:
    st.session_state.resultados = []
if "historico" not in st.session_state:
    st.session_state.historico = []
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
        if area_copa_val <= 0:
            return 0.0
        lai = area_foliar_total / area_copa_val
        return round(lai, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0

def predict_evapotranspiration(image, altura, diametro, copa, lai, temperatura, umidade, radiacao, vento):
    # Fórmula ajustada para incluir variáveis climáticas
    et = (altura * 0.3 + diametro * 0.2 + copa * 0.1 + lai * 0.2 + temperatura * 0.1 + umidade * 0.05 + radiacao * 0.03 + vento * 0.02) * 10
    return round(et, 2)

# ---------------------------------------------------------------
# 3. Cabeçalho e título com ícone/emoji
# ---------------------------------------------------------------
st.title("🌱 Estimativa de Evapotranspiração (Rios Aéreos)")
st.markdown("Aplicação para estimar a evapotranspiração de espécimes arbóreos ou arbustivos, comparando resultados de um modelo empírico simplificado com medições experimentais e realizando análises estatísticas.")

# ---------------------------------------------------------------
# 4. Carregar imagem
# ---------------------------------------------------------------
st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")
uploaded_file = st.file_uploader("📷 Faça o upload da imagem (formato JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Erro ao carregar a imagem: {e}")

# ---------------------------------------------------------------
# 5. Dados dos espécimes
# ---------------------------------------------------------------
st.header("2) Insira as Variáveis Físicas dos Espécimes")
num_especies = st.number_input("🔢 Quantidade de Espécimes:", min_value=1, step=1, value=1)
especies_data = []
for i in range(num_especies):
    st.subheader(f"🌿 Espécime {i+1}")
    altura = st.text_input(f"📏 Altura (m) - Espécime {i+1}:", "0", key=f"altura_{i}")
    diametro = st.text_input(f"📐 Diâmetro do Tronco (cm) - Espécime {i+1}:", "0", key=f"diametro_{i}")
    copa = st.text_input(f"🌳 Área da Copa (m²) - Espécime {i+1}:", "0", key=f"copa_{i}")
    galhos = st.number_input(f"🌿 Quantidade de Galhos - Espécime {i+1}:", min_value=1, step=1, value=1, key=f"galhos_{i}")
    folhas_data = []
    for j in range(galhos):
        st.markdown(f"**🌱 Galho {j+1} - Espécime {i+1}**")
        largura_folha = st.text_input(f"Largura da Folha (cm) - Galho {j+1}:", "0", key=f"largura_folha_{i}_{j}")
        comprimento_folha = st.text_input(f"Comprimento da Folha (cm) - Galho {j+1}:", "0", key=f"comprimento_folha_{i}_{j}")
        folhas_data.append((largura_folha, comprimento_folha))
    especies_data.append((altura, diametro, copa, galhos, folhas_data))

# ---------------------------------------------------------------
# 6. Variáveis Climáticas
# ---------------------------------------------------------------
st.header("3) Insira as Variáveis Climáticas")
col_clima1, col_clima2 = st.columns(2)

with col_clima1:
    temperatura = st.text_input("🌡️ Temperatura (°C):", "25", key="temperatura")
    umidade = st.text_input("💧 Umidade Relativa (%):", "60", key="umidade")
    radiacao = st.text_input("☀️ Radiação Solar (MJ/m²):", "20", key="radiacao")

with col_clima2:
    vento = st.text_input("🌬️ Velocidade do Vento (m/s):", "5", key="vento")
    # Adicione mais variáveis climáticas se necessário

# ---------------------------------------------------------------
# 7. Cálculo da Evapotranspiração (Modelo)
# ---------------------------------------------------------------
st.header("4) Cálculo da Evapotranspiração (Modelo)")
if st.button("💧 Calcular Evapotranspiração"):
    st.session_state.resultados = []
    if st.session_state.uploaded_image is None:
        st.error("⚠️ É necessário carregar uma imagem antes de calcular.")
    else:
        # Tentar converter as variáveis climáticas
        try:
            temperatura_val = float(temperatura)
            umidade_val = float(umidade)
            radiacao_val = float(radiacao)
            vento_val = float(vento)
        except ValueError:
            st.error("⚠️ Insira valores numéricos válidos para as variáveis climáticas.")
            temperatura_val = umidade_val = radiacao_val = vento_val = 0.0

        for i, (altura_str, diametro_str, copa_str, galhos, folhas_data) in enumerate(especies_data):
            try:
                altura_val = float(altura_str)
                diametro_val = float(diametro_str)
                copa_val = float(copa_str)

                if not (0.5 <= altura_val <= 100):
                    st.warning(f"⚠️ Altura do Espécime {i+1} fora do intervalo plausível (0,5m - 100m).\n"
                               "**Interpretação:** Verifique a altura inserida.")
                else:
                    st.success(f"✅ Altura do Espécime {i+1} plausível.")

                aft = calculate_area_foliar_total(folhas_data, galhos)
                lai_val = calculate_lai(aft, copa_val)
                et_val = predict_evapotranspiration(
                    st.session_state.uploaded_image,
                    altura_val,
                    diametro_val,
                    copa_val,
                    lai_val,
                    temperatura_val,
                    umidade_val,
                    radiacao_val,
                    vento_val
                )
                st.session_state.resultados.append(et_val)
                st.write(f"🌿 **Evapotranspiração estimada para o Espécime {i+1}:** {et_val} litros/dia")
                st.write("""
                **Explicação:** Este valor mostra a evapotranspiração estimada para cada espécime, calculada com base no modelo.
    
                **Interpretação:** Indica a água liberada pelas folhas por dia.
                """)
                st.session_state.historico.append((i+1, et_val))
            except ValueError:
                st.error(f"⚠️ Espécime {i+1}: Insira valores numéricos válidos.")
                break

# ---------------------------------------------------------------
# 8. Contraprova Experimental
# ---------------------------------------------------------------
st.header("5) Contraprova Experimental com Múltiplas Medições")
num_experimentos = st.number_input("🔢 Quantidade de medições experimentais para cada Espécime:", min_value=1, step=1, value=1)
contraprovas = {}
for i in range(num_especies):
    st.subheader(f"🌿 Espécime {i+1} - Valores Experimentais (mL)")
    valores_experimentais = []
    for j in range(num_experimentos):
        val = st.text_input(
            f"Medição {j+1} (mL) - Espécime {i+1}:",
            "0",
            key=f"medicao_{i}_{j}"
        )
        valores_experimentais.append(val)
    contraprovas[i] = valores_experimentais
tempo_coleta_horas = st.number_input("⏱️ Tempo (horas) de coleta para cada medição:", min_value=1, step=1, value=24, key="tempo_coleta")

# ---------------------------------------------------------------
# 9. Escolha do Teste Estatístico e Comparação
# ---------------------------------------------------------------
st.header("6) Escolha o Teste Estatístico")
test_type = st.selectbox(
    "📊 Escolha o teste estatístico para comparação:",
    ("Teste t de Student (1 amostra)",
     "Teste de Mann-Whitney",
     "Teste de Wilcoxon",
     "Teste de Sinal",
     "Diferença Absoluta")
)

if st.button("🔄 Comparar com a Contraprova"):
    if len(st.session_state.resultados) == num_especies:
        all_experimental_means = []
        all_model_predictions = []
        all_residuals = []

        for i in range(num_especies):
            st.markdown(f"---")
            st.subheader(f"🔎 Análise Detalhada - Espécime {i+1}")
            try:
                # Converter valores experimentais para float
                valores_exp_float = [float(x) for x in contraprovas[i]]
                evap_exps = []
                for vol_mL in valores_exp_float:
                    vol_L = vol_mL / 1000.0
                    vol_L_dia = vol_L / (tempo_coleta_horas / 24.0)
                    evap_exps.append(vol_L_dia)

                st.write("🌡️ **Medições (litros/dia):**", [f"{v:.2f}" for v in evap_exps])
                st.write("""
                **Explicação:** Estas são as medições experimentais de evapotranspiração convertidas para litros por dia, baseadas nas leituras de mL e no tempo de coleta informado.
    
                **Interpretação:** A evapotranspiração experimental reflete os valores observados diretamente. Comparar estas medições com as estimativas do modelo permite avaliar a precisão do modelo e a variabilidade inerente às medições de campo.
                """)
                media_experimental = np.mean(evap_exps)
                et_modelo = st.session_state.resultados[i]
                st.write(f"📊 **Média experimental:** {media_experimental:.2f} litros/dia")
                st.write("""
                **Explicação:** Exibe a média das medições experimentais de evapotranspiração, que são coletadas diretamente.
    
                **Interpretação:** Esta média reflete o valor central das medições experimentais. Se a média experimental for semelhante ao valor previsto pelo modelo, isso sugere que o modelo está representando adequadamente a evapotranspiração. Caso contrário, pode ser necessário investigar as causas da discrepância.
                """)
                st.write(f"🔮 **Valor previsto pelo modelo:** {et_modelo:.2f} litros/dia")
                st.write("""
                **Explicação:** Exibe o valor previsto pelo modelo empírico simplificado de evapotranspiração, calculado com base nas variáveis inseridas (altura, diâmetro, copa, LAI, temperatura, umidade, radiação, vento) e seus respectivos pesos.
    
                **Interpretação:** Este é o valor que o modelo estima para a evapotranspiração do espécime. Comparar esse valor com as medições experimentais, juntamente com a análise estatística, ajuda a avaliar a adequação do modelo para as condições específicas observadas. Uma diferença significativa pode indicar limitações do modelo ou a necessidade de ajustes nos coeficientes.
                """)
                all_experimental_means.append(media_experimental)
                all_model_predictions.append(et_modelo)
                all_residuals.append(media_experimental - et_modelo)

                valores_unicos = set(evap_exps)
                if len(evap_exps) < 2 or len(valores_unicos) < 2:
                    st.warning(
                        "⚠️ Análise estatística limitada: Não é possível realizar testes de hipótese robustos com uma única medição ou valores experimentais idênticos. "
                        "A maioria dos testes exige variabilidade nos dados. Considere coletar múltiplas medições sob diferentes condições.\n"
                        "📈 **Análise Descritiva:**"
                    )
                    diferenca_abs = abs(media_experimental - et_modelo)
                    st.write(f"📉 **Diferença Absoluta (modelo vs. experimento):** {diferenca_abs:.2f} litros/dia")
                    st.write("""
                    **Explicação:** Mostra a magnitude da diferença entre a média experimental e o valor previsto pelo modelo, sem considerar a variabilidade dos dados experimentais.
    
                    **Interpretação:** Uma diferença absoluta pequena sugere que o modelo está razoavelmente próximo da média observada neste caso específico.
                    """)
                else:
                    p_value = None # Initialize p_value
                    st.markdown("📈 **Análise Estatística Inferencial (para este espécime):**")
                    if test_type == "Teste t de Student (1 amostra)":
                        # Assuming evap_exps represents a sample from a population with mean et_modelo under the null
                        try:
                            stat, p_value = stats.ttest_1samp(evap_exps, et_modelo)
                            st.write(f"📈 **T-estatística (Teste t de 1 Amostra):** {stat:.4f}")
                            st.write("""
                            **Explicação:** A T-estatística quantifica a diferença entre a média da amostra experimental (as medições coletadas para este espécime) e o valor hipotético (a previsão do modelo para este espécime), normalizada pela variabilidade estimada da amostra.
    
                            **Interpretação:** Um valor absoluto alto da T-estatística sugere que a média experimental observada está distante do valor previsto pelo modelo, considerando a dispersão das medições.
                        """)
                            st.write(f"🔢 **P-valor (Teste t de 1 Amostra):** {p_value:.6f}")
                            st.write("""
                            **Explicação:** O P-valor é a probabilidade de observar uma T-estatística tão extrema (ou mais) quanto a calculada, *se* a evapotranspiração verdadeira do espécime for igual ao valor previsto pelo modelo (hipótese nula).
    
                            **Interpretação:** 
                            - **Se p < alpha (geralmente 0.05):** Rejeitamos a hipótese nula. Há evidência estatística de que a média experimental é significativamente diferente do valor previsto pelo modelo.
                            - **Se p >= alpha:** Não temos evidência estatística forte para rejeitar a hipótese nula. A diferença observada pode ser devida ao acaso amostral.
                        """)
                        except Exception as e:
                             st.warning(f"⚠️ Não foi possível executar o Teste t de Student: {e}. Verifique se há dados suficientes e variabilidade na amostra experimental.")


                    elif test_type == "Teste de Mann-Whitney":
                        # Comparing the distribution of experimental values to a constant (the model's prediction)
                        # This application of Mann-Whitney is a bit unconventional as it typically compares two samples.
                        # A more appropriate non-parametric test for comparing a sample to a theoretical median/value
                        # would be the Wilcoxon Signed-Rank Test (if applicable to paired differences) or a permutation test.
                        # However, following the user's initial selection:
                        st.warning("⚠️ Nota sobre o Teste de Mann-Whitney: Este teste é tipicamente usado para comparar duas amostras independentes. Comparar uma amostra (medições experimentais) com um valor constante (previsão do modelo) não é a aplicação padrão e pode não ser a mais interpretável. Considere o Teste de Wilcoxon para dados pareados ou testes baseados em bootstrap/permutação para uma análise não paramétrica mais robusta.")
                        # To technically run it by comparing the sample to a 'sample' of the model prediction:
                        try:
                            # Create a synthetic sample of the model prediction for comparison
                            model_sample = [et_modelo] * len(evap_exps)
                            stat, p_value = stats.mannwhitneyu(evap_exps, model_sample, alternative='two-sided')
                            st.write(f"📉 **Estatística U (Teste de Mann-Whitney adaptado):** {stat:.4f}")
                            st.write("""
                            **Explicação:** A Estatística U mede a diferença entre as "posições relativas" dos dados experimentais comparados com o valor previsto pelo modelo. Um U baixo indica que as medições experimentais tendem a ser menores que o valor previsto, e um U alto (próximo de len(evap_exps)*len(model_sample)) indica que tendem a ser maiores.
    
                            **Interpretação:** Um valor U que difere significativamente do esperado sob a hipótese nula (tipicamente U = (n1*n2)/2) sugere uma diferença nas medianas ou distribuições. O P-valor ajuda a formalizar essa conclusão.
                        """)
                            st.write(f"🔢 **P-valor (Teste de Mann-Whitney adaptado):** {p_value:.6f}")
                            st.write("""
                            **Explicação:** O P-valor é a probabilidade de observar uma Estatística U tão extrema (ou mais) quanto a calculada, *se* as medições experimentais forem distribuídas de forma semelhante em torno do valor previsto pelo modelo.
    
                            **Interpretação:** 
                            - **Se p < alpha:** Rejeitamos a hipótese nula (que as medianas/distribuições são semelhantes).
                            - **Se p >= alpha:** Não temos evidência estatística forte para rejeitar a hipótese nula.
                        """)
                        except Exception as e:
                             st.warning(f"⚠️ Não foi possível executar o Teste de Mann-Whitney: {e}. Verifique se há dados suficientes.")


                    elif test_type == "Teste de Wilcoxon":
                        # Wilcoxon Signed-Rank Test compares the median of differences to zero.
                        # Here, the 'differences' are evap_exps - et_modelo. It tests if the median difference is zero.
                        differences = np.array(evap_exps) - et_modelo
                        # Check if all differences are zero, which makes the test impossible
                        if np.all(differences == 0):
                            st.warning("⚠️ Teste de Wilcoxon impossibilitado: Todas as diferenças entre as medições experimentais e o valor do modelo são zero. Não há variabilidade para o teste analisar.")
                        else:
                            try:
                                # The test requires non-zero differences
                                nonzero_differences = differences[differences != 0]
                                # A two-sided test is appropriate for checking if the median difference is not zero
                                stat, p_value = stats.wilcoxon(nonzero_differences, alternative='two-sided') # Use nonzero_differences
                                st.write(f"📈 **Estatística W (Teste de Wilcoxon):** {stat:.4f}")
                                st.write("""
                                **Explicação:** A Estatística W (ou V, dependendo da implementação) do Teste de Wilcoxon Signed-Rank é baseada nas postos (ranks) das diferenças absolutas entre cada medição experimental e o valor previsto pelo modelo. Soma os postos das diferenças positivas ou negativas.
    
                                **Interpretação:** Um valor W pequeno (ou grande, dependendo da definição) sugere que as diferenças tendem a ter o mesmo sinal, indicando que as medições experimentais são consistentemente maiores ou menores que a previsão do modelo.
                                """)
                                st.write(f"🔢 **P-valor (Teste de Wilcoxon):** {p_value:.6f}")
                                st.write("""
                                **Explicação:** O P-valor é a probabilidade de observar uma Estatística W tão extrema (ou mais) quanto a calculada, *se* a mediana das diferenças entre as medições experimentais e o valor previsto pelo modelo for zero (hipótese nula).
    
                                **Interpretação:** 
                                - **Se p < alpha:** Rejeitamos a hipótese nula. Há evidência estatística de que a mediana das diferenças não é zero, sugerindo uma diferença sistemática entre as medições e o modelo.
                                - **Se p >= alpha:** Não temos evidência estatística forte para rejeitar a hipótese nula. A mediana das diferenças pode ser zero.
                                """)
                            except Exception as e:
                                st.error(f"⚠️ Erro ao executar o Teste de Wilcoxon: {e}. Verifique se há dados suficientes e variabilidade nas diferenças não-nulas.")

                    elif test_type == "Teste de Sinal":
                        # Tests if the median of the differences is different from zero based on the sign of the differences.
                        # It's less powerful than Wilcoxon as it only uses the sign, not the magnitude of differences.
                        differences = np.array(evap_exps) - et_modelo
                        nonzero_diff = differences[differences != 0]
                        n = len(nonzero_diff)
                        if n == 0:
                            st.warning("⚠️ Teste de Sinal impossibilitado: Todas as diferenças entre as medições experimentais e o valor do modelo são zero. Não há sinais para analisar.")
                        else:
                            pos = np.sum(nonzero_diff > 0)
                            neg = np.sum(nonzero_diff < 0)
                            # The null hypothesis is that P(positive difference) = P(negative difference) = 0.5
                            # We can use a binomial test for this.
                            try:
                                res = stats.binomtest(pos, n, 0.5, alternative='two-sided')
                                st.write(f"📊 **Número de diferenças não-nulas:** {n}")
                                st.write("""
                                **Explicação:** Este valor indica quantas das diferenças entre as medições experimentais e o valor previsto pelo modelo não são zero. O Teste de Sinal ignora as diferenças iguais a zero.
        
                                **Interpretação:** O número de diferenças não-nulas determina o tamanho da amostra efetiva para o Teste de Sinal.
                                """)
                                st.write(f"📈 **Número de sinais positivos:** {pos}")
                                st.write(f"📉 **Número de sinais negativos:** {neg}")
                                st.write("""
                                **Explicação:** Conta quantas medições experimentais foram maiores (sinal positivo) ou menores (sinal negativo) que o valor previsto pelo modelo.
        
                                **Interpretação:** Sob a hipótese nula (mediana das diferenças é zero), esperaríamos aproximadamente um número igual de sinais positivos e negativos. Um desequilíbrio sugere que as medições tendem a estar consistentemente acima ou abaixo da previsão do modelo.
                                """)
                                st.write(f"🔢 **P-valor (Teste de Sinal - Binomial):** {res.pvalue:.6f}")
                                st.write("""
                                **Explicação:** O P-valor é a probabilidade de observar uma proporção de sinais positivos (ou negativos) tão extrema (ou mais) quanto a calculada, *se* a verdadeira mediana das diferenças for zero (ou seja, os sinais positivo e negativo são igualmente prováveis).
        
                                **Interpretação:** 
                                - **Se p < alpha:** Rejeitamos a hipótese nula. Há evidência estatística de um desequilíbrio significativo entre sinais positivos e negativos, sugerindo que a mediana das diferenças não é zero.
                                - **Se p >= alpha:** Não temos evidência estatística forte para rejeitar a hipótese nula. O desequilíbrio observado pode ser devido ao acaso.
                                """)
                            except Exception as e:
                                st.error(f"⚠️ Erro ao executar o Teste de Sinal: {e}. Verifique se há dados suficientes.")

                    else:  # Diferença Absoluta (já tratada na seção de análise descritiva para caso de N<2)
                         if len(evap_exps) >= 2 and len(valores_unicos) >= 2:
                            diferenca_abs = abs(media_experimental - et_modelo)
                            st.write(f"📉 **Diferença Absoluta (modelo vs. experimento):** {diferenca_abs:.2f} litros/dia")
                            st.write("""
                            **Explicação:** Calcula a diferença direta entre a média experimental e o valor previsto pelo modelo. Esta métrica fornece uma medida de erro simples, mas não considera a variabilidade dentro das medições experimentais.
        
                            **Interpretação:** Uma diferença absoluta pequena indica que o modelo está próximo da média observada. É uma métrica útil para resumir o erro, mas a análise estatística inferencial (acima) é necessária para avaliar se essa diferença é estatisticamente significativa.
                            """)


                    # Conclusion based on p-value if available
                    if p_value is not None:
                        alpha = 0.05 # Standard significance level
                        st.markdown("---")
                        st.subheader("Conclusão Estatística:")
                        if p_value is not None and p_value < alpha: # Check if p_value was set
                            st.error("❌ **Resultado Estatisticamente Significativo (p < 0.05).**")
                            st.write("""
                            **Interpretação (para este espécime):** A análise estatística indica que a diferença observada entre as medições experimentais e o valor previsto pelo modelo para este espécime **é estatisticamente significativa** ao nível de significância de 5%. Isso sugere que é improvável que a diferença observada seja apenas devido ao acaso amostral. Portanto, pode haver uma limitação no modelo em prever a evapotranspiração para este espécime sob as condições observadas, ou podem existir fatores não considerados pelo modelo que influenciam a evapotranspiração real.
                            """)
                        elif p_value is not None:
                            st.info("✅ **Resultado Não Estatisticamente Significativo (p ≥ 0.05).**")
                            st.write("""
                            **Interpretação:** A análise estatística indica que a diferença observada entre as medições experimentais e o valor previsto pelo modelo para este espécime **não é estatisticamente significativa** ao nível de significância de 5%. Isso significa que não há evidência estatística forte para concluir que a média experimental difere do valor previsto pelo modelo. A diferença observada pode ser explicada pela variabilidade natural ou pelo acaso amostral. Isso sugere que o modelo pode ser adequado para prever a evapotranspiração para este espécime sob as condições observadas, embora a ausência de significância não prove que o modelo está "correto", apenas que os dados atuais não fornecem evidência suficiente para rejeitá-lo.
                            """)
                        else:
                             st.info("ℹ️ Teste estatístico inferencial não aplicável ou não produziu P-valor para este espécime devido a dados insuficientes/idênticos ou erro no teste.")


            except ValueError:
                st.error(f"⚠️ Espécime {i+1}: Insira valores experimentais válidos (números).")
        
        # --- Análise Global do Modelo (após o loop dos espécimes) ---
        if len(all_experimental_means) > 1 and len(all_model_predictions) > 1: # Precisa de pelo menos 2 pontos para métricas globais e regressão
            st.markdown("---")
            st.header("🌍 Análise Global do Desempenho do Modelo")

            # Converter para arrays numpy para cálculos
            exp_means_np = np.array(all_experimental_means)
            model_preds_np = np.array(all_model_predictions)
            residuals_np = np.array(all_residuals)

            # Métricas Globais
            global_rmse = np.sqrt(mean_squared_error(exp_means_np, model_preds_np))
            global_mae = mean_absolute_error(exp_means_np, model_preds_np)
            global_r2 = r2_score(exp_means_np, model_preds_np)

            st.subheader("📊 Métricas Globais de Desempenho")
            st.write(f"**Root Mean Squared Error (RMSE) Global:** {global_rmse:.4f} litros/dia")
            st.write(f"**Mean Absolute Error (MAE) Global:** {global_mae:.4f} litros/dia")
            st.write(f"**R-squared (R²) Global:** {global_r2:.4f}")
            st.write("""
            **Interpretação das Métricas Globais:**
            - **RMSE e MAE:** Medem o erro médio das previsões do modelo em relação às médias experimentais de todos os espécimes. Valores menores indicam melhor ajuste.
            - **R²:** Indica a proporção da variância nas médias experimentais que é explicada pelo modelo. Um valor próximo de 1 sugere que o modelo explica bem a variabilidade observada. Um R² baixo ou negativo indica um mau ajuste.
            """)

            # Análise de Regressão Linear
            st.subheader("📈 Análise de Regressão: Experimental vs. Modelo")
            slope, intercept, r_value, p_value_reg, std_err = stats.linregress(model_preds_np, exp_means_np)
            
            fig_reg, ax_reg = plt.subplots()
            ax_reg.scatter(model_preds_np, exp_means_np, label='Dados (Espécimes)', color='blue', alpha=0.7)
            ax_reg.plot(model_preds_np, intercept + slope * model_preds_np, 'r', label=f'Linha de Regressão\ny={slope:.2f}x + {intercept:.2f}\nR²={r_value**2:.2f}')
            ax_reg.plot([min(model_preds_np.min(), exp_means_np.min()), max(model_preds_np.max(), exp_means_np.max())], 
                        [min(model_preds_np.min(), exp_means_np.min()), max(model_preds_np.max(), exp_means_np.max())], 
                        'k--', label='Linha 1:1 (Ideal)')
            ax_reg.set_xlabel("ET Prevista pelo Modelo (litros/dia)")
            ax_reg.set_ylabel("ET Média Experimental (litros/dia)")
            ax_reg.set_title("Regressão: ET Experimental vs. ET Modelo")
            ax_reg.legend()
            ax_reg.grid(True)
            st.pyplot(fig_reg)
            st.write(f"**Intercepto:** {intercept:.4f}, **Inclinação (Slope):** {slope:.4f}, **P-valor da Regressão:** {p_value_reg:.4f}")
            st.write("""
            **Interpretação da Regressão:**
            - **Linha 1:1 (Ideal):** Se os pontos estiverem próximos desta linha, o modelo prevê com precisão.
            - **Intercepto:** Idealmente próximo de 0. Um intercepto significativamente diferente de 0 indica um viés sistemático (o modelo consistentemente superestima ou subestima).
            - **Inclinação (Slope):** Idealmente próximo de 1. Uma inclinação < 1 sugere que o modelo subestima valores altos e superestima baixos (ou vice-versa se > 1).
            - **R² da Regressão:** Similar ao R² global, mede o quão bem a linha de regressão se ajusta aos dados.
            - **P-valor da Regressão:** Testa a significância da relação linear. Um P-valor baixo (<0.05) sugere que a relação linear é estatisticamente significativa.
            """)

            # Análise de Resíduos
            st.subheader("📉 Análise de Resíduos")
            fig_res, ax_res = plt.subplots()
            ax_res.scatter(model_preds_np, residuals_np, color='green', alpha=0.7)
            ax_res.axhline(0, color='red', linestyle='--')
            ax_res.set_xlabel("ET Prevista pelo Modelo (litros/dia)")
            ax_res.set_ylabel("Resíduos (Experimental - Modelo)")
            ax_res.set_title("Resíduos vs. Valores Previstos")
            ax_res.grid(True)
            st.pyplot(fig_res)
            st.write("""
            **Interpretação dos Resíduos:**
            - Idealmente, os resíduos devem estar dispersos aleatoriamente em torno da linha horizontal em 0, sem padrões óbvios.
            - **Padrões (ex: forma de funil, curva):** Podem indicar problemas como heterocedasticidade (variância não constante dos erros), não linearidade não capturada pelo modelo, ou variáveis omitidas.
            - **Outliers:** Pontos muito distantes da linha zero podem ser investigados.
            """)
        elif len(all_experimental_means) <= 1:
            st.info("ℹ️ Análise global do modelo requer dados de pelo menos dois espécimes com medições experimentais válidas.")

    else:
        st.warning("⚠️ É necessário primeiro calcular a evapotranspiração pelo modelo para todos os espécimes.")

# ---------------------------------------------------------------
# 10. Exibição do Histórico e Gráfico na Segunda Coluna
# ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col2:
    st.header("📋 Histórico de Resultados e Gráfico")
    if st.session_state.historico:
        # Criar DataFrame a partir do histórico
        data = {'Espécime': [], 'Evapotranspiração (litros/dia)': []}
        for rec in st.session_state.historico:
            data['Espécime'].append(rec[0])
            data['Evapotranspiração (litros/dia)'].append(rec[1])
        df_hist = pd.DataFrame(data)
        st.dataframe(df_hist)
        st.line_chart(df_hist.set_index('Espécime')['Evapotranspiração (litros/dia)'])
        
        # Visualizações adicionais: Histograma e Boxplot dos resultados do MODELO
        st.markdown("### 📊 Visualizações Adicionais (Evapotranspiração Estimada pelo Modelo)")
        
        # Histograma
        if not df_hist.empty:
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(df_hist['Evapotranspiração (litros/dia)'], bins=10, color='skyblue', edgecolor='black')
            ax_hist.set_title('Histograma de Evapotranspiração Estimada pelo Modelo')
            ax_hist.set_xlabel('Litros/dia')
            ax_hist.set_ylabel('Frequência')
            st.pyplot(fig_hist)

            # Boxplot
            fig_box, ax_box = plt.subplots()
            ax_box.boxplot(df_hist['Evapotranspiração (litros/dia)'], patch_artist=True)
            ax_box.set_title('Boxplot de Evapotranspiração Estimada pelo Modelo')
            ax_box.set_ylabel('Litros/dia')
            st.pyplot(fig_box)
    else:
        st.write("Nenhum cálculo realizado ainda.")

# ---------------------------------------------------------------
# 11. Seção Explicativa Expandida com Fórmulas e Interpretações (Enhanced for PhD Level)
# ---------------------------------------------------------------
with st.expander("🔍 Explicação Técnica e Interpretação Detalhada (Nível PhD)"):
    st.markdown("### 📚 Fundamentos do Modelo e Cálculos")
    st.markdown("""
    O modelo de evapotranspiração aqui apresentado é uma **abordagem empírica simplificada**. Ele combina variáveis físicas do espécime (Altura, Diâmetro, Área da Copa, LAI) com variáveis climáticas (Temperatura, Umidade, Radiação, Vento) utilizando pesos fixos. É crucial entender que, em um estudo de nível de doutorado, um modelo mais robusto seria idealmente:
    
    1.  **Baseado em princípios biofísicos:** Modelos como Penman-Monteith ou Priestley-Taylor, que derivam a ET de forma mais mecanística a partir do balanço de energia e resistência aerodinâmica/superficial.
    2.  **Calibrado e Validado com Dados Reais:** Os pesos (coeficientes) e a estrutura do modelo seriam determinados e ajustados usando extensos conjuntos de dados de medições de ET (por exemplo, usando câmaras de fluxo, lisímetros ou técnicas de covariância de vórtices) sob diversas condições ambientais e para diferentes espécies.
    3.  **Considerar Dinâmicas Temporais:** A ET varia significativamente ao longo do dia e das estações. Um modelo robusto incorporaria essas dinâmicas.
    """)
    st.markdown("**Área Foliar Total (AFT):** Uma métrica da área total das folhas do espécime. A fórmula usada (`largura * comprimento`) é uma aproximação simples para a área de folha individual e `* galhos` assume uma homogeneidade. Em estudos avançados, a AFT seria estimada usando métodos mais precisos como análise de imagem 3D, varredura a laser (LiDAR) ou relações alométricas espécie-específicas. A unidade é m² (se largura/comprimento em cm, converter).")
    st.latex(r'''
    \text{AFT} = \sum_{\text{folhas}} (\text{área da folha}) \times \text{número total de folhas (ou estimado por galho/árvore)}
    ''')
    st.markdown("**Índice de Área Foliar (LAI):** Uma variável adimensional crucial em ecologia e modelagem hidrológica. Representa a área foliar unilateral por unidade de área de solo projetada pela copa. Um LAI alto indica uma densa cobertura foliar, o que geralmente se correlaciona com taxas de ET mais altas (até certo ponto). É calculado como AFT / Área da Copa.")
    st.latex(r'''
    \text{LAI} = \frac{\text{Área Foliar Total (m}^2\text{)}}{\text{Área da Copa Projetada no Solo (m}^2\text{)}}
    ''')
    st.markdown("**Evapotranspiração (Modelo Empírico Atual):** A fórmula linear é um *proxy* ou uma simplificação extrema. Cada termo tenta capturar a influência relativa de diferentes variáveis na ET. Os pesos (0.3, 0.2, etc.) são arbitrários neste contexto de demonstração. Em um contexto de pesquisa rigoroso, eles seriam parâmetros do modelo a serem estimados (calibrados) a partir de dados experimentais usando regressão, otimização ou métodos de aprendizado de máquina. A unidade estimada é litros/dia.")
    st.latex(r'''
    \text{ET}_{\text{modelo}} = f(\text{Características Físicas, Variáveis Climáticas})
    ''')
    st.markdown("### 📊 Análise Estatística e Comparação Modelo-Experimento (Nível PhD)")
    st.markdown("""
    A comparação entre as estimativas do modelo e as medições experimentais é fundamental para **validar o modelo** e entender sua precisão e limitações. Várias abordagens estatísticas podem ser usadas:

    -   **Métricas de Erro e Desempenho:**
        -   **RMSE (Root Mean Squared Error):** Mede o desvio quadrático médio entre as previsões e as observações. É sensível a grandes erros. Unidade: litros/dia.
        -   **MAE (Mean Absolute Error):** Mede o desvio absoluto médio. Menos sensível a outliers que o RMSE. Unidade: litros/dia.
        -   **R-squared (R²):** (Mais relevante ao comparar um conjunto de previsões com um conjunto de observações para múltiplos espécimes ou ao longo do tempo) Indica a proporção da variância nas medições experimentais que é "explicada" pelo modelo. Valores mais próximos de 1 indicam melhor ajuste.
    
    -   **Testes de Hipótese:** Usados para avaliar formalmente se a diferença observada entre a previsão do modelo e a média experimental é estatisticamente significativa (improvável de ocorrer por acaso).
        -   **Teste t de Student (1 amostra):** Assume que as medições experimentais seguem uma distribuição normal e compara a média da amostra experimental com o valor previsto pelo modelo (tratado como um valor hipotético populacional).
        -   **Testes Não Paramétricos (Mann-Whitney, Wilcoxon, Sinal):** Úteis quando as suposições de normalidade do teste t não são atendidas ou com amostras pequenas. Eles comparam medianas ou distribuições, ou se baseiam apenas no sinal das diferenças. Embora o Mann-Whitney seja tipicamente para duas amostras, aqui foi adaptado para comparar com um valor constante, mas sua interpretação requer cautela.
    
    -   **Análise de Regressão (Experimental vs. Modelo):** Uma abordagem poderosa é regredir as médias experimentais observadas (Y) contra os valores previstos pelo modelo (X) através de múltiplos espécimes. Uma regressão ideal \( (\text{Experimental} = \beta_0 + \beta_1 \times \text{Modelo} + \epsilon) \) teria um intercepto \( \beta_0 \approx 0 \) (indicando ausência de viés sistemático) e uma inclinação \( \beta_1 \approx 1 \) (indicando que o modelo escala corretamente as previsões), com um \( R^2 \) alto e resíduos \( \epsilon \) distribuídos aleatoriamente sem padrões.
    
    ### 🎯 Aprofundamento e Robustez (Caminhos para PhD)
    
    Para uma avaliação probabilisticamente mais rica e robusta, considere:
    
    1.  **Quantificação da Incerteza:**
        -   **Intervalos de Confiança:** Para a média experimental.
        -   **Intervalos de Predição:** Para as futuras medições de ET, incorporando a incerteza do modelo e a variabilidade residual.
        -   **Métodos Bayesianos:** Permitem incorporar conhecimento prévio (priors), estimar distribuições de probabilidade para os parâmetros do modelo e obter intervalos de credibilidade para as previsões. Fornecem uma estrutura formal para atualizar o conhecimento à medida que novos dados se tornam disponíveis.
        -   **Propagação de Erro/Análise de Sensibilidade:** Analisar como a incerteza nas variáveis de entrada (medições físicas, dados climáticos) se propaga para a previsão da ET.

    2.  **Análise de Resíduos:** Examinar os resíduos (\( \text{Experimental} - \text{Modelo} \)) para identificar padrões (por exemplo, heterocedasticidade, viés em certas faixas de valores, dependência temporal/espacial) que indicam falhas nas suposições do modelo ou variáveis preditoras ausentes.
    
    3.  **Validação Cruzada:** Dividir o conjunto de dados em subconjuntos de treino e teste para avaliar o desempenho do modelo em dados não vistos.
    
    4.  **Testes Baseados em Reamostragem:** Bootstrap (para estimar a distribuição de métricas de desempenho ou incerteza) e Testes de Permutação (para testes de hipótese não paramétricos robustos).
    
    5.  **Modelagem Hierárquica ou de Efeitos Mistos:** Se houver dados agrupados (por exemplo, múltiplas medições no mesmo espécime, múltiplos espécimes no mesmo local), estes modelos podem lidar com a estrutura de dependência dos dados e permitir a estimação de efeitos específicos por grupo (espécime/local) e efeitos gerais.
    
    6.  **Integração com Modelos de Machine Learning (ML):**
        -   **Substituição do Modelo Empírico:** Treinar modelos de ML (ex: Random Forest, Gradient Boosting, Support Vector Regression, Redes Neurais) usando um conjunto de dados abrangente (características físicas, climáticas, LAI, e ET experimental) para desenvolver uma função `predict_evapotranspiration` mais precisa e potencialmente não linear. Bibliotecas como `scikit-learn` são fundamentais.
        -   **Extração de Características da Imagem:** Utilizar Redes Neurais Convolucionais (CNNs) para extrair características relevantes da imagem da planta (ex: densidade da copa, verdor, stress hídrico aparente) que podem servir como preditores adicionais para o modelo de ET.
        -   **Modelos Híbridos:** Combinar abordagens baseadas em física com ML.
    
    7.  **Considerações sobre q-Estatística (Estatística de Tsallis):**
        -   A q-estatística generaliza a estatística de Boltzmann-Gibbs e é aplicada em sistemas complexos com características não extensivas, como interações de longo alcance, memória, ou estruturas multifractais.
        -   No contexto ecológico da evapotranspiração, a aplicação da q-estatística (por exemplo, usando q-distribuições como a q-Gaussiana ou q-Exponencial para modelar a distribuição da ET ou dos erros do modelo) seria uma linha de pesquisa avançada. Exigiria uma justificativa teórica robusta para supor que os processos subjacentes à ET exibem tais propriedades não extensivas. Se essa hipótese for válida, a q-estatística poderia oferecer ferramentas para descrever distribuições de cauda pesada ou outras anomalias não capturadas pela estatística tradicional.
    
    Implementar essas abordagens transforma a análise de uma simples comparação ponto a ponto para uma avaliação probabilística rigorosa do modelo, essencial para uma tese de doutorado.
    
    ### ⚠️ Limitações do Modelo Atual
    
    É fundamental reconhecer as limitações do modelo linear simplificado e da abordagem de comparação atual para um trabalho de doutorado. A robustez e a complexidade de nível PhD viriam da aplicação das técnicas avançadas descritas acima e da construção/validação rigorosa de um modelo biofísico ou estatístico/ML mais sofisticado, calibrado com dados experimentais abrangentes e de alta qualidade.
    """)

# ---------------------------------------------------------------
# 12. Avaliação Prática Máxima (Keep this section as a summary/roadmap)
# ---------------------------------------------------------------
st.header("7) Avaliação Prática e Direções Futuras (Nível PhD)")
st.markdown("""
Para solidificar a abordagem a um nível de doutorado e garantir a validade ecológica e estatística dos resultados, os seguintes passos são cruciais, construindo sobre as análises realizadas:

### 📝 Roteiro para Aprofundamento:

1.  **Revisão Crítica do Modelo Empírico:** Avaliar a justificação teórica para os pesos e a estrutura do modelo linear. Idealmente, transitar para um modelo biofísico (Penman-Monteith, etc.) ou desenvolver um modelo estatístico/ML a partir de dados.
2.  **Expansão e Cura de Dados:** Coletar um conjunto de dados experimentais muito maior, abrangendo diversas espécies, locais, condições climáticas e estágios fenológicos. Garantir a qualidade dos dados (tratamento de outliers, dados faltantes).
3.  **Calibração e Validação Rigorosa:** Utilizar o conjunto de dados expandido para calibrar os parâmetros do modelo (se for um modelo paramétrico) e validar seu desempenho em um conjunto de dados independente (Validação Cruzada).
4.  **Quantificação Completa da Incerteza:** Estimar e apresentar a incerteza associada às previsões do modelo e às medições experimentais. Utilizar métodos como bootstrap, simulações de Monte Carlo ou abordagens Bayesianas para obter intervalos de predição ou credibilidade.
5.  **Análise Profunda de Resíduos:** Investigar os resíduos para entender onde e por que o modelo erra. Isso pode revelar a necessidade de incluir novas variáveis preditoras ou refinar a estrutura do modelo.
6.  **Análise de Sensibilidade Global:** Determinar quais variáveis de entrada (características da planta, clima) têm o maior impacto na previsão da ET e na sua incerteza.
7.  **Considerações Espaciais e Temporais:** Se os dados permitirem, incorporar a autocorrelação espacial e temporal nas análises estatísticas e na modelagem. Modelos de séries temporais ou modelos espaciais podem ser necessários.
8.  **Modelagem Hierárquica:** Se aplicável, usar modelos de efeitos mistos para lidar com a estrutura aninhada dos dados (por exemplo, medições dentro de galhos, galhos dentro de árvores, árvores dentro de locais).
9.  **Integração Multimodal:** Explorar a integração de dados de imagem (usando técnicas de Computer Vision para extrair características como área foliar, densidade da copa, saúde da planta) com dados tabulares (características físicas e climáticas) em modelos unificados (por exemplo, redes neurais multimodais).
10. **Interpretação Ecológica:** Relacionar os resultados estatísticos e as limitações do modelo com os princípios ecológicos subjacentes que regem a evapotranspiração. Discutir as implicações dos resultados para a gestão da água, ecologia florestal ou modelagem climática.

### 🛡️ Confiabilidade e Reproducibilidade:

-   **Documentação Detalhada:** Documentar rigorosamente todo o pipeline de dados, o modelo, as análises estatísticas e as suposições feitas.
-   **Código Aberto e Reprodutível:** Compartilhar o código e os dados (se possível e apropriado) para garantir a reprodutibilidade dos resultados.
-   **Revisão por Pares:** Submeter o trabalho à revisão crítica de especialistas em ecologia, hidrologia e estatística.

### 📈 Visualizações Avançadas:

-   **Gráficos de Resíduos:** Para diagnosticar problemas do modelo.
-   **Gráficos de Dispersão (Observed vs. Predicted):** Com linha 1:1 e linha de regressão, incluindo intervalos de incerteza.
-   **Mapas de Calor ou Gráficos de Superfície:** Para visualizar interações entre variáveis ou padrões espaciais/temporais.
-   **Boxplots ou Gráficos de Violino:** Para comparar distribuições de ET entre diferentes grupos de espécimes ou sob diferentes condições.

**Conclusão:** Alcançar um nível de doutorado na avaliação probabilística da evapotranspiração requer ir além da aplicação de testes estatísticos básicos. Envolve a construção e validação rigorosa de modelos (sejam eles biofísicos, estatísticos ou de machine learning), a quantificação exaustiva da incerteza, a análise crítica das suposições do modelo e a integração de conhecimentos de diferentes disciplinas (ecologia, estatística, sensoriamento remoto, ciência de dados). Esta aplicação Streamlit serve como um excelente ponto de partida para explorar esses conceitos e visualizar resultados preliminares.
""")
