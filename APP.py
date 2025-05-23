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
            # Convertendo cm² para m² (1 cm² = 0.0001 m²)
            total_area += ( (largura/100) * (comprimento/100) ) * galhos
        except ValueError:
            continue
    return total_area # Área em m²

def calculate_lai(area_foliar_total, area_copa):
    try:
        area_copa_val = float(area_copa) # Espera-se que a área da copa já esteja em m²
        if area_copa_val <= 0:
            return 0.0
        lai = area_foliar_total / area_copa_val # AFT já está em m²
        return round(lai, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0

def predict_evapotranspiration(image, altura, diametro, copa, lai, temperatura, umidade, radiacao, vento):
    # Fórmula ajustada para incluir variáveis climáticas
    # Os pesos são exemplificativos e precisariam de calibração rigorosa
    et = (altura * 0.3 +         # m
          (diametro/100) * 0.2 + # Convertendo cm para m para consistência, se o peso foi pensado para m
          copa * 0.1 +           # m²
          lai * 0.2 +            # adimensional
          temperatura * 0.1 +    # °C
          (umidade/100) * 0.05 + # Fração (0-1)
          radiacao * 0.03 +      # MJ/m²
          vento * 0.02           # m/s
         ) * 10 # Fator de ajuste/escala (arbitrário)
    return round(et, 2) # litros/dia (unidade final depende da calibração dos pesos)

def estimate_carbon_absorption_simplified(area_foliar_total_m2, et_litros_dia):
    # Coeficientes EXTREMAMENTE simplificados e apenas para ilustração
    # Estes valores NÃO têm base científica robusta sem pesquisa específica para as espécies e local.
    k_aft_carbon = 0.005  # kg C / m² de AFT / dia (valor hipotético)
    c_et_carbon = 0.001   # kg C / litro de ET / dia (valor hipotético)

    # Estimativa baseada em AFT
    carbono_via_aft = area_foliar_total_m2 * k_aft_carbon

    # Estimativa baseada em ET (muito indireta)
    carbono_via_et = et_litros_dia * c_et_carbon

    # Poderíamos retornar uma média ou a estimativa baseada em AFT que é um pouco mais direta
    # Aqui, retornamos a baseada em AFT para simplicidade do exemplo.
    # Em um estudo real, a abordagem seria muito mais complexa.
    return round(carbono_via_aft, 4)


# ---------------------------------------------------------------
# 3. Cabeçalho e título com ícone/emoji
# ---------------------------------------------------------------
st.title("🌱 Estimativa de Evapotranspiração e Análise Avançada (Rios Aéreos)")
st.markdown("""
Aplicação para estimar a evapotranspiração de espécimes arbóreos ou arbustivos, 
comparando resultados de um modelo empírico simplificado com medições experimentais, 
realizando análises estatísticas e explorando conceitos avançados para pesquisa.
Local: Crateús, Ceará, Brasil.
""")

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
num_especies = st.number_input("🔢 Quantidade de Espécimes:", min_value=1, step=1, value=1, key="num_especies")
especies_data = []

for i in range(num_especies):
    with st.container(border=True):
        st.subheader(f"🌿 Espécime {i+1}")
        col_especime1, col_especime2 = st.columns(2)
        with col_especime1:
            altura = st.text_input(f"📏 Altura (m) - Espécime {i+1}:", "1.5", key=f"altura_{i}")
            diametro = st.text_input(f"📐 Diâmetro do Tronco (cm) - Espécime {i+1}:", "10", key=f"diametro_{i}")
            copa = st.text_input(f"🌳 Área da Copa Projetada (m²) - Espécime {i+1}:", "2", key=f"copa_{i}")
        with col_especime2:
            galhos = st.number_input(f"🌿 Quantidade Estimada de Galhos Principais - Espécime {i+1}:", min_value=1, step=1, value=5, key=f"galhos_{i}")
            num_folhas_exemplo_galho = st.number_input(f"🍃 Número de Folhas de Exemplo por Galho Principal - Espécime {i+1}:", min_value=1, max_value=10, step=1, value=3, key=f"num_folhas_ex_galho_{i}")

        folhas_data_especime = []
        st.markdown(f"**Medidas das Folhas de Exemplo (Espécime {i+1}):**")
        cols_folhas = st.columns(num_folhas_exemplo_galho)
        for j in range(num_folhas_exemplo_galho):
            with cols_folhas[j]:
                st.markdown(f"🌱 Folha {j+1}")
                largura_folha = st.text_input(f"Largura (cm) F{j+1}:", "5", key=f"largura_folha_{i}_{j}")
                comprimento_folha = st.text_input(f"Comprimento (cm) F{j+1}:", "10", key=f"comprimento_folha_{i}_{j}")
                folhas_data_especime.append((largura_folha, comprimento_folha))
        especies_data.append({
            "altura": altura, "diametro": diametro, "copa": copa,
            "galhos": galhos, "folhas_data": folhas_data_especime,
            "id_especime": i+1
        })

# ---------------------------------------------------------------
# 6. Variáveis Climáticas
# ---------------------------------------------------------------
st.header("3) Insira as Variáveis Climáticas Médias do Período")
col_clima1, col_clima2 = st.columns(2)

with col_clima1:
    temperatura = st.text_input("🌡️ Temperatura Média (°C):", "28", key="temperatura")
    umidade = st.text_input("💧 Umidade Relativa Média (%):", "60", key="umidade")
with col_clima2:
    radiacao = st.text_input("☀️ Radiação Solar Média Diária (MJ/m²/dia):", "18", key="radiacao")
    vento = st.text_input("🌬️ Velocidade Média do Vento (m/s a 2m):", "2", key="vento")

# ---------------------------------------------------------------
# 7. Cálculo da Evapotranspiração (Modelo) e Carbono
# ---------------------------------------------------------------
st.header("4) Cálculo da Evapotranspiração (Modelo) e Estimativa de Carbono")
if st.button("💧 Calcular ET e Estimativa de Carbono", key="calc_et_carbon"):
    st.session_state.resultados = [] # Limpa resultados anteriores
    st.session_state.historico = [] # Limpa histórico anterior

    if st.session_state.uploaded_image is None and num_especies > 0 : # Permitir cálculo sem imagem, mas alertar
        st.warning("⚠️ Imagem não carregada. O cálculo prosseguirá, mas a imagem é recomendada para referência visual.")

    # Tentar converter as variáveis climáticas
    try:
        temperatura_val = float(temperatura)
        umidade_val = float(umidade)
        radiacao_val = float(radiacao)
        vento_val = float(vento)
    except ValueError:
        st.error("⚠️ Insira valores numéricos válidos para as variáveis climáticas.")
        st.stop() # Impede a continuação se climáticas forem inválidas

    for i, data_especime in enumerate(especies_data):
        st.markdown(f"---")
        st.subheader(f"Resultados para Espécime {data_especime['id_especime']}")
        try:
            altura_val = float(data_especime['altura'])
            diametro_val = float(data_especime['diametro'])
            copa_val = float(data_especime['copa'])
            galhos_val = int(data_especime['galhos'])

            if not (0.1 <= altura_val <= 150): # Ajustado intervalo
                st.warning(f"⚠️ Altura do Espécime {data_especime['id_especime']} ({altura_val}m) fora do intervalo plausível (0,1m - 150m). Verifique a unidade ou o valor.")
            else:
                st.success(f"✅ Altura do Espécime {data_especime['id_especime']} ({altura_val}m) plausível.")

            # AFT agora é em m²
            aft_m2 = calculate_area_foliar_total(data_especime['folhas_data'], galhos_val)
            st.write(f"🌿 **Área Foliar Total (AFT) estimada para o Espécime {data_especime['id_especime']}:** {aft_m2:.4f} m²")

            lai_val = calculate_lai(aft_m2, copa_val)
            st.write(f"🌿 **Índice de Área Foliar (LAI) estimado para o Espécime {data_especime['id_especime']}:** {lai_val:.2f}")

            et_val = predict_evapotranspiration(
                st.session_state.uploaded_image, # Passando a imagem (pode ser None)
                altura_val, diametro_val, copa_val, lai_val,
                temperatura_val, umidade_val, radiacao_val, vento_val
            )
            st.session_state.resultados.append(et_val)
            st.write(f"💧 **Evapotranspiração (ET) estimada para o Espécime {data_especime['id_especime']}:** {et_val:.2f} litros/dia")
            st.write("""
            **Explicação (ET):** Este valor mostra a evapotranspiração estimada para cada espécime, calculada com base no modelo empírico simplificado e nas variáveis fornecidas.
            **Interpretação (ET):** Indica a quantidade de água que se estima ser liberada pela planta para a atmosfera por dia.
            """)

            # Estimativa de Carbono Simplificada
            carbono_estimado_kg_dia = estimate_carbon_absorption_simplified(aft_m2, et_val)
            st.write(f"🌳 **Estimativa Simplificada de Absorção de Carbono para o Espécime {data_especime['id_especime']}:** {carbono_estimado_kg_dia:.4f} kg C/dia")
            st.caption("""
            **Nota:** Esta é uma estimativa **altamente simplificada** e apenas para fins ilustrativos, baseada em coeficientes hipotéticos. Uma análise de carbono rigorosa requer modelos e dados específicos.
            """)

            st.session_state.historico.append({
                "Espécime ID": data_especime['id_especime'],
                "ET (litros/dia)": et_val,
                "AFT (m²)": aft_m2,
                "LAI": lai_val,
                "Carbono Est. (kg C/dia)": carbono_estimado_kg_dia
            })

        except ValueError:
            st.error(f"⚠️ Espécime {data_especime['id_especime']}: Insira valores numéricos válidos para todas as variáveis físicas.")
            continue # Pula para o próximo espécime em caso de erro neste

# ---------------------------------------------------------------
# 8. Contraprova Experimental
# ---------------------------------------------------------------
st.header("5) Contraprova Experimental com Múltiplas Medições")
num_experimentos = st.number_input("🔢 Quantidade de medições experimentais para cada Espécime:", min_value=1, step=1, value=1, key="num_experimentos")
contraprovas = {}
for i in range(num_especies):
    with st.container(border=True):
        st.subheader(f"🌿 Espécime {especies_data[i]['id_especime']} - Valores Experimentais (mL)")
        valores_experimentais = []
        cols_exp = st.columns(num_experimentos)
        for j in range(num_experimentos):
            with cols_exp[j]:
                val = st.text_input(
                    f"Medição {j+1} (mL):",
                    "0",
                    key=f"medicao_{especies_data[i]['id_especime']}_{j}"
                )
                valores_experimentais.append(val)
        contraprovas[especies_data[i]['id_especime']] = valores_experimentais
tempo_coleta_horas = st.number_input("⏱️ Tempo (horas) de coleta para cada medição experimental:", min_value=0.1, step=0.1, value=24.0, key="tempo_coleta")

# ---------------------------------------------------------------
# 9. Escolha do Teste Estatístico e Comparação
# ---------------------------------------------------------------
st.header("6) Escolha o Teste Estatístico e Compare")
test_type = st.selectbox(
    "📊 Escolha o teste estatístico para comparação (Modelo vs. Experimental):",
    ("Teste t de Student (1 amostra)",
     "Teste de Wilcoxon (Signed-Rank Test)", # Mais apropriado que Mann-Whitney para 1 amostra vs valor
     "Teste de Sinal (Binomial Test)",
     "Diferença Absoluta e Percentual"),
    key="test_type_selector"
)

if st.button("🔄 Comparar com a Contraprova", key="compare_button"):
    if not st.session_state.resultados or len(st.session_state.resultados) != num_especies:
        st.warning("⚠️ É necessário primeiro calcular a evapotranspiração pelo modelo para todos os espécimes antes de comparar.")
    elif not contraprovas:
        st.warning("⚠️ Insira os dados da contraprova experimental.")
    else:
        all_experimental_means = []
        all_model_predictions = []
        all_residuals = []

        for idx_especime_modelo, data_modelo in enumerate(st.session_state.historico):
            especime_id = data_modelo["Espécime ID"]
            et_modelo = data_modelo["ET (litros/dia)"]

            st.markdown(f"---")
            st.subheader(f"🔎 Análise Detalhada - Espécime {especime_id}")

            if especime_id not in contraprovas or not contraprovas[especime_id]:
                st.warning(f"Dados experimentais não encontrados ou vazios para o Espécime {especime_id}.")
                continue
            try:
                valores_exp_str = contraprovas[especime_id]
                valores_exp_float_mL = [float(x) for x in valores_exp_str]

                evap_exps_litros_dia = []
                if tempo_coleta_horas <= 0:
                    st.error("Tempo de coleta deve ser maior que zero.")
                    continue

                for vol_mL in valores_exp_float_mL:
                    vol_L = vol_mL / 1000.0
                    vol_L_dia = vol_L / (tempo_coleta_horas / 24.0)
                    evap_exps_litros_dia.append(vol_L_dia)

                st.write(f"💧 **Medições Experimentais (convertidas para litros/dia) - Espécime {especime_id}:**", [f"{v:.2f}" for v in evap_exps_litros_dia])
                media_experimental = np.mean(evap_exps_litros_dia)
                st.write(f"📊 **Média Experimental (ET):** {media_experimental:.2f} litros/dia")
                st.write(f"🔮 **Valor Previsto pelo Modelo (ET):** {et_modelo:.2f} litros/dia")

                all_experimental_means.append(media_experimental)
                all_model_predictions.append(et_modelo)
                all_residuals.append(media_experimental - et_modelo)

                # Análise Estatística para este espécime
                p_value_current_test = None
                if len(evap_exps_litros_dia) < 2 and test_type != "Diferença Absoluta e Percentual":
                    st.warning(f"⚠️ Análise estatística inferencial limitada para Espécime {especime_id}: Número insuficiente de medições experimentais ({len(evap_exps_litros_dia)}) para o teste selecionado. Apenas a Diferença Absoluta será mostrada.")
                    diferenca_abs = abs(media_experimental - et_modelo)
                    percent_diff = (diferenca_abs / media_experimental) * 100 if media_experimental != 0 else float('inf')
                    st.write(f"📉 **Diferença Absoluta (Modelo vs. Experimento):** {diferenca_abs:.2f} litros/dia")
                    st.write(f"📊 **Diferença Percentual:** {percent_diff:.2f}%")
                else:
                    st.markdown("📈 **Análise Estatística Inferencial (para este espécime):**")
                    if test_type == "Teste t de Student (1 amostra)":
                        if len(set(evap_exps_litros_dia)) == 1 and len(evap_exps_litros_dia) > 1 : # all values are same
                             st.warning("Teste t não pode ser calculado pois todos os valores experimentais são idênticos, resultando em desvio padrão zero.")
                        else:
                            try:
                                stat_t, p_value_t = stats.ttest_1samp(evap_exps_litros_dia, et_modelo)
                                p_value_current_test = p_value_t
                                st.write(f"T-estatística: {stat_t:.4f}, P-valor: {p_value_t:.6f}")
                            except Exception as e:
                                st.warning(f"Não foi possível executar o Teste t de Student: {e}")

                    elif test_type == "Teste de Wilcoxon (Signed-Rank Test)":
                        differences_wilcoxon = np.array(evap_exps_litros_dia) - et_modelo
                        if np.all(differences_wilcoxon == 0):
                            st.warning("Teste de Wilcoxon não pode ser calculado pois todas as diferenças são zero.")
                        else:
                            try:
                                stat_w, p_value_w = stats.wilcoxon(differences_wilcoxon, alternative='two-sided')
                                p_value_current_test = p_value_w
                                st.write(f"Estatística W: {stat_w:.4f}, P-valor: {p_value_w:.6f}")
                            except Exception as e:
                                st.warning(f"Não foi possível executar o Teste de Wilcoxon: {e}")


                    elif test_type == "Teste de Sinal (Binomial Test)":
                        differences_sign = np.array(evap_exps_litros_dia) - et_modelo
                        nonzero_diff = differences_sign[differences_sign != 0]
                        if len(nonzero_diff) == 0:
                            st.warning("Teste de Sinal não pode ser calculado pois todas as diferenças são zero.")
                        else:
                            pos_signs = np.sum(nonzero_diff > 0)
                            n_signs = len(nonzero_diff)
                            try:
                                res_binom = stats.binomtest(pos_signs, n_signs, 0.5, alternative='two-sided')
                                p_value_current_test = res_binom.pvalue
                                st.write(f"Número de sinais positivos: {pos_signs} de {n_signs} não-nulos. P-valor (Binomial): {res_binom.pvalue:.6f}")
                            except Exception as e:
                                st.warning(f"Não foi possível executar o Teste de Sinal: {e}")

                    elif test_type == "Diferença Absoluta e Percentual":
                        diferenca_abs = abs(media_experimental - et_modelo)
                        percent_diff = (diferenca_abs / media_experimental) * 100 if media_experimental != 0 else float('inf')
                        st.write(f"📉 **Diferença Absoluta (Modelo vs. Experimento):** {diferenca_abs:.2f} litros/dia")
                        st.write(f"📊 **Diferença Percentual:** {percent_diff:.2f}%")

                    # Conclusão Estatística para o espécime
                    if p_value_current_test is not None:
                        alpha = 0.05
                        st.markdown("---")
                        st.subheader("Conclusão Estatística (para este espécime):")
                        if p_value_current_test < alpha:
                            st.error(f"❌ **Resultado Estatisticamente Significativo (p = {p_value_current_test:.4f} < {alpha}).**")
                            st.write("A diferença observada entre a média experimental e o valor previsto pelo modelo para este espécime é estatisticamente significativa.")
                        else:
                            st.info(f"✅ **Resultado Não Estatisticamente Significativo (p = {p_value_current_test:.4f} ≥ {alpha}).**")
                            st.write("Não há evidência estatística forte para concluir que a média experimental difere significativamente do valor previsto pelo modelo para este espécime.")
                    elif test_type == "Diferença Absoluta e Percentual":
                         st.info("Para 'Diferença Absoluta e Percentual', a interpretação é direta do valor da diferença, não havendo p-valor.")


            except ValueError:
                st.error(f"⚠️ Espécime {especime_id}: Insira valores experimentais válidos (números).")
                continue
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado ao processar o espécime {especime_id}: {e}")
                continue


        # --- Análise Global do Modelo (após o loop dos espécimes) ---
        if len(all_experimental_means) > 1 and len(all_model_predictions) > 1:
            st.markdown("---")
            st.header("🌍 Análise Global do Desempenho do Modelo")

            exp_means_np = np.array(all_experimental_means)
            model_preds_np = np.array(all_model_predictions)

            global_rmse = np.sqrt(mean_squared_error(exp_means_np, model_preds_np))
            global_mae = mean_absolute_error(exp_means_np, model_preds_np)
            global_r2 = r2_score(exp_means_np, model_preds_np)

            st.subheader("📊 Métricas Globais de Desempenho")
            st.write(f"**Root Mean Squared Error (RMSE) Global:** {global_rmse:.4f} litros/dia")
            st.write(f"**Mean Absolute Error (MAE) Global:** {global_mae:.4f} litros/dia")
            st.write(f"**R-squared (R²) Global:** {global_r2:.4f}")

            st.subheader("📈 Análise de Regressão: Experimental vs. Modelo")
            try:
                slope, intercept, r_value_reg, p_value_reg, std_err_reg = stats.linregress(model_preds_np, exp_means_np)
                fig_reg, ax_reg = plt.subplots()
                ax_reg.scatter(model_preds_np, exp_means_np, label='Dados (Espécimes)', color='blue', alpha=0.7)
                ax_reg.plot(model_preds_np, intercept + slope * model_preds_np, 'r', label=f'Linha de Regressão\ny={slope:.2f}x + {intercept:.2f}\nR²={r_value_reg**2:.2f}')
                min_val = min(model_preds_np.min(), exp_means_np.min())
                max_val = max(model_preds_np.max(), exp_means_np.max())
                ax_reg.plot([min_val, max_val], [min_val, max_val], 'k--', label='Linha 1:1 (Ideal)')
                ax_reg.set_xlabel("ET Prevista pelo Modelo (litros/dia)")
                ax_reg.set_ylabel("ET Média Experimental (litros/dia)")
                ax_reg.set_title("Regressão: ET Experimental vs. ET Modelo")
                ax_reg.legend()
                ax_reg.grid(True)
                st.pyplot(fig_reg)
                st.write(f"**Intercepto:** {intercept:.4f}, **Inclinação (Slope):** {slope:.4f}, **P-valor da Regressão:** {p_value_reg:.4f}")
            except Exception as e:
                st.warning(f"Não foi possível realizar a análise de regressão: {e}")


            st.subheader("📉 Análise de Resíduos")
            residuals_np = np.array(all_residuals)
            fig_res, ax_res = plt.subplots()
            ax_res.scatter(model_preds_np, residuals_np, color='green', alpha=0.7)
            ax_res.axhline(0, color='red', linestyle='--')
            ax_res.set_xlabel("ET Prevista pelo Modelo (litros/dia)")
            ax_res.set_ylabel("Resíduos (Experimental - Modelo) (litros/dia)")
            ax_res.set_title("Resíduos vs. Valores Previstos")
            ax_res.grid(True)
            st.pyplot(fig_res)
        elif len(all_experimental_means) <=1 :
             st.info("ℹ️ Análise global do modelo requer dados de pelo menos dois espécimes com medições experimentais válidas para calcular métricas como R² e regressão.")


# ---------------------------------------------------------------
# 10. Exibição do Histórico e Gráfico
# ---------------------------------------------------------------
st.header("📜 Histórico de Resultados e Gráficos (Modelo)")
if st.session_state.historico:
    df_hist = pd.DataFrame(st.session_state.historico)
    st.dataframe(df_hist)

    if not df_hist.empty:
        st.subheader("📊 Gráficos dos Resultados do Modelo")
        col_graf1, col_graf2 = st.columns(2)

        with col_graf1:
            if 'ET (litros/dia)' in df_hist.columns:
                st.markdown("#### ET Estimada pelo Modelo por Espécime")
                st.line_chart(df_hist.set_index('Espécime ID')['ET (litros/dia)'])

                st.markdown("#### Distribuição da ET Estimada (Histograma)")
                fig_hist_et, ax_hist_et = plt.subplots()
                ax_hist_et.hist(df_hist['ET (litros/dia)'], bins=10, color='skyblue', edgecolor='black')
                ax_hist_et.set_title('Histograma de ET Estimada')
                ax_hist_et.set_xlabel('ET (litros/dia)')
                ax_hist_et.set_ylabel('Frequência')
                st.pyplot(fig_hist_et)

        with col_graf2:
            if 'Carbono Est. (kg C/dia)' in df_hist.columns:
                st.markdown("#### Carbono Estimado por Espécime")
                st.line_chart(df_hist.set_index('Espécime ID')['Carbono Est. (kg C/dia)'])

                st.markdown("#### Distribuição do Carbono Estimado (Histograma)")
                fig_hist_c, ax_hist_c = plt.subplots()
                ax_hist_c.hist(df_hist['Carbono Est. (kg C/dia)'], bins=10, color='lightgreen', edgecolor='black')
                ax_hist_c.set_title('Histograma de Carbono Estimado')
                ax_hist_c.set_xlabel('Carbono (kg C/dia)')
                ax_hist_c.set_ylabel('Frequência')
                st.pyplot(fig_hist_c)

        if 'LAI' in df_hist.columns:
            st.markdown("#### Boxplot do LAI Estimado")
            fig_box_lai, ax_box_lai = plt.subplots()
            ax_box_lai.boxplot(df_hist['LAI'].dropna(), patch_artist=True) #dropna para evitar erros se houver NaN
            ax_box_lai.set_title('Boxplot de LAI Estimado')
            ax_box_lai.set_ylabel('LAI')
            st.pyplot(fig_box_lai)
else:
    st.write("Nenhum cálculo realizado ainda para exibir no histórico.")


# ---------------------------------------------------------------
# 11. Seção Explicativa Expandida com Fórmulas e Interpretações
# ---------------------------------------------------------------
with st.expander("🔍 Explicação Técnica e Interpretação Detalhada (Nível PhD)", expanded=False):
    st.markdown("### 📚 Fundamentos do Modelo e Cálculos")
    st.markdown("""
    O modelo de evapotranspiração (ET) aqui apresentado é uma **abordagem empírica simplificada**. Ele combina variáveis físicas do espécime (Altura, Diâmetro, Área da Copa, LAI) com variáveis climáticas (Temperatura, Umidade, Radiação, Vento) utilizando pesos fixos. É crucial entender que, em um estudo de nível de doutorado, um modelo mais robusto seria idealmente:
    1.  **Baseado em princípios biofísicos:** Modelos como Penman-Monteith ou Priestley-Taylor, que derivam a ET de forma mais mecanística a partir do balanço de energia e resistência aerodinâmica/superficial.
    2.  **Calibrado e Validado com Dados Reais:** Os pesos (coeficientes) e a estrutura do modelo seriam determinados e ajustados usando extensos conjuntos de dados de medições de ET (por exemplo, usando câmaras de fluxo, lisímetros ou técnicas de covariância de vórtices) sob diversas condições ambientais e para diferentes espécies.
    3.  **Considerar Dinâmicas Temporais:** A ET varia significativamente ao longo do dia e das estações. Um modelo robusto incorporaria essas dinâmicas.
    """)
    st.markdown("**Área Foliar Total (AFT):** Uma métrica da área total das folhas do espécime, convertida para m². A fórmula usada (`(largura/100) * (comprimento/100)`) é uma aproximação para a área de folha individual e `* galhos * (número médio de folhas por galho)` (implícito na forma como os dados de folhas são coletados por galho e depois agregados) assume uma homogeneidade. Em estudos avançados, a AFT seria estimada usando métodos mais precisos como análise de imagem 3D, varredura a laser (LiDAR) ou relações alométricas espécie-específicas.")
    st.latex(r'''
    \text{AFT (m}^2\text{)} = \sum_{\text{folhas}} (\text{área da folha em m}^2\text{)}
    ''')
    st.markdown("**Índice de Área Foliar (LAI):** Uma variável adimensional crucial em ecologia e modelagem hidrológica. Representa a área foliar unilateral por unidade de área de solo projetada pela copa. Um LAI alto indica uma densa cobertura foliar, o que geralmente se correlaciona com taxas de ET mais altas (até certo ponto). É calculado como AFT (m²) / Área da Copa Projetada (m²).")
    st.latex(r'''
    \text{LAI} = \frac{\text{Área Foliar Total (m}^2\text{)}}{\text{Área da Copa Projetada no Solo (m}^2\text{)}}
    ''')
    st.markdown("**Evapotranspiração (Modelo Empírico Atual):** A fórmula linear é um *proxy* ou uma simplificação extrema. Cada termo tenta capturar a influência relativa de diferentes variáveis na ET. Os pesos (0.3, 0.2, etc.) são arbitrários neste contexto de demonstração. Em um contexto de pesquisa rigoroso, eles seriam parâmetros do modelo a serem estimados (calibrados) a partir de dados experimentais usando regressão, otimização ou métodos de aprendizado de máquina. A unidade estimada é litros/dia.")
    st.latex(r'''
    \text{ET}_{\text{modelo}} = k \times \sum (\text{peso}_i \times \text{variável}_i)
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
        -   **Testes Não Paramétricos (Wilcoxon Signed-Rank, Teste de Sinal):** Úteis quando as suposições de normalidade do teste t não são atendidas ou com amostras pequenas. O Teste de Wilcoxon é adequado para comparar uma amostra com um valor de referência (mediana), considerando as magnitudes das diferenças. O Teste de Sinal apenas considera a direção (positiva/negativa) das diferenças.
    -   **Análise de Regressão (Experimental vs. Modelo):** Uma abordagem poderosa é regredir as médias experimentais observadas (Y) contra os valores previstos pelo modelo (X) através de múltiplos espécimes. Uma regressão ideal \\( \\text{Experimental} = \\beta_0 + \\beta_1 \\times \\text{Modelo} + \\epsilon \\) teria um intercepto \\( \\beta_0 \\approx 0 \\) (indicando ausência de viés sistemático) e uma inclinação \\( \\beta_1 \\approx 1 \\) (indicando que o modelo escala corretamente as previsões), com um \\( R^2 \\) alto e resíduos \\( \\epsilon \\) distribuídos aleatoriamente sem padrões.
    """)
    st.markdown("### ⚛️ Considerações sobre q-Estatística e q-Exponencial")
    st.markdown("""
    A **q-estatística**, ou Estatística de Tsallis, emerge como uma generalização da mecânica estatística padrão de Boltzmann-Gibbs, sendo particularmente útil em sistemas complexos que exibem não-extensividade, como correlações de longo alcance ou multifractalidade.
    No contexto da evapotranspiração e da análise do nosso modelo:
    1.  **Distribuição dos Dados:** Poderíamos investigar se a distribuição das medições experimentais de evapotranspiração ou dos resíduos do nosso modelo (Experimental - Modelo) aderem melhor a q-distribuições (como a q-Gaussiana ou q-Exponencial) do que às distribuições Gaussianas tradicionais. Isso exigiria o ajuste dessas q-distribuições aos dados e a comparação do ajuste usando critérios apropriados (e.g., q-AIC).
    2.  **Modelagem de Erros:** Se os erros do modelo apresentarem caudas pesadas ou outras características não Gaussianas, a q-estatística poderia fornecer uma estrutura mais adequada para descrever sua distribuição, influenciando a forma como quantificamos a incerteza e realizamos testes de hipótese.
    3.  **Processos Não Lineares e Complexos:** A evapotranspiração é um processo complexo influenciado por inúmeros fatores que podem interagir de forma não linear. A q-estatística tem sido aplicada em outros sistemas ecológicos e ambientais complexos, e sua aplicação aqui exigiria uma fundamentação teórica para identificar potenciais aspectos de não-extensividade no sistema estudado.

    **q-Exponencial:** Uma das distribuições centrais na q-estatística. Sua forma funcional é:
    """)
    st.latex(r'''
    f(x; q, \beta) = N \exp_q(-\beta x) = N [1 - (1-q) \beta x]_+^{1/(1-q)}
    ''')
    st.markdown("""
    onde \\( q \\) é o índice entrópico de Tsallis, \\( \\beta \\) é um parâmetro relacionado à escala, \\( N \\) é a constante de normalização e \\( [z]_+ = \\max(0, z) \\). A q-exponencial generaliza a exponencial padrão (obtida no limite \\( q \\to 1 \\)) e pode descrever decadimentos mais lentos ou mais rápidos.
    A aplicação da q-estatística exigiria:
    -   **Justificativa Teórica:** Por que esperar um comportamento não extensivo no sistema de evapotranspiração estudado?
    -   **Ferramentas Computacionais:** Bibliotecas para ajustar q-distribuições aos dados e realizar inferência estatística baseada em q-estatística.
    -   **Interpretação Física/Ecológica:** O que os valores dos parâmetros q (diferentes de 1) nos diriam sobre a natureza do sistema?
    A integração da q-estatística seria uma direção de pesquisa avançada para uma tese de doutorado, explorando as possíveis propriedades não extensivas da evapotranspiração e suas implicações para a modelagem e análise.
    """)

    st.markdown("### 🤖 Integração de Métodos de Machine Learning")
    st.markdown("""
    O **Machine Learning (ML)** oferece um conjunto poderoso de ferramentas para construir modelos preditivos complexos a partir de dados, sem necessariamente depender de relações lineares predefinidas ou de um conhecimento biofísico completo dos processos subjacentes. No contexto da evapotranspiração:
    1.  **Modelos Preditivos:** Algoritmos de ML como Regressão Linear Múltipla (com seleção de características), Árvores de Decisão, Random Forests, Gradient Boosting Machines (e.g., XGBoost, LightGBM), Support Vector Machines (SVMs) e Redes Neurais podem ser treinados para prever a evapotranspiração usando as características físicas da planta (altura, diâmetro, copa, estimativas de LAI) e as variáveis climáticas como preditores.
    2.  **Aprendizado Não Linear:** Modelos de ML podem capturar relações não lineares e interações complexas entre as variáveis preditoras que um modelo linear simples (como o atual) não consegue.
    3.  **Seleção de Características:** Algoritmos de ML podem ajudar a identificar quais variáveis têm maior poder preditivo para a evapotranspiração, potencialmente simplificando o modelo ou revelando novas relações importantes.
    4.  **Previsão Baseada em Imagem:** Técnicas avançadas como Redes Neurais Convolucionais (CNNs) poderiam ser usadas para extrair características diretamente das imagens das plantas (e.g., textura da folha, densidade da copa, indicadores de saúde) e usar essas características como entrada para um modelo de previsão de ET. Isso poderia contornar a necessidade de medições diretas de LAI ou outras características físicas.

    **Implementação (Nível Conceitual):**
    -   **Coleta de Dados:** Um conjunto de dados robusto contendo medições de evapotranspiração (obtidas experimentalmente ou de bancos de dados existentes), juntamente com as características da planta e variáveis climáticas, seria essencial para treinar um modelo de ML.
    -   **Engenharia de Características:** Poderíamos criar novas características a partir das existentes (e.g., combinações de temperatura e umidade, índices baseados em LAI estimado e radiação).
    -   **Seleção de Modelo:** A escolha do algoritmo de ML dependeria do tamanho e da natureza dos dados, bem como do desempenho desejado (acurácia, interpretabilidade).
    -   **Treinamento e Validação:** O conjunto de dados seria dividido em treino e teste para treinar o modelo e avaliar seu desempenho em dados não vistos. Técnicas de validação cruzada seriam importantes para obter uma estimativa robusta do desempenho.
    -   **Interpretação do Modelo:** Dependendo do tipo de modelo de ML, a interpretabilidade pode variar. Técnicas para entender a importância das características podem ser aplicadas.
    Bibliotecas Python como `scikit-learn`, `tensorflow` e `pytorch` seriam ferramentas fundamentais para implementar abordagens de machine learning para a previsão da evapotranspiração em um projeto de pesquisa de doutorado. A integração de ML poderia levar a modelos preditivos mais precisos e a uma compreensão mais rica das complexas relações que influenciam a evapotranspiração.
    """)
    st.markdown("### 🎯 Aprofundamento e Robustez (Caminhos para PhD)")
    st.markdown("""
    Para uma avaliação probabilisticamente mais rica e robusta, considere:
    1.  **Quantificação da Incerteza:**
        -   **Intervalos de Confiança:** Para a média experimental.
        -   **Intervalos de Predição:** Para as futuras medições de ET, incorporando a incerteza do modelo e a variabilidade residual.
        -   **Métodos Bayesianos:** Permitem incorporar conhecimento prévio (priors), estimar distribuições de probabilidade para os parâmetros do modelo e obter intervalos de credibilidade para as previsões. Fornecem uma estrutura formal para atualizar o conhecimento à medida que novos dados se tornam disponíveis.
        -   **Propagação de Erro/Análise de Sensibilidade:** Analisar como a incerteza nas variáveis de entrada (medições físicas, dados climáticos) se propaga para a previsão da ET.
    2.  **Análise de Resíduos:** Examinar os resíduos (Experimental - Modelo) para identificar padrões (por exemplo, heterocedasticidade, viés em certas faixas de valores, dependência temporal/espacial) que indicam falhas nas suposições do modelo ou variáveis preditoras ausentes.
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
    """)
    st.markdown("### ⚠️ Limitações do Modelo Atual")
    st.markdown("""
    É fundamental reconhecer as limitações do modelo linear simplificado e da abordagem de comparação atual para um trabalho de doutorado. A robustez e a complexidade de nível PhD viriam da aplicação das técnicas avançadas descritas acima e da construção/validação rigorosa de um modelo biofísico ou estatístico/ML mais sofisticado, calibrado com dados experimentais abrangentes e de alta qualidade.
    """)

# ---------------------------------------------------------------
# 12. Estimativa Simplificada de Absorção/Captura de Carbono
# ---------------------------------------------------------------
with st.expander("🌳 Estimativa Simplificada de Absorção/Captura de Carbono (Conceitual)", expanded=False):
    st.markdown("""
    Estimar a absorção ou captura de carbono por plantas é um processo complexo que depende de muitos fatores, incluindo a espécie, estágio de crescimento, saúde da planta, condições ambientais (luz, temperatura, disponibilidade de água e nutrientes) e a taxa de fotossíntese líquida.
    A evapotranspiração está indiretamente relacionada à absorção de carbono através da abertura dos estômatos nas folhas. Os estômatos abrem para permitir a entrada de dióxido de carbono (CO₂) para a fotossíntese, mas também levam à perda de água por transpiração. Portanto, existe um trade-off entre a absorção de CO₂ e a perda de água.

    **Modelo Simplificado (Apenas para Ilustração na Aplicação Atual):**
    A estimativa de carbono fornecida nesta aplicação é **extremamente simplificada** e baseada em coeficientes hipotéticos multiplicados pela Área Foliar Total (AFT) ou pela Evapotranspiração (ET).
    """)
    st.latex(r'''
    \text{Absorção de Carbono (kg C/dia)} \approx k_{\text{AFT}} \times \text{AFT (m}^2\text{)}
    ''')
    st.markdown(r"""
    ou
    """)
    st.latex(r'''
    \text{Absorção de Carbono (kg C/dia)} \approx k_{\text{ET}} \times \text{ET (litros/dia)}
    ''')
    st.markdown(r"""
    onde \( k_{\text{AFT}} \) e \( k_{\text{ET}} \) são coeficientes de conversão **hipotéticos e não validados cientificamente neste contexto**.
    A função `estimate_carbon_absorption_simplified` no código usa um valor hipotético para \( k_{\text{AFT}} \).

    **Limitações Cruciais:**
    -   Essas são simplificações extremas e não levam em conta a complexidade da fisiologia da fotossíntese, os fatores ambientais dinâmicos, a respiração da planta, a alocação de carbono para diferentes partes da planta, etc.
    -   Uma estimativa precisa da absorção de carbono requer modelos biofísicos detalhados do ciclo do carbono, medições diretas das trocas de gases (CO₂ e H₂O) usando sistemas como câmaras de fluxo ou covariância de vórtices, ou o uso de modelos de sensoriamento remoto calibrados.
    Para uma pesquisa de nível de doutorado, a estimativa da absorção de carbono exigiria uma abordagem muito mais sofisticada, baseada em princípios fisiológicos e possivelmente na integração de dados de evapotranspiração com outras informações (e.g., radiação fotossinteticamente ativa - PAR, concentração de CO₂, dados de biomassa). O uso de modelos de balanço de carbono específicos para o tipo de vegetação em Crateús, Ceará, seria essencial.
    """)

# ---------------------------------------------------------------
# 13. Avaliação Prática Máxima
# ---------------------------------------------------------------
st.header("7) Avaliação Prática e Direções Futuras (Nível PhD)")
with st.expander("Ver Detalhes", expanded=False):
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
    10. **Interpretação Ecológica:** Relacionar os resultados estatísticos e as limitações do modelo com os princípios ecológicos subjacentes que regem a evapotranspiração. Discutir as implicações dos resultados para a gestão da água, ecologia florestal ou modelagem climática em Crateús, Ceará.

    ### ⚛️ Integração da q-Estatística na Avaliação:
    -   **Análise da Distribuição de Erros:** Ajustar q-distribuições (e.g., q-Gaussiana, q-Exponencial) aos resíduos do modelo de ET para verificar se elas fornecem um ajuste melhor do que as distribuições Gaussianas.
    -   **Testes de Hipótese q-Generalizados:** Explorar o uso de testes de hipótese baseados na q-estatística, se apropriado para a natureza dos dados e as perguntas de pesquisa.
    -   **Quantificação de Incerteza q-Generalizada:** Se a distribuição dos erros for melhor descrita por q-distribuições, usar essas distribuições para quantificar a incerteza nas previsões do modelo (e.g., construir q-intervalos de confiança/predição).

    ### 🤖 Aplicação de Machine Learning para Modelagem de ET e Absorção de Carbono:
    -   **Desenvolvimento de Modelos de ET Baseados em ML:** Treinar modelos de ML (Random Forest, Redes Neurais, etc.) usando um conjunto de dados expandido (incluindo variáveis climáticas, características da planta e, idealmente, medições diretas de ET) para obter previsões mais precisas e capturar não linearidades.
    -   **Previsão da Absorção de Carbono com ML:** Se houver dados de absorção de carbono disponíveis (de literatura, bancos de dados ou medições diretas), treinar modelos de ML para prever a absorção de carbono usando variáveis como características da planta, clima e potencialmente ET como preditores.
    -   **Extração de Características de Imagem para Modelos de ML:** Usar técnicas de visão computacional (CNNs) para extrair informações relevantes das imagens das plantas (e.g., área foliar, índice de vegetação) e usar essas informações como entrada para modelos de ML de ET ou absorção de carbono.

    ### 🌿 Modelagem da Absorção de Carbono (Nível Avançado):
    -   **Uso de Modelos Biofísicos do Ciclo do Carbono:** Empregar modelos ecofisiológicos que simulam os processos de fotossíntese, respiração e alocação de carbono para estimar a absorção de carbono de forma mais mecanística.
    -   **Integração com Dados de Sensoriamento Remoto:** Usar dados de satélite (e.g., índices de vegetação, radiação absorvida fotossinteticamente ativa - APAR) para escalar as estimativas de absorção de carbono para áreas maiores.
    -   **Medições Experimentais de Trocas de Gases:** Realizar medições de campo das taxas de fotossíntese e respiração usando câmaras portáteis para calibrar e validar os modelos de absorção de carbono.
    -   **Análise da Eficiência do Uso da Água (WUE):** Investigar a relação entre a absorção de carbono e a perda de água (evapotranspiração) para diferentes espécies e condições ambientais em Crateús. A WUE pode fornecer insights sobre as estratégias das plantas para maximizar a fixação de carbono com o mínimo de perda de água.


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

st.markdown("---")
st.caption(f"Aplicação desenvolvida para fins de pesquisa e demonstração. Localização de referência: Crateús, Ceará, Brasil. Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
