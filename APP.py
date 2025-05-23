import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------------------------------------------------------------
# 1. Configurações da Página e Estado da Sessão
# ---------------------------------------------------------------
st.set_page_config(layout="wide", page_title="PhD EvapoCarbon Estimator", page_icon="🌳")

# Inicialização do estado da sessão
if "resultados_modelo" not in st.session_state:
    st.session_state.resultados_modelo = [] # Armazena dicts com todos os resultados por espécime
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "et_model_coeffs" not in st.session_state:
    st.session_state.et_model_coeffs = {
        "altura": 0.3, "diametro": 0.2, "copa": 0.1, "lai": 0.2,
        "temperatura": 0.1, "umidade": 0.05, "radiacao": 0.03, "vento": 0.02,
        "fator_escala": 10.0
    }
if "carbon_model_coeffs" not in st.session_state:
    st.session_state.carbon_model_coeffs = {
        "k_aft_carbon": 0.005, # kg C / m² AFT / dia
        "c_et_wue_carbon": 0.002 # kg C / L ET / dia (considerando uma WUE hipotética)
    }

# ---------------------------------------------------------------
# 2. Funções Científicas e de Cálculo
# ---------------------------------------------------------------
def calculate_area_foliar_total(folhas_data_list, galhos_principais):
    """Calcula a Área Foliar Total (AFT) em m²."""
    area_foliar_total_m2 = 0.0
    if not folhas_data_list:
        return 0.0

    # Assume que folhas_data_list contém tuplas (largura_cm_str, comprimento_cm_str)
    # Calcula a área média de uma folha de exemplo em m²
    soma_area_folhas_exemplo_m2 = 0
    num_folhas_validas = 0
    for largura_str, comprimento_str in folhas_data_list:
        try:
            largura_m = float(largura_str) / 100.0
            comprimento_m = float(comprimento_str) / 100.0
            soma_area_folhas_exemplo_m2 += (largura_m * comprimento_m) # Área de uma folha em m²
            num_folhas_validas +=1
        except ValueError:
            continue # Ignora entradas inválidas

    if num_folhas_validas == 0:
        return 0.0

    area_media_folha_m2 = soma_area_folhas_exemplo_m2 / num_folhas_validas

    # Estimativa MUITO simplificada: AFT = area_media_folha * num_folhas_por_galho_principal * num_galhos_principais
    # Para um estudo de doutorado, isso seria muito mais complexo (e.g., alometria, amostragem estratificada)
    # Vamos assumir um número hipotético de folhas por galho principal para ilustração
    num_folhas_estimado_por_galho = st.session_state.get(f"num_folhas_por_galho_estimado", 50) # Default, pode ser ajustado
    area_foliar_total_m2 = area_media_folha_m2 * num_folhas_estimado_por_galho * galhos_principais
    return area_foliar_total_m2

def calculate_lai(area_foliar_total_m2, area_copa_m2_str):
    """Calcula o Índice de Área Foliar (LAI)."""
    try:
        area_copa_m2 = float(area_copa_m2_str)
        if area_copa_m2 <= 0:
            return 0.0
        lai = area_foliar_total_m2 / area_copa_m2
        return round(lai, 3)
    except (ZeroDivisionError, ValueError):
        return 0.0

def predict_evapotranspiration_phd(
    altura_m, diametro_cm, area_copa_m2, lai,
    temperatura_c, umidade_perc, radiacao_mj_m2_dia, vento_m_s,
    coeffs, image_data=None # image_data não usado neste modelo, mas poderia ser em ML
    ):
    """
    Prevê a Evapotranspiração (ET) em litros/dia usando um modelo linear ponderado.
    Para um estudo de PhD, este modelo seria substituído por um biofísico (Penman-Monteith)
    ou um modelo de Machine Learning treinado e validado.
    Os coeficientes (coeffs) são demonstrativos e necessitariam de calibração rigorosa.
    """
    diametro_m = diametro_cm / 100.0
    umidade_frac = umidade_perc / 100.0

    et = (
        altura_m * coeffs["altura"] +
        diametro_m * coeffs["diametro"] +
        area_copa_m2 * coeffs["copa"] +
        lai * coeffs["lai"] +
        temperatura_c * coeffs["temperatura"] +
        umidade_frac * coeffs["umidade"] +
        radiacao_mj_m2_dia * coeffs["radiacao"] +
        vento_m_s * coeffs["vento"]
    ) * coeffs["fator_escala"]
    return round(et, 2)

def estimate_carbon_absorption_phd(area_foliar_total_m2, et_litros_dia, coeffs_carbon, especie_info=None):
    """
    Estima a absorção de carbono (kg C/dia) de forma simplificada.
    Para um estudo de PhD, isso envolveria modelos ecofisiológicos complexos,
    considerando fotossíntese, respiração, alocação de biomassa, WUE específica da espécie, etc.
    """
    # Estimativa baseada em AFT (mais direta, mas ainda muito simplificada)
    # k_aft_carbon: taxa média de fixação de C por m² de folha por dia. Varia enormemente.
    carbono_via_aft = area_foliar_total_m2 * coeffs_carbon["k_aft_carbon"]

    # Estimativa baseada em ET e WUE (muito indireta e conceitual)
    # c_et_wue_carbon: kg de C fixado por Litro de água transpirada.
    # Isso é uma proxy para a Eficiência no Uso da Água (WUE = Carbono ganho / Água perdida).
    # Varia drasticamente com espécie, CO2 atmosférico, VPD, etc.
    carbono_via_et_wue = et_litros_dia * coeffs_carbon["c_et_wue_carbon"]

    # Para esta ferramenta, podemos apresentar ambas ou uma média, com muitas ressalvas.
    # Vamos retornar uma média ponderada ou a mais defensável (AFT, com ressalvas).
    # Para uma banca, a discussão sobre como obter k_aft_carbon e c_et_wue_carbon seria crucial.
    # Ex: k_aft_carbon poderia vir de taxas fotossintéticas líquidas médias para espécies da Caatinga.
    # Ex: c_et_wue_carbon viria de estudos de WUE para essas espécies.

    # No contexto desta ferramenta, vamos usar a estimativa via AFT como principal,
    # pois é um pouco menos indireta que a via ET/WUE sem dados de WUE.
    return round(carbono_via_aft, 4), round(carbono_via_et_wue, 4)

# ---------------------------------------------------------------
# 3. Interface Streamlit
# ---------------------------------------------------------------
st.title("🌳 Plataforma Avançada de Estimativa de Evapotranspiração e Análise de Carbono para Ecossistemas Semiáridos")
st.subheader("Foco: Região de Crateús, Ceará, Brasil - Ferramenta de Suporte à Pesquisa de Doutorado")
st.markdown("---")

# --- Coluna da Esquerda: Entradas e Controles ---
left_column, right_column = st.columns([2, 3])

with left_column:
    st.header("⚙️ Entradas e Configurações do Modelo")

    with st.expander("🖼️ 1. Identificação Visual do Espécime (Opcional)", expanded=True):
        uploaded_file = st.file_uploader("Carregar imagem do espécime (JPG/PNG)", type=["jpg", "png"], key="img_uploader")
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Erro ao carregar imagem: {e}")
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_column_width=True)

    with st.expander("🌿 2. Dados Biométricos e Estruturais do Espécime", expanded=True):
        num_especimes = st.number_input("Número de Espécimes para Análise:", min_value=1, step=1, value=1, key="num_especimes_input")
        st.session_state.especimes_data_list = [] # Lista para armazenar dados de cada espécime

        for i in range(num_especimes):
            st.markdown(f"--- \n **Espécime {i+1}**")
            especime_id_user = st.text_input(f"Identificador do Espécime {i+1} (e.g., Tag001, EspecieX-LocalY):", f"Espécime_{i+1}", key=f"id_especime_{i}")
            altura_m_str = st.text_input(f"📏 Altura Total (m) - Espécime {i+1}:", "2.5", key=f"altura_m_{i}")
            diametro_cm_str = st.text_input(f"📐 Diâmetro à Altura do Peito (DAP) ou do Tronco (cm) - Espécime {i+1}:", "15", key=f"diametro_cm_{i}")
            area_copa_m2_str = st.text_input(f"🌳 Área da Copa Projetada no Solo (m²) - Espécime {i+1}:", "3.0", key=f"area_copa_m2_{i}")
            galhos_principais = st.number_input(f"🌿 Número Estimado de Galhos Estruturais Principais - Espécime {i+1}:", min_value=1, value=5, step=1, key=f"galhos_princ_{i}")
            st.session_state[f"num_folhas_por_galho_estimado_{i}"] = st.number_input(f"🍂 Número Médio Estimado de Folhas por Galho Principal - Espécime {i+1}:", min_value=1, value=50, step=5, key=f"num_folhas_galho_est_{i}", help="Este é um parâmetro crucial e difícil. Em uma pesquisa real, exigiria amostragem e estudo alométrico.")

            st.markdown(f"**Medidas de Folhas de Amostra (Espécime {i+1}):** (Para estimar área foliar média)")
            num_folhas_amostra = st.number_input(f"Quantas folhas de amostra para o Espécime {i+1}?", min_value=1, max_value=10, value=3, step=1, key=f"num_folhas_amostra_{i}")
            folhas_data_especime_list = []
            cols_folhas_amostra = st.columns(num_folhas_amostra)
            for j in range(num_folhas_amostra):
                with cols_folhas_amostra[j]:
                    st.markdown(f"🍃 Folha {j+1}")
                    largura_folha_cm_str = st.text_input(f"Largura (cm):", "6", key=f"larg_f_{i}_{j}")
                    comprimento_folha_cm_str = st.text_input(f"Comp. (cm):", "12", key=f"comp_f_{i}_{j}")
                    folhas_data_especime_list.append((largura_folha_cm_str, comprimento_folha_cm_str))

            st.session_state.especimes_data_list.append({
                "id_user": especime_id_user, "altura_m_str": altura_m_str, "diametro_cm_str": diametro_cm_str,
                "area_copa_m2_str": area_copa_m2_str, "galhos_principais": galhos_principais,
                "folhas_data_list": folhas_data_especime_list,
                "num_folhas_por_galho_estimado": st.session_state[f"num_folhas_por_galho_estimado_{i}"]
            })

    with st.expander("🌦️ 3. Variáveis Climáticas Médias do Período de Análise", expanded=True):
        st.markdown("Valores médios para o período de interesse (e.g., diário, semanal, mensal).")
        temp_c_str = st.text_input("🌡️ Temperatura Média do Ar (°C):", "28.5", key="temp_c")
        umid_perc_str = st.text_input("💧 Umidade Relativa Média do Ar (%):", "55", key="umid_perc")
        rad_mj_m2_dia_str = st.text_input("☀️ Radiação Solar Global Incidente Média Diária (MJ/m²/dia):", "19.5", key="rad_mj")
        vento_m_s_str = st.text_input("🌬️ Velocidade Média do Vento a 2m de Altura (m/s):", "2.2", key="vento_ms")

    with st.expander("🛠️ 4. Coeficientes do Modelo (Demonstrativo/Ajustável)", expanded=False):
        st.markdown("**Modelo de Evapotranspiração Empírico:**")
        st.caption("Estes coeficientes são para o modelo linear simplificado. Em uma pesquisa de doutorado, seriam calibrados ou o modelo seria substituído.")
        cols_coeffs_et1 = st.columns(2)
        st.session_state.et_model_coeffs["altura"] = cols_coeffs_et1[0].number_input("Peso Altura:", value=st.session_state.et_model_coeffs["altura"], step=0.01, format="%.2f", key="coeff_alt")
        st.session_state.et_model_coeffs["diametro"] = cols_coeffs_et1[1].number_input("Peso Diâmetro:", value=st.session_state.et_model_coeffs["diametro"], step=0.01, format="%.2f", key="coeff_diam")
        cols_coeffs_et2 = st.columns(2)
        st.session_state.et_model_coeffs["copa"] = cols_coeffs_et2[0].number_input("Peso Área Copa:", value=st.session_state.et_model_coeffs["copa"], step=0.01, format="%.2f", key="coeff_copa")
        st.session_state.et_model_coeffs["lai"] = cols_coeffs_et2[1].number_input("Peso LAI:", value=st.session_state.et_model_coeffs["lai"], step=0.01, format="%.2f", key="coeff_lai")
        cols_coeffs_et3 = st.columns(2)
        st.session_state.et_model_coeffs["temperatura"] = cols_coeffs_et3[0].number_input("Peso Temperatura:", value=st.session_state.et_model_coeffs["temperatura"], step=0.01, format="%.2f", key="coeff_temp")
        st.session_state.et_model_coeffs["umidade"] = cols_coeffs_et3[1].number_input("Peso Umidade:", value=st.session_state.et_model_coeffs["umidade"], step=0.01, format="%.3f", key="coeff_umid") # Mais precisão para umidade
        cols_coeffs_et4 = st.columns(2)
        st.session_state.et_model_coeffs["radiacao"] = cols_coeffs_et4[0].number_input("Peso Radiação:", value=st.session_state.et_model_coeffs["radiacao"], step=0.001, format="%.3f", key="coeff_rad")
        st.session_state.et_model_coeffs["vento"] = cols_coeffs_et4[1].number_input("Peso Vento:", value=st.session_state.et_model_coeffs["vento"], step=0.001, format="%.3f", key="coeff_vento")
        st.session_state.et_model_coeffs["fator_escala"] = st.number_input("Fator de Escala ET Geral:", value=st.session_state.et_model_coeffs["fator_escala"], step=0.1, format="%.1f", key="coeff_escala_et")

        st.markdown("**Modelo de Absorção de Carbono Simplificado:**")
        st.caption("Coeficientes altamente hipotéticos. Em pesquisa, seriam derivados de estudos ecofisiológicos detalhados para espécies da Caatinga.")
        st.session_state.carbon_model_coeffs["k_aft_carbon"] = st.number_input("Coef. Carbono via AFT (kg C/m²/dia):", value=st.session_state.carbon_model_coeffs["k_aft_carbon"], step=0.0001, format="%.4f", key="coeff_k_aft_c", help="Taxa média de fixação de Carbono por área foliar.")
        st.session_state.carbon_model_coeffs["c_et_wue_carbon"] = st.number_input("Coef. Carbono via ET/WUE (kg C/L ET):", value=st.session_state.carbon_model_coeffs["c_et_wue_carbon"], step=0.0001, format="%.4f", key="coeff_c_et_wue_c", help="Proxy para Eficiência no Uso da Água em termos de Carbono ganho por água perdida.")

    # --- Botão de Cálculo Principal ---
    st.markdown("---")
    if st.button("🚀 Executar Simulação e Análise", type="primary", key="run_simulation_button", use_container_width=True):
        st.session_state.resultados_modelo = [] # Limpa resultados anteriores

        # Validar e converter dados climáticos
        try:
            temp_val = float(temp_c_str)
            umid_val = float(umid_perc_str)
            rad_val = float(rad_mj_m2_dia_str)
            vento_val = float(vento_m_s_str)
            if not (0 < umid_val <= 100):
                st.error("Umidade Relativa deve estar entre 0 e 100%.")
                st.stop()
        except ValueError:
            st.error("Erro: Verifique se todas as variáveis climáticas são números válidos.")
            st.stop()

        # Processar cada espécime
        for i, especime_input_data in enumerate(st.session_state.especimes_data_list):
            try:
                altura_m = float(especime_input_data["altura_m_str"])
                diametro_cm = float(especime_input_data["diametro_cm_str"])
                area_copa_m2 = float(especime_input_data["area_copa_m2_str"])
                galhos_p = int(especime_input_data["galhos_principais"])
                num_folhas_galho_est = int(especime_input_data["num_folhas_por_galho_estimado"])


                if not (0.1 <= altura_m <= 200 and 0.1 <= diametro_cm <= 500 and 0.1 <= area_copa_m2 <= 1000):
                     st.warning(f"Espécime {especime_input_data['id_user']}: Valores biométricos parecem fora de um intervalo comum. Verifique as unidades e valores.")
                if galhos_p <=0 or num_folhas_galho_est <=0:
                    st.warning(f"Espécime {especime_input_data['id_user']}: Número de galhos e folhas por galho deve ser positivo.")
                    aft_m2_calc = 0
                    lai_calc = 0
                else:
                    aft_m2_calc = calculate_area_foliar_total(especime_input_data["folhas_data_list"], galhos_p) # Passa num_folhas_galho_est implicitamente via session_state na função
                    lai_calc = calculate_lai(aft_m2_calc, especime_input_data["area_copa_m2_str"])

                et_litros_dia_calc = predict_evapotranspiration_phd(
                    altura_m, diametro_cm, area_copa_m2, lai_calc,
                    temp_val, umid_val, rad_val, vento_val,
                    st.session_state.et_model_coeffs
                )

                carbono_aft_kg_dia, carbono_et_wue_kg_dia = estimate_carbon_absorption_phd(
                    aft_m2_calc, et_litros_dia_calc, st.session_state.carbon_model_coeffs
                )

                st.session_state.resultados_modelo.append({
                    "ID Espécime": especime_input_data["id_user"],
                    "Altura (m)": altura_m,
                    "Diâmetro (cm)": diametro_cm,
                    "Área Copa (m²)": area_copa_m2,
                    "AFT Estimada (m²)": round(aft_m2_calc, 3),
                    "LAI Estimado": lai_calc,
                    "ET Modelo (L/dia)": et_litros_dia_calc,
                    "Carbono (AFT) (kgC/dia)": carbono_aft_kg_dia,
                    "Carbono (ET/WUE) (kgC/dia)": carbono_et_wue_kg_dia
                })
            except ValueError:
                st.error(f"Erro ao processar dados do Espécime {especime_input_data['id_user']}. Verifique se todos os campos numéricos são válidos.")
                continue # Pula para o próximo espécime
            except Exception as e:
                st.error(f"Erro inesperado ao processar Espécime {especime_input_data['id_user']}: {e}")
                continue
        st.success(f"Simulação concluída para {len(st.session_state.resultados_modelo)} espécime(s). Veja os resultados à direita.")

# --- Coluna da Direita: Resultados e Análises ---
with right_column:
    st.header("📊 Resultados da Simulação e Análises")

    if not st.session_state.resultados_modelo:
        st.info("Aguardando execução da simulação. Configure as entradas à esquerda e clique em 'Executar Simulação'.")
    else:
        df_resultados = pd.DataFrame(st.session_state.resultados_modelo)
        st.subheader("Resumo dos Resultados do Modelo:")
        st.dataframe(df_resultados.style.format("{:.2f}", subset=pd.IndexSlice[:, ['Altura (m)', 'Diâmetro (cm)', 'Área Copa (m²)', 'AFT Estimada (m²)', 'LAI Estimado', 'ET Modelo (L/dia)']]).format("{:.4f}", subset=pd.IndexSlice[:, ['Carbono (AFT) (kgC/dia)', 'Carbono (ET/WUE) (kgC/dia)']]))

        st.subheader("Visualizações Gráficas dos Resultados do Modelo:")
        if not df_resultados.empty:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                if "ET Modelo (L/dia)" in df_resultados.columns:
                    fig_et, ax_et = plt.subplots()
                    ax_et.bar(df_resultados["ID Espécime"], df_resultados["ET Modelo (L/dia)"], color='skyblue')
                    ax_et.set_xlabel("ID do Espécime")
                    ax_et.set_ylabel("ET Estimada (L/dia)")
                    ax_et.set_title("Evapotranspiração Estimada por Espécime")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_et)

                    fig_hist_et, ax_hist_et = plt.subplots()
                    ax_hist_et.hist(df_resultados["ET Modelo (L/dia)"].dropna(), bins=10, color='lightblue', edgecolor='black')
                    ax_hist_et.set_xlabel("ET Estimada (L/dia)")
                    ax_hist_et.set_ylabel("Frequência")
                    ax_hist_et.set_title("Distribuição da ET Estimada")
                    st.pyplot(fig_hist_et)

            with col_g2:
                if "Carbono (AFT) (kgC/dia)" in df_resultados.columns:
                    fig_c, ax_c = plt.subplots()
                    ax_c.bar(df_resultados["ID Espécime"], df_resultados["Carbono (AFT) (kgC/dia)"], color='lightgreen')
                    ax_c.set_xlabel("ID do Espécime")
                    ax_c.set_ylabel("Carbono Estimado (kgC/dia - via AFT)")
                    ax_c.set_title("Absorção de Carbono Estimada (via AFT)")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_c)

                    fig_hist_c, ax_hist_c = plt.subplots()
                    ax_hist_c.hist(df_resultados["Carbono (AFT) (kgC/dia)"].dropna(), bins=10, color='green', edgecolor='black')
                    ax_hist_c.set_xlabel("Carbono Estimado (kgC/dia - via AFT)")
                    ax_hist_c.set_ylabel("Frequência")
                    ax_hist_c.set_title("Distribuição do Carbono Estimado (via AFT)")
                    st.pyplot(fig_hist_c)
            
            if "LAI Estimado" in df_resultados.columns and not df_resultados["LAI Estimado"].empty:
                st.markdown("#### Boxplot do LAI Estimado entre Espécimes")
                fig_box_lai, ax_box_lai = plt.subplots()
                ax_box_lai.boxplot(df_resultados["LAI Estimado"].dropna(), patch_artist=True, vert=False)
                ax_box_lai.set_yticklabels(['LAI'])
                ax_box_lai.set_xlabel('LAI Estimado')
                ax_box_lai.set_title('Distribuição do LAI Estimado')
                st.pyplot(fig_box_lai)


        st.markdown("---")
        st.subheader("🔬 Contraprova Experimental e Análise Estatística Comparativa")
        st.markdown("Insira abaixo os dados experimentais para comparação com as predições do modelo.")

        # Coleta de dados experimentais
        st.session_state.contraprovas_data = {}
        num_medicoes_exp = st.number_input("Número de Medições Experimentais por Espécime:", min_value=1, value=3, step=1, key="num_med_exp")
        tempo_coleta_exp_horas_str = st.text_input("Tempo de Coleta para Cada Medição Experimental (horas):", "24", key="tempo_coleta_exp_h")

        for _, row_modelo in df_resultados.iterrows():
            id_especime_modelo = row_modelo["ID Espécime"]
            with st.container(border=True):
                st.markdown(f"**Valores Experimentais para Espécime: {id_especime_modelo}**")
                medicoes_especime_list = []
                cols_med_exp = st.columns(num_medicoes_exp)
                for k in range(num_medicoes_exp):
                    with cols_med_exp[k]:
                        val_exp_ml_str = st.text_input(f"Medição {k+1} (mL):", "0", key=f"med_exp_{id_especime_modelo}_{k}")
                        medicoes_especime_list.append(val_exp_ml_str)
                st.session_state.contraprovas_data[id_especime_modelo] = medicoes_especime_list

        tipo_teste_estatistico = st.selectbox(
            "Escolha o Teste Estatístico para Comparação (Modelo vs. Experimental):",
            ("Teste t de Student (1 amostra)", "Teste de Wilcoxon (Signed-Rank)", "Diferença Absoluta e Percentual"),
            key="tipo_teste_stat_phd"
        )

        if st.button("🔄 Comparar Modelo com Dados Experimentais", key="run_comparison_button", use_container_width=True):
            if not st.session_state.contraprovas_data:
                st.warning("Por favor, insira os dados experimentais.")
            else:
                try:
                    tempo_coleta_h = float(tempo_coleta_exp_horas_str)
                    if tempo_coleta_h <= 0:
                        st.error("Tempo de coleta experimental deve ser positivo.")
                        st.stop()
                except ValueError:
                    st.error("Tempo de coleta experimental inválido.")
                    st.stop()

                st.markdown("### Resultados da Comparação Estatística:")
                all_exp_means_list = []
                all_model_preds_list = []

                for _, row_modelo in df_resultados.iterrows():
                    id_especime = row_modelo["ID Espécime"]
                    et_modelo_val = row_modelo["ET Modelo (L/dia)"]

                    if id_especime not in st.session_state.contraprovas_data:
                        st.warning(f"Dados experimentais não fornecidos para o espécime {id_especime}.")
                        continue

                    medicoes_exp_str_list = st.session_state.contraprovas_data[id_especime]
                    try:
                        medicoes_exp_ml_float = [float(m) for m in medicoes_exp_str_list]
                        medicoes_exp_L_dia = [(m_ml / 1000.0) / (tempo_coleta_h / 24.0) for m_ml in medicoes_exp_ml_float]
                        media_exp_L_dia = np.mean(medicoes_exp_L_dia)

                        all_exp_means_list.append(media_exp_L_dia)
                        all_model_preds_list.append(et_modelo_val)

                        st.markdown(f"#### Análise para Espécime: {id_especime}")
                        st.write(f"- ET Prevista pelo Modelo: {et_modelo_val:.2f} L/dia")
                        st.write(f"- ET Média Experimental: {media_exp_L_dia:.2f} L/dia (Baseado em {len(medicoes_exp_L_dia)} medições: {[f'{x:.2f}' for x in medicoes_exp_L_dia]})")

                        p_valor_teste_atual = None
                        if tipo_teste_estatistico == "Teste t de Student (1 amostra)":
                            if len(medicoes_exp_L_dia) < 2 or len(set(medicoes_exp_L_dia)) == 1:
                                st.warning("Teste t requer pelo menos 2 medições com variabilidade.")
                            else:
                                stat_t, p_valor_teste_atual = stats.ttest_1samp(medicoes_exp_L_dia, et_modelo_val)
                                st.write(f"  - Teste t: Estatística t = {stat_t:.3f}, p-valor = {p_valor_teste_atual:.4f}")
                        elif tipo_teste_estatistico == "Teste de Wilcoxon (Signed-Rank)":
                            if len(medicoes_exp_L_dia) < 1 : # scipy wilcoxon needs at least 1, but practically more
                                st.warning("Teste de Wilcoxon requer pelo menos algumas medições.")
                            else:
                                diffs = np.array(medicoes_exp_L_dia) - et_modelo_val
                                if np.all(diffs == 0):
                                     st.warning("Teste de Wilcoxon não aplicável: todas as diferenças são zero.")
                                else:
                                    try:
                                        stat_w, p_valor_teste_atual = stats.wilcoxon(diffs, alternative='two-sided') # Test if median of differences is zero
                                        st.write(f"  - Teste de Wilcoxon: Estatística W = {stat_w:.3f}, p-valor = {p_valor_teste_atual:.4f}")
                                    except ValueError as e_wilcoxon:
                                        st.warning(f"  - Teste de Wilcoxon não pôde ser calculado: {e_wilcoxon}")


                        diferenca_abs = abs(media_exp_L_dia - et_modelo_val)
                        diferenca_perc = (diferenca_abs / media_exp_L_dia) * 100 if media_exp_L_dia != 0 else float('inf')
                        st.write(f"  - Diferença Absoluta: {diferenca_abs:.2f} L/dia")
                        st.write(f"  - Diferença Percentual: {diferenca_perc:.2f}%")

                        if p_valor_teste_atual is not None:
                            alpha = 0.05
                            if p_valor_teste_atual < alpha:
                                st.error(f"  - Conclusão: Diferença estatisticamente significativa (p < {alpha}). O modelo difere da média experimental para este espécime.")
                            else:
                                st.success(f"  - Conclusão: Diferença não estatisticamente significativa (p ≥ {alpha}). Não há evidência forte de que o modelo difere da média experimental.")

                    except ValueError:
                        st.error(f"Erro ao converter dados experimentais para o espécime {id_especime}. Verifique os valores.")
                        continue
                    except Exception as e_stat:
                        st.error(f"Erro na análise estatística para {id_especime}: {e_stat}")
                        continue
                
                # Análise Global (se houver múltiplos espécimes com dados)
                if len(all_exp_means_list) > 1 and len(all_model_preds_list) > 1:
                    st.markdown("--- \n ### Análise Global de Desempenho do Modelo (Comparativo)")
                    exp_means_np_global = np.array(all_exp_means_list)
                    model_preds_np_global = np.array(all_model_preds_list)

                    rmse_global = np.sqrt(mean_squared_error(exp_means_np_global, model_preds_np_global))
                    mae_global = mean_absolute_error(exp_means_np_global, model_preds_np_global)
                    try:
                        r2_global = r2_score(exp_means_np_global, model_preds_np_global)
                    except ValueError: # Can happen if only one sample or no variance
                        r2_global = np.nan

                    st.write(f"**Métricas Globais de Comparação:**")
                    st.write(f"- RMSE Global: {rmse_global:.3f} L/dia")
                    st.write(f"- MAE Global: {mae_global:.3f} L/dia")
                    st.write(f"- R² Global: {r2_global:.3f}" if not np.isnan(r2_global) else "- R² Global: N/A (requer mais variabilidade/pontos)")

                    fig_scatter_global, ax_scatter_global = plt.subplots()
                    ax_scatter_global.scatter(model_preds_np_global, exp_means_np_global, alpha=0.7, edgecolors='k')
                    min_val_plot = min(min(model_preds_np_global), min(exp_means_np_global))
                    max_val_plot = max(max(model_preds_np_global), max(exp_means_np_global))
                    ax_scatter_global.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', label="Linha 1:1 (Ideal)")
                    ax_scatter_global.set_xlabel("ET Prevista pelo Modelo (L/dia)")
                    ax_scatter_global.set_ylabel("ET Média Experimental (L/dia)")
                    ax_scatter_global.set_title("Comparação Global: Modelo vs. Experimental")
                    ax_scatter_global.legend()
                    ax_scatter_global.grid(True)
                    st.pyplot(fig_scatter_global)
                elif len(all_exp_means_list) <= 1:
                    st.info("Análise global requer dados comparativos de pelo menos dois espécimes.")


# ---------------------------------------------------------------
# 11. Seção Explicativa Expandida (Nível PhD)
# ---------------------------------------------------------------
st.sidebar.title("Navegação e Informações")
st.sidebar.info(f"""
**Plataforma de Simulação e Análise para Pesquisa de Doutorado**
Foco: Evapotranspiração e Dinâmica de Carbono em Ecossistemas Semiáridos (Crateús, Ceará).
Versão: 1.0.0 (Robusta para Discussão em Banca)
Data: {pd.Timestamp.now().strftime('%Y-%m-%d')}
""")

with st.sidebar.expander("⚠️ Limitações e Próximos Passos (PhD)", expanded=False):
    st.markdown("""
    Esta ferramenta demonstra conceitos e um fluxo de análise. Para uma tese de doutorado:
    - **Modelo de ET:** Implementar modelos biofísicos robustos (e.g., Penman-Monteith com calibração de condutância estomática) ou modelos de Machine Learning (Random Forest, Redes Neurais) treinados com dados de campo extensivos de Crateús.
    - **Modelo de Carbono:** Desenvolver/aplicar modelos ecofisiológicos de fotossíntese e alocação de carbono (e.g., Farquhar, modelos baseados em LUE - Light Use Efficiency), calibrados para espécies da Caatinga.
    - **Dados de Campo:** Coleta extensiva de dados biométricos, microclimáticos, de fluxo de seiva (para ET), e trocas gasosas (para fotossíntese/respiração) em Crateús.
    - **Análise de Incerteza e Sensibilidade:** Aplicar métodos formais (e.g., Monte Carlo, GSA) para os modelos desenvolvidos.
    - **Validação Cruzada:** Rigorosa para modelos de ML.
    - **q-Estatística:** Investigar se as distribuições de variáveis ou erros do modelo exibem características não extensivas que justifiquem a aplicação da Estatística de Tsallis para uma descrição mais precisa.
    - **Escalonamento Espacial:** Utilizar sensoriamento remoto e SIG para extrapolar estimativas para a paisagem de Crateús.
    """)

with st.expander("🔍 Fundamentos Teóricos e Metodológicos (Discussão para Banca)", expanded=False):
    st.markdown("### 📚 Modelo de Evapotranspiração (ET)")
    st.markdown("""
    A ET é um componente crucial do ciclo hidrológico e do balanço energético, especialmente em regiões semiáridas como Crateús.
    - **Modelo Empírico Simplificado (Usado Aqui):** Uma função linear ponderada de variáveis biométricas e climáticas.
        - **Vantagens:** Simplicidade, fácil implementação, útil para análises exploratórias iniciais.
        - **Desvantagens para PhD:** Falta de base biofísica robusta, coeficientes arbitrários sem calibração, não captura interações complexas nem respostas não lineares.
    - **Abordagem de Doutorado (Recomendada):**
        1.  **Modelo de Penman-Monteith (FAO-56 PM):** Padrão ouro, combina balanço de energia com termos de transporte aerodinâmico e resistência superficial (condutância estomática, $g_s$). Requer calibração de $g_s$ para espécies locais da Caatinga, considerando fatores como déficit de pressão de vapor (VPD), radiação, umidade do solo.
            $ET_0 = \\frac{0.408 \\Delta (R_n - G) + \\gamma \\frac{900}{T+273} u_2 (e_s - e_a)}{\\Delta + \\gamma (1 + 0.34 u_2)}$ (para cultura de referência)
            Para ET real ($ET_c$), $ET_c = K_c ET_0$ ou modelagem direta de $g_s$.
        2.  **Modelos de Machine Learning:** Random Forest, Gradient Boosting, Redes Neurais, treinados com dados de ET medidos (e.g., fluxo de seiva, lisímetros, covariância de vórtices) e preditores ambientais/biométricos. Exigem grandes conjuntos de dados para treinamento e validação.
    """)

    st.markdown("### 🍂 Área Foliar Total (AFT) e Índice de Área Foliar (LAI)")
    st.markdown("""
    - **AFT:** Área total de superfície foliar fotossinteticamente ativa. Crucial para trocas gasosas.
    - **LAI:** AFT por unidade de área de solo ($LAI = AFT/A_{copa}$). Adimensional, indica a densidade do dossel.
    - **Estimativa (Usada Aqui):** Baseada na área média de folhas de amostra e estimativas do número de folhas. Altamente simplificado.
    - **Abordagem de Doutorado:**
        1.  **Métodos Diretos (Destrutivos):** Coleta de todas as folhas (inviável para árvores grandes).
        2.  **Métodos Indiretos:**
            -   **Ópticos:** Ceptômetros (e.g., LAI-2000/2200), câmeras hemisféricas, DHP (Digital Hemispherical Photography). Requerem calibração e correção para agrupamento de folhas.
            -   **Alometria:** Relações entre AFT/LAI e variáveis fáceis de medir (DAP, altura). Requer desenvolvimento de equações alométricas específicas para as espécies da Caatinga em Crateús.
            -   **Sensoriamento Remoto:** Índices de vegetação (NDVI, EVI) de imagens de satélite/drone, correlacionados com LAI medido em campo.
    """)

    st.markdown("### 🌳 Estimativa de Absorção/Captura de Carbono")
    st.markdown("""
    A fixação de carbono via fotossíntese é o principal mecanismo de entrada de C nos ecossistemas terrestres.
    - **Estimativa Simplificada (Usada Aqui):** Coeficientes fixos multiplicados por AFT ou ET. Meramente ilustrativo.
    - **Abordagem de Doutorado:**
        1.  **Modelos Ecofisiológicos de Fotossíntese:** Modelo de Farquhar, von Caemmerer & Berry (FvCB) para fotossíntese da folha, considerando limitações por Rubisco, regeneração de RuBP e exportação de triose-fosfato. Requer parametrização de $V_{cmax}$, $J_{max}$, etc., para espécies da Caatinga.
            $A = \min(A_c, A_j, A_p) - R_d$
        2.  **Modelos Baseados em Eficiência no Uso da Luz (LUE):** $NPP = APAR \times LUE_{eco}$, onde APAR é a radiação fotossinteticamente ativa absorvida e $LUE_{eco}$ é a eficiência do ecossistema em converter luz em biomassa. $LUE_{eco}$ é modulada por fatores ambientais.
        3.  **Balanço de Carbono do Ecossistema:** Medições de fluxos de CO₂ (covariância de vórtices) para estimar a Troca Líquida do Ecossistema (NEE), que integra a Produtividade Primária Bruta (GPP) e a Respiração do Ecossistema ($R_{eco}$). $NPP \approx GPP - R_a$ (respiração autotrófica).
        4.  **Alocação de Biomassa:** Entender como o carbono fixado é distribuído entre folhas, caules, raízes.
        5.  **Eficiência no Uso da Água (WUE):** $WUE = A / E$ (Fotossíntese / Transpiração). Crucial em ambientes semiáridos. Varia com CO₂, VPD, espécie.
    """)

    st.markdown("### 🔬 Análise Estatística e Validação de Modelos (PhD)")
    st.markdown("""
    - **Métricas de Desempenho:** RMSE, MAE, R², Bias, Índice de Willmott (d).
    - **Testes de Hipótese:** Para comparar médias ou distribuições (modelo vs. observado).
    - **Análise de Resíduos:** Verificar normalidade, homocedasticidade, ausência de autocorrelação. Padrões nos resíduos indicam falhas do modelo.
    - **Validação Cruzada (k-fold, Leave-One-Out):** Essencial para modelos de ML, para avaliar a capacidade de generalização.
    - **Análise de Incerteza:** Propagação de incertezas dos parâmetros de entrada e da estrutura do modelo para as previsões (e.g., Monte Carlo, GLUE).
    - **Análise de Sensibilidade:** Identificar quais parâmetros de entrada mais influenciam as saídas do modelo (e.g., métodos locais OAT, métodos globais como Sobol).
    """)

    st.markdown("### ⚛️ q-Estatística (Estatística de Tsallis) em Pesquisas Ecológicas")
    st.markdown("""
    A q-estatística generaliza a estatística de Boltzmann-Gibbs, sendo útil para sistemas complexos com:
    -   **Não-extensividade:** Onde a entropia de um sistema composto não é a soma das entropias das partes.
    -   **Correlações de Longo Alcance, Efeitos de Memória, Hierarquias Fractais.**
    -   **Distribuições de Cauda Pesada (Power-laws):** Frequentemente observadas em dados ecológicos (e.g., distribuição de tamanhos de organismos, frequência de eventos extremos).

    **Aplicação Potencial em ET e Carbono (Nível PhD):**
    1.  **Modelagem de Distribuições:** Se dados de ET, fluxos de carbono, ou erros de modelos exibirem caudas pesadas, distribuições q-generalizadas (q-Gaussiana, q-Exponencial) podem fornecer um ajuste melhor que as distribuições clássicas.
        -   A **q-Gaussiana** emerge da maximização da q-entropia de Tsallis sob certas restrições.
        -   A **q-Exponencial** pode descrever processos de relaxação ou distribuições de probabilidade em sistemas não extensivos.
    2.  **Análise de Séries Temporais:** Investigar se séries temporais de fluxos exibem memória de longo alcance ou multifractalidade, que podem ser caracterizadas usando ferramentas da q-estatística.
    3.  **Otimização e Inferência:** Métodos de otimização q-generalizados (e.g., Simulated Annealing q-generalizado) ou abordagens de inferência Bayesiana com q-distribuições.

    **Justificativa para Banca:** A aplicação da q-estatística seria justificada se houver evidência (teórica ou empírica dos dados de Crateús) de que os processos ecológicos subjacentes à ET e ao ciclo do carbono na Caatinga exibem características de sistemas complexos não adequadamente descritos pela estatística tradicional. Isso representaria uma fronteira de pesquisa, buscando uma compreensão mais fundamental da dinâmica do ecossistema.
    """)

st.markdown("---")
st.caption(f"Plataforma de Simulação Avançada - Versão para Discussão em Banca de Doutorado. {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
