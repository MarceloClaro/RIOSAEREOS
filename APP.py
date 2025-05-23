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
    st.session_state.resultados_modelo = []
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
        "k_aft_carbon": 0.005,
        "c_et_wue_carbon": 0.002
    }
if "especimes_data_list" not in st.session_state:
    st.session_state.especimes_data_list = []
if "contraprovas_data" not in st.session_state:
    st.session_state.contraprovas_data = {}

# CORREÇÃO: Inicializar com um valor válido >= min_value do number_input
if 'num_medicoes_exp_anterior' not in st.session_state:
    st.session_state.num_medicoes_exp_anterior = 3 # Default value, >= min_value (1)

if 'num_especimes_anterior_bio' not in st.session_state:
    st.session_state.num_especimes_anterior_bio = 1
if 'num_especimes_anterior_comp' not in st.session_state:
    st.session_state.num_especimes_anterior_comp = 0


# ---------------------------------------------------------------
# 2. Funções Científicas e de Cálculo
# ---------------------------------------------------------------
def calculate_area_foliar_total(folhas_data_list, galhos_principais, num_folhas_estimado_por_galho):
    area_foliar_total_m2 = 0.0
    if not folhas_data_list or galhos_principais <= 0 or num_folhas_estimado_por_galho <=0:
        return 0.0
    soma_area_folhas_exemplo_m2 = 0
    num_folhas_validas = 0
    for idx_folha, (largura_str, comprimento_str) in enumerate(folhas_data_list):
        try:
            largura_m = float(largura_str) / 100.0
            comprimento_m = float(comprimento_str) / 100.0
            soma_area_folhas_exemplo_m2 += (largura_m * comprimento_m)
            num_folhas_validas +=1
        except ValueError:
            st.warning(f"Valor inválido ('{largura_str}' ou '{comprimento_str}') para dimensão da folha de amostra {idx_folha+1}. Será ignorado.")
            continue
    if num_folhas_validas == 0: return 0.0
    area_media_folha_m2 = soma_area_folhas_exemplo_m2 / num_folhas_validas
    area_foliar_total_m2 = area_media_folha_m2 * num_folhas_estimado_por_galho * galhos_principais
    return area_foliar_total_m2

def calculate_lai(area_foliar_total_m2, area_copa_m2_str):
    try:
        area_copa_m2 = float(area_copa_m2_str)
        if area_copa_m2 <= 0: return 0.0
        lai = area_foliar_total_m2 / area_copa_m2
        return round(lai, 3)
    except (ZeroDivisionError, ValueError): return 0.0

def predict_evapotranspiration_phd(altura_m, diametro_cm, area_copa_m2, lai, temperatura_c, umidade_perc, radiacao_mj_m2_dia, vento_m_s, coeffs, image_data=None):
    diametro_m = diametro_cm / 100.0
    umidade_frac = umidade_perc / 100.0
    et = (altura_m*coeffs["altura"] + diametro_m*coeffs["diametro"] + area_copa_m2*coeffs["copa"] + lai*coeffs["lai"] + temperatura_c*coeffs["temperatura"] + umidade_frac*coeffs["umidade"] + radiacao_mj_m2_dia*coeffs["radiacao"] + vento_m_s*coeffs["vento"]) * coeffs["fator_escala"]
    return round(et, 2)

def estimate_carbon_absorption_phd(area_foliar_total_m2, et_litros_dia, coeffs_carbon, especie_info=None):
    carbono_via_aft = area_foliar_total_m2 * coeffs_carbon["k_aft_carbon"]
    carbono_via_et_wue = et_litros_dia * coeffs_carbon["c_et_wue_carbon"]
    return round(carbono_via_aft, 4), round(carbono_via_et_wue, 4)

# ---------------------------------------------------------------
# 3. Interface Streamlit
# ---------------------------------------------------------------
st.title("🌳 Plataforma Avançada de Estimativa de Evapotranspiração e Análise de Carbono para Ecossistemas Semiáridos")
st.subheader("Foco: Região de Crateús, Ceará, Brasil - Ferramenta de Suporte à Pesquisa de Doutorado")
st.markdown("---")

left_column, right_column = st.columns([2, 3])

with left_column:
    st.header("⚙️ Entradas e Configurações do Modelo")
    with st.expander("🖼️ 1. Identificação Visual do Espécime (Opcional)", expanded=True):
        uploaded_file = st.file_uploader("Carregar imagem do espécime (JPG/PNG)", type=["jpg", "png"], key="img_uploader_main")
        if uploaded_file is not None:
            try: st.session_state.uploaded_image = Image.open(uploaded_file)
            except Exception as e: st.error(f"Erro ao carregar imagem: {e}")
        if st.session_state.uploaded_image: st.image(st.session_state.uploaded_image, caption="Imagem Carregada", use_column_width=True)

    with st.expander("🌿 2. Dados Biométricos e Estruturais do Espécime", expanded=True):
        num_especimes_bio = st.number_input("Número de Espécimes para Análise Biometrica:", min_value=1, step=1, value=st.session_state.get('num_especimes_anterior_bio', 1), key="num_especimes_bio_input")
        if st.session_state.num_especimes_anterior_bio != num_especimes_bio:
            st.session_state.especimes_data_list = [{}] * num_especimes_bio 
            st.session_state.num_especimes_anterior_bio = num_especimes_bio
        
        for i in range(num_especimes_bio):
            if i >= len(st.session_state.especimes_data_list) : # Garante que a lista tem tamanho suficiente
                 st.session_state.especimes_data_list.append({})
            if not isinstance(st.session_state.especimes_data_list[i], dict): # Garante que é um dict
                 st.session_state.especimes_data_list[i] = {}
            
            current_data = st.session_state.especimes_data_list[i]
            st.markdown(f"--- \n **Espécime {i+1}**")
            # ... (inputs biométricos como antes, usando .get() para defaults) ...
            especime_id_user = st.text_input(f"Identificador do Espécime {i+1}:", value=current_data.get("id_user", f"Espécime_{i+1}"), key=f"id_especime_bio_{i}")
            altura_m_str = st.text_input(f"📏 Altura Total (m) - Espécime {i+1}:", value=current_data.get("altura_m_str", "2.5"), key=f"altura_m_bio_{i}")
            diametro_cm_str = st.text_input(f"📐 Diâmetro (cm) - Espécime {i+1}:", value=current_data.get("diametro_cm_str", "15"), key=f"diametro_cm_bio_{i}")
            area_copa_m2_str = st.text_input(f"🌳 Área da Copa Projetada (m²) - Espécime {i+1}:", value=current_data.get("area_copa_m2_str", "3.0"), key=f"area_copa_m2_bio_{i}")
            galhos_principais = st.number_input(f"🌿 Galhos Estruturais Principais - Espécime {i+1}:", min_value=1, value=current_data.get("galhos_principais", 5), step=1, key=f"galhos_princ_bio_{i}")
            num_folhas_por_galho_estimado = st.number_input(f"🍂 Folhas Médias Estimadas / Galho Principal - Espécime {i+1}:", min_value=1, value=current_data.get("num_folhas_por_galho_estimado", 50), step=5, key=f"num_folhas_galho_est_bio_{i}", help="Parâmetro crucial. Exige amostragem e estudo alométrico em pesquisa real.")
            num_folhas_amostra = st.number_input(f"Quantas folhas de amostra para Espécime {i+1}?",min_value=1, max_value=10, value=current_data.get("num_folhas_amostra", 3),step=1, key=f"num_folhas_amostra_bio_{i}")
            
            folhas_data_temp = current_data.get("folhas_data_list", [("6","12")] * num_folhas_amostra)
            if len(folhas_data_temp) != num_folhas_amostra: folhas_data_temp = [("6","12")] * num_folhas_amostra
            new_folhas_data_list_for_specimen = []
            cols_folhas = st.columns(num_folhas_amostra)
            for j in range(num_folhas_amostra):
                with cols_folhas[j]:
                    st.markdown(f"🍃 F{j+1}"); val_l, val_c = folhas_data_temp[j]
                    l_str = st.text_input(f"L (cm):", value=val_l, key=f"larg_f_bio_{i}_{j}")
                    c_str = st.text_input(f"C (cm):", value=val_c, key=f"comp_f_bio_{i}_{j}")
                    new_folhas_data_list_for_specimen.append((l_str, c_str))
            st.session_state.especimes_data_list[i] = {
                "id_user": especime_id_user, "altura_m_str": altura_m_str, "diametro_cm_str": diametro_cm_str,
                "area_copa_m2_str": area_copa_m2_str, "galhos_principais": galhos_principais,
                "folhas_data_list": new_folhas_data_list_for_specimen, "num_folhas_amostra": num_folhas_amostra,
                "num_folhas_por_galho_estimado": num_folhas_por_galho_estimado
            }

    with st.expander("🌦️ 3. Variáveis Climáticas Médias do Período de Análise", expanded=True):
        temp_c_str = st.text_input("🌡️ Temperatura Média do Ar (°C):", "28.5", key="temp_c_clim")
        umid_perc_str = st.text_input("💧 Umidade Relativa Média do Ar (%):", "55", key="umid_perc_clim")
        rad_mj_m2_dia_str = st.text_input("☀️ Radiação Solar Global Incidente Média Diária (MJ/m²/dia):", "19.5", key="rad_mj_clim")
        vento_m_s_str = st.text_input("🌬️ Velocidade Média do Vento a 2m de Altura (m/s):", "2.2", key="vento_ms_clim")

    with st.expander("🛠️ 4. Coeficientes do Modelo (Demonstrativo/Ajustável)", expanded=False):
        # ... (inputs dos coeficientes, como antes) ...
        st.markdown("**Modelo de Evapotranspiração Empírico:**"); st.caption("Estes coeficientes são para o modelo linear simplificado. Em uma pesquisa de doutorado, seriam calibrados ou o modelo seria substituído.")
        cols_coeffs_et1 = st.columns(2); st.session_state.et_model_coeffs["altura"] = cols_coeffs_et1[0].number_input("Peso Altura:", value=st.session_state.et_model_coeffs["altura"], step=0.01, format="%.2f", key="coeff_alt_cfg"); st.session_state.et_model_coeffs["diametro"] = cols_coeffs_et1[1].number_input("Peso Diâmetro:", value=st.session_state.et_model_coeffs["diametro"], step=0.01, format="%.2f", key="coeff_diam_cfg")
        cols_coeffs_et2 = st.columns(2); st.session_state.et_model_coeffs["copa"] = cols_coeffs_et2[0].number_input("Peso Área Copa:", value=st.session_state.et_model_coeffs["copa"], step=0.01, format="%.2f", key="coeff_copa_cfg"); st.session_state.et_model_coeffs["lai"] = cols_coeffs_et2[1].number_input("Peso LAI:", value=st.session_state.et_model_coeffs["lai"], step=0.01, format="%.2f", key="coeff_lai_cfg")
        cols_coeffs_et3 = st.columns(2); st.session_state.et_model_coeffs["temperatura"] = cols_coeffs_et3[0].number_input("Peso Temperatura:", value=st.session_state.et_model_coeffs["temperatura"], step=0.01, format="%.2f", key="coeff_temp_cfg"); st.session_state.et_model_coeffs["umidade"] = cols_coeffs_et3[1].number_input("Peso Umidade:", value=st.session_state.et_model_coeffs["umidade"], step=0.01, format="%.3f", key="coeff_umid_cfg")
        cols_coeffs_et4 = st.columns(2); st.session_state.et_model_coeffs["radiacao"] = cols_coeffs_et4[0].number_input("Peso Radiação:", value=st.session_state.et_model_coeffs["radiacao"], step=0.001, format="%.3f", key="coeff_rad_cfg"); st.session_state.et_model_coeffs["vento"] = cols_coeffs_et4[1].number_input("Peso Vento:", value=st.session_state.et_model_coeffs["vento"], step=0.001, format="%.3f", key="coeff_vento_cfg")
        st.session_state.et_model_coeffs["fator_escala"] = st.number_input("Fator de Escala ET Geral:", value=st.session_state.et_model_coeffs["fator_escala"], step=0.1, format="%.1f", key="coeff_escala_et_cfg")
        st.markdown("**Modelo de Absorção de Carbono Simplificado:**"); st.caption("Coeficientes altamente hipotéticos. Em pesquisa, seriam derivados de estudos ecofisiológicos detalhados para espécies da Caatinga.")
        st.session_state.carbon_model_coeffs["k_aft_carbon"] = st.number_input("Coef. Carbono via AFT (kg C/m²/dia):", value=st.session_state.carbon_model_coeffs["k_aft_carbon"], step=0.0001, format="%.4f", key="coeff_k_aft_c_cfg", help="Taxa média de fixação de Carbono por área foliar.")
        st.session_state.carbon_model_coeffs["c_et_wue_carbon"] = st.number_input("Coef. Carbono via ET/WUE (kg C/L ET):", value=st.session_state.carbon_model_coeffs["c_et_wue_carbon"], step=0.0001, format="%.4f", key="coeff_c_et_wue_c_cfg", help="Proxy para Eficiência no Uso da Água em termos de Carbono ganho por água perdida.")

    st.markdown("---")
    if st.button("🚀 Executar Simulação e Análise", type="primary", key="run_simulation_main_button", use_container_width=True):
        st.session_state.resultados_modelo = []
        try:
            temp_val = float(temp_c_str); umid_val = float(umid_perc_str)
            rad_val = float(rad_mj_m2_dia_str); vento_val = float(vento_m_s_str)
            if not (0 < umid_val <= 100): st.error("Umidade Relativa deve estar entre 0 (exclusive) e 100%."); st.stop()
        except ValueError: st.error("Erro: Verifique se todas as variáveis climáticas são números válidos."); st.stop()

        for i, especime_input_data in enumerate(st.session_state.especimes_data_list):
            if not especime_input_data or not especime_input_data.get("id_user"): 
                st.warning(f"Dados para o Espécime {i+1} não foram completamente preenchidos ou são inválidos. Pulando.")
                continue
            try:
                altura_m = float(especime_input_data["altura_m_str"])
                diametro_cm = float(especime_input_data["diametro_cm_str"])
                area_copa_m2 = float(especime_input_data["area_copa_m2_str"])
                galhos_p = int(especime_input_data["galhos_principais"])
                num_folhas_gh_est = int(especime_input_data["num_folhas_por_galho_estimado"])
                aft_m2_calc = calculate_area_foliar_total(especime_input_data["folhas_data_list"], galhos_p, num_folhas_gh_est)
                lai_calc = calculate_lai(aft_m2_calc, especime_input_data["area_copa_m2_str"])
                et_litros_dia_calc = predict_evapotranspiration_phd(altura_m, diametro_cm, area_copa_m2, lai_calc, temp_val, umid_val, rad_val, vento_val, st.session_state.et_model_coeffs)
                carbono_aft_kg_dia, carbono_et_wue_kg_dia = estimate_carbon_absorption_phd(aft_m2_calc, et_litros_dia_calc, st.session_state.carbon_model_coeffs)
                st.session_state.resultados_modelo.append({
                    "ID Espécime": especime_input_data["id_user"], "Altura (m)": altura_m,
                    "Diâmetro (cm)": diametro_cm, "Área Copa (m²)": area_copa_m2,
                    "AFT Estimada (m²)": round(aft_m2_calc, 3), "LAI Estimado": lai_calc,
                    "ET Modelo (L/dia)": et_litros_dia_calc,
                    "Carbono (AFT) (kgC/dia)": carbono_aft_kg_dia,
                    "Carbono (ET/WUE) (kgC/dia)": carbono_et_wue_kg_dia
                })
            except KeyError as ke: st.error(f"Erro de chave ao processar Espécime {especime_input_data.get('id_user', f'Índice {i}')}: {ke}. Verifique se todos os campos foram preenchidos."); continue
            except ValueError as ve: st.error(f"Erro de valor ao processar Espécime {especime_input_data.get('id_user', f'Índice {i}')}: {ve}. Verifique os tipos de dados."); continue
            except Exception as e: st.error(f"Erro inesperado processando Espécime {especime_input_data.get('id_user', f'Índice {i}')}: {e}"); continue
        
        if st.session_state.resultados_modelo: st.success(f"Simulação concluída para {len(st.session_state.resultados_modelo)} espécime(s). Veja os resultados à direita.")
        else: st.warning("Nenhum espécime pôde ser processado. Verifique as entradas e mensagens de erro.")


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
                    fig_et, ax_et = plt.subplots(); ax_et.bar(df_resultados["ID Espécime"], df_resultados["ET Modelo (L/dia)"], color='skyblue'); ax_et.set_xlabel("ID do Espécime"); ax_et.set_ylabel("ET Estimada (L/dia)"); ax_et.set_title("Evapotranspiração Estimada por Espécime"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); st.pyplot(fig_et)
                    fig_hist_et, ax_hist_et = plt.subplots(); ax_hist_et.hist(df_resultados["ET Modelo (L/dia)"].dropna(), bins=10, color='lightblue', edgecolor='black'); ax_hist_et.set_xlabel("ET Estimada (L/dia)"); ax_hist_et.set_ylabel("Frequência"); ax_hist_et.set_title("Distribuição da ET Estimada"); st.pyplot(fig_hist_et)
            with col_g2:
                if "Carbono (AFT) (kgC/dia)" in df_resultados.columns:
                    fig_c, ax_c = plt.subplots(); ax_c.bar(df_resultados["ID Espécime"], df_resultados["Carbono (AFT) (kgC/dia)"], color='lightgreen'); ax_c.set_xlabel("ID do Espécime"); ax_c.set_ylabel("Carbono Estimado (kgC/dia - via AFT)"); ax_c.set_title("Absorção de Carbono Estimada (via AFT)"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); st.pyplot(fig_c)
                    fig_hist_c, ax_hist_c = plt.subplots(); ax_hist_c.hist(df_resultados["Carbono (AFT) (kgC/dia)"].dropna(), bins=10, color='green', edgecolor='black'); ax_hist_c.set_xlabel("Carbono Estimado (kgC/dia - via AFT)"); ax_hist_c.set_ylabel("Frequência"); ax_hist_c.set_title("Distribuição do Carbono Estimado (via AFT)"); st.pyplot(fig_hist_c)
            if "LAI Estimado" in df_resultados.columns and not df_resultados["LAI Estimado"].empty:
                st.markdown("#### Boxplot do LAI Estimado entre Espécimes"); fig_box_lai, ax_box_lai = plt.subplots(); ax_box_lai.boxplot(df_resultados["LAI Estimado"].dropna(), patch_artist=True, vert=False); ax_box_lai.set_yticklabels(['LAI']); ax_box_lai.set_xlabel('LAI Estimado'); ax_box_lai.set_title('Distribuição do LAI Estimado'); st.pyplot(fig_box_lai)

        st.markdown("---")
        st.subheader("🔬 Contraprova Experimental e Análise Estatística Comparativa")
        
        # CORREÇÃO: Usar o valor de session_state como default para o number_input
        num_medicoes_exp_atual = st.number_input("Número de Medições Experimentais por Espécime:", 
                                                 min_value=1, 
                                                 value=st.session_state.num_medicoes_exp_anterior, # Usar o valor do estado
                                                 step=1, 
                                                 key="num_med_exp_comp_input")
        tempo_coleta_exp_horas_str = st.text_input("Tempo de Coleta para Cada Medição Experimental (horas):", 
                                                 st.session_state.get('tempo_coleta_exp_horas_str_anterior', "24"), 
                                                 key="tempo_coleta_exp_h_comp_input")

        # Lógica de atualização do st.session_state.contraprovas_data
        if df_resultados.shape[0] > 0: # Só faz sentido se houver resultados do modelo
            if st.session_state.num_especimes_anterior_comp != df_resultados.shape[0] or \
               st.session_state.num_medicoes_exp_anterior != num_medicoes_exp_atual:
                st.session_state.contraprovas_data = {} 
                for _, row_modelo in df_resultados.iterrows():
                    id_especime_modelo = row_modelo["ID Espécime"]
                    st.session_state.contraprovas_data[id_especime_modelo] = ["0"] * num_medicoes_exp_atual
                st.session_state.num_especimes_anterior_comp = df_resultados.shape[0]
                st.session_state.num_medicoes_exp_anterior = num_medicoes_exp_atual # Atualiza o estado com o novo valor
        
        st.session_state['tempo_coleta_exp_horas_str_anterior'] = tempo_coleta_exp_horas_str

        for _, row_modelo in df_resultados.iterrows():
            id_especime_modelo = row_modelo["ID Espécime"]
            if id_especime_modelo not in st.session_state.contraprovas_data or \
               len(st.session_state.contraprovas_data[id_especime_modelo]) != num_medicoes_exp_atual:
                st.session_state.contraprovas_data[id_especime_modelo] = ["0"] * num_medicoes_exp_atual
            
            with st.container(border=True):
                st.markdown(f"**Valores Experimentais para Espécime: {id_especime_modelo}**")
                medicoes_atuais_str_input = [] # Lista temporária para os inputs desta renderização
                cols_med_exp_display = st.columns(num_medicoes_exp_atual)
                for k in range(num_medicoes_exp_atual):
                    with cols_med_exp_display[k]:
                        # Garante que o valor do estado é usado para o widget
                        default_val_med = st.session_state.contraprovas_data[id_especime_modelo][k] if k < len(st.session_state.contraprovas_data[id_especime_modelo]) else "0"
                        val_exp_input = st.text_input(
                            f"Medição {k+1} (mL):",
                            value=default_val_med,
                            key=f"med_exp_val_{id_especime_modelo}_{k}"
                        )
                        medicoes_atuais_str_input.append(val_exp_input)
                # Atualiza o session_state com os valores que acabaram de ser renderizados/inputados
                st.session_state.contraprovas_data[id_especime_modelo] = medicoes_atuais_str_input 

        tipo_teste_estatistico = st.selectbox(
            "Escolha o Teste Estatístico para Comparação (Modelo vs. Experimental):",
            ("Teste t de Student (1 amostra)", "Teste de Wilcoxon (Signed-Rank)", "Diferença Absoluta e Percentual"),
            key="tipo_teste_stat_phd_select_comp"
        )

        if st.button("🔄 Comparar Modelo com Dados Experimentais", key="run_comparison_main_button", use_container_width=True):
            if not st.session_state.contraprovas_data: st.warning("Por favor, insira os dados experimentais."); st.stop()
            try:
                tempo_coleta_h = float(tempo_coleta_exp_horas_str)
                if tempo_coleta_h <= 0: st.error("Tempo de coleta experimental deve ser positivo."); st.stop()
            except ValueError: st.error("Tempo de coleta experimental inválido."); st.stop()

            st.markdown("### Resultados da Comparação Estatística:")
            all_exp_means_list, all_model_preds_list, valid_comparison_count = [], [], 0

            for _, row_modelo in df_resultados.iterrows():
                id_especime = row_modelo["ID Espécime"]
                et_modelo_val = row_modelo["ET Modelo (L/dia)"]

                if id_especime not in st.session_state.contraprovas_data:
                    st.warning(f"Dados experimentais não encontrados para o espécime {id_especime}. Pulando."); continue
                
                medicoes_exp_str_list = st.session_state.contraprovas_data[id_especime]
                medicoes_exp_ml_float_validas = []
                valores_exp_invalidos_flag = False

                for idx_med, med_str_val in enumerate(medicoes_exp_str_list):
                    try:
                        med_str_cleaned = med_str_val.strip()
                        if not med_str_cleaned: st.warning(f"Medição {idx_med+1} (Espécime {id_especime}) está vazia. Ignorando."); continue
                        med_float = float(med_str_cleaned)
                        medicoes_exp_ml_float_validas.append(med_float)
                    except ValueError: st.error(f"Valor experimental '{med_str_val}' (Medição {idx_med+1}, Espécime {id_especime}) não é um número válido."); valores_exp_invalidos_flag = True; break 
                
                if valores_exp_invalidos_flag: st.warning(f"Análise para Espécime {id_especime} interrompida devido a valores experimentais inválidos."); continue
                if not medicoes_exp_ml_float_validas: st.warning(f"Nenhum dado experimental válido para Espécime {id_especime}. Pulando."); continue

                medicoes_exp_L_dia = [(m_ml / 1000.0) / (tempo_coleta_h / 24.0) for m_ml in medicoes_exp_ml_float_validas]
                if not medicoes_exp_L_dia: st.warning(f"Nenhuma medição experimental processável para Espécime {id_especime}."); continue
                
                media_exp_L_dia = np.mean(medicoes_exp_L_dia)
                all_exp_means_list.append(media_exp_L_dia); all_model_preds_list.append(et_modelo_val); valid_comparison_count += 1
                
                st.markdown(f"#### Análise para Espécime: {id_especime}")
                st.write(f"- ET Prevista pelo Modelo: {et_modelo_val:.2f} L/dia")
                st.write(f"- ET Média Experimental: {media_exp_L_dia:.2f} L/dia (Baseado em {len(medicoes_exp_L_dia)} medições válidas: {[f'{x:.2f}' for x in medicoes_exp_L_dia]})")
                
                p_valor_teste_atual = None
                if tipo_teste_estatistico == "Teste t de Student (1 amostra)":
                    if len(medicoes_exp_L_dia) < 2 or len(set(medicoes_exp_L_dia)) < 2 : st.warning("Teste t requer pelo menos 2 medições com variabilidade.")
                    else: stat_t, p_valor_teste_atual = stats.ttest_1samp(medicoes_exp_L_dia, et_modelo_val); st.write(f"  - Teste t: Estatística t = {stat_t:.3f}, p-valor = {p_valor_teste_atual:.4f}")
                elif tipo_teste_estatistico == "Teste de Wilcoxon (Signed-Rank)":
                    if len(medicoes_exp_L_dia) < 1: st.warning("Teste de Wilcoxon requer pelo menos uma medição.")
                    else:
                        diffs = np.array(medicoes_exp_L_dia) - et_modelo_val
                        if np.all(diffs == 0) and len(diffs)>0 : st.warning("Teste de Wilcoxon não aplicável: todas as diferenças entre modelo e experimento são zero.")
                        elif len(diffs) == 0: st.warning("Teste de Wilcoxon não aplicável: não há dados para calcular as diferenças.")
                        else: 
                            try:
                                if len(diffs[diffs != 0]) == 0 and len(diffs) > 0: st.warning("Teste de Wilcoxon não aplicável: todas as diferenças são zero (após remover zeros).")
                                elif len(diffs) > 0: stat_w, p_valor_teste_atual = stats.wilcoxon(diffs, alternative='two-sided', zero_method='wilcox'); st.write(f"  - Teste de Wilcoxon: Estatística W = {stat_w:.3f}, p-valor = {p_valor_teste_atual:.4f}")
                                else: st.warning("Não há dados suficientes para o Teste de Wilcoxon após o processamento das diferenças.")
                            except ValueError as e_wilcoxon: st.warning(f"  - Teste de Wilcoxon não pôde ser calculado: {e_wilcoxon}. Pode ser devido a poucos dados ou empates excessivos.")
                diferenca_abs = abs(media_exp_L_dia - et_modelo_val); diferenca_perc = (diferenca_abs / media_exp_L_dia) * 100 if media_exp_L_dia != 0 else float('inf')
                st.write(f"  - Diferença Absoluta: {diferenca_abs:.2f} L/dia"); st.write(f"  - Diferença Percentual: {diferenca_perc:.2f}%")
                if p_valor_teste_atual is not None:
                    alpha = 0.05
                    if p_valor_teste_atual < alpha: st.error(f"  - Conclusão: Diferença estatisticamente significativa (p < {alpha}). O modelo difere da média experimental para este espécime.")
                    else: st.success(f"  - Conclusão: Diferença não estatisticamente significativa (p ≥ {alpha}). Não há evidência forte de que o modelo difere da média experimental.")
            
            if valid_comparison_count > 1:
                st.markdown("--- \n ### Análise Global de Desempenho do Modelo (Comparativo)")
                exp_means_np_global = np.array(all_exp_means_list); model_preds_np_global = np.array(all_model_preds_list)
                rmse_global = np.sqrt(mean_squared_error(exp_means_np_global, model_preds_np_global)); mae_global = mean_absolute_error(exp_means_np_global, model_preds_np_global)
                try: r2_global = r2_score(exp_means_np_global, model_preds_np_global)
                except ValueError: r2_global = np.nan
                st.write(f"**Métricas Globais de Comparação:**"); st.write(f"- RMSE Global: {rmse_global:.3f} L/dia"); st.write(f"- MAE Global: {mae_global:.3f} L/dia"); st.write(f"- R² Global: {r2_global:.3f}" if not np.isnan(r2_global) else "- R² Global: N/A (requer mais variabilidade/pontos)")
                fig_scatter_global, ax_scatter_global = plt.subplots(); ax_scatter_global.scatter(model_preds_np_global, exp_means_np_global, alpha=0.7, edgecolors='k'); 
                min_val_plot = min(min(model_preds_np_global), min(exp_means_np_global)) if len(model_preds_np_global)>0 and len(exp_means_np_global)>0 else 0
                max_val_plot = max(max(model_preds_np_global), max(exp_means_np_global)) if len(model_preds_np_global)>0 and len(exp_means_np_global)>0 else 1
                ax_scatter_global.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', label="Linha 1:1 (Ideal)"); ax_scatter_global.set_xlabel("ET Prevista pelo Modelo (L/dia)"); ax_scatter_global.set_ylabel("ET Média Experimental (L/dia)"); ax_scatter_global.set_title("Comparação Global: Modelo vs. Experimental"); ax_scatter_global.legend(); ax_scatter_global.grid(True); st.pyplot(fig_scatter_global)
            elif df_resultados.shape[0] > 0: st.info("Análise global comparativa requer dados experimentais válidos de pelo menos dois espécimes.")

# ---------------------------------------------------------------
# Seções Explicativas (LaTeX corrigido com r"...")
# ---------------------------------------------------------------
st.sidebar.title("Navegação e Informações")
st.sidebar.info(f"""
**Plataforma de Simulação e Análise para Pesquisa de Doutorado**
Foco: Evapotranspiração e Dinâmica de Carbono em Ecossistemas Semiáridos (Crateús, Ceará).
Versão: 1.0.3 (Correção Erro de Inicialização e LaTeX)
Data: {pd.Timestamp.now().strftime('%Y-%m-%d')}
""")
with st.sidebar.expander("⚠️ Limitações e Próximos Passos (PhD)", expanded=False):
    st.markdown(r"""Esta ferramenta demonstra conceitos e um fluxo de análise. Para uma tese de doutorado:
    - **Modelo de ET:** Implementar modelos biofísicos robustos (e.g., Penman-Monteith com calibração de condutância estomática) ou modelos de Machine Learning (Random Forest, Redes Neurais) treinados com dados de campo extensivos de Crateús.
    - **Modelo de Carbono:** Desenvolver/aplicar modelos ecofisiológicos de fotossíntese e alocação de carbono (e.g., Farquhar, modelos baseados em LUE - Light Use Efficiency), calibrados para espécies da Caatinga.
    - **Dados de Campo:** Coleta extensiva de dados biométricos, microclimáticos, de fluxo de seiva (para ET), e trocas gasosas (para fotossíntese/respiração) em Crateús.
    - **Análise de Incerteza e Sensibilidade:** Aplicar métodos formais (e.g., Monte Carlo, GSA) para os modelos desenvolvidos.
    - **Validação Cruzada:** Rigorosa para modelos de ML.
    - **q-Estatística:** Investigar se as distribuições de variáveis ou erros do modelo exibem características não extensivas que justifiquem a aplicação da Estatística de Tsallis para uma descrição mais precisa.
    - **Escalonamento Espacial:** Utilizar sensoriamento remoto e SIG para extrapolar estimativas para a paisagem de Crateús.
    """)

with st.expander("🔍 Fundamentos Teóricos e Metodológicos (Discussão para Banca)", expanded=False):
    st.markdown(r"### 📚 Modelo de Evapotranspiração (ET)")
    st.markdown(r"""A ET é um componente crucial do ciclo hidrológico e do balanço energético, especialmente em regiões semiáridas como Crateús.
    - **Modelo Empírico Simplificado (Usado Aqui):** Uma função linear ponderada de variáveis biométricas e climáticas.
        - **Vantagens:** Simplicidade, fácil implementação, útil para análises exploratórias iniciais.
        - **Desvantagens para PhD:** Falta de base biofísica robusta, coeficientes arbitrários sem calibração, não captura interações complexas nem respostas não lineares.
    - **Abordagem de Doutorado (Recomendada):**
        1.  **Modelo de Penman-Monteith (FAO-56 PM):** Padrão ouro, combina balanço de energia com termos de transporte aerodinâmico e resistência superficial (condutância estomática, $g_s$). Requer calibração de $g_s$ para espécies locais da Caatinga, considerando fatores como déficit de pressão de vapor (VPD), radiação, umidade do solo.
            $ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T+273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}$ (para cultura de referência)
            Para ET real ($ET_c$), $ET_c = K_c ET_0$ ou modelagem direta de $g_s$.
        2.  **Modelos de Machine Learning:** Random Forest, Gradient Boosting, Redes Neurais, treinados com dados de ET medidos (e.g., fluxo de seiva, lisímetros, covariância de vórtices) e preditores ambientais/biométricos. Exigem grandes conjuntos de dados para treinamento e validação.
    """)

    st.markdown(r"### 🍂 Área Foliar Total (AFT) e Índice de Área Foliar (LAI)")
    st.markdown(r"""
    - **AFT:** Área total de superfície foliar fotossinteticamente ativa. Crucial para trocas gasosas.
    - **LAI:** AFT por unidade de área de solo ($LAI = AFT/A_{copa}$). Adimensional, indica a densidade do dossel.
    - **Estimativa (Usada Aqui):** Baseada na área média de folhas de amostra e estimativas do número de folhas. Altamente simplificado.
    - **Abordagem de Doutorado:** (conteúdo mantido)
    """)
    st.latex(r'''\text{AFT (m}^2\text{)} = \sum_{\text{folhas}} (\text{área da folha em m}^2\text{)}''') # Exemplo isolado
    st.latex(r'''\text{LAI} = \frac{\text{Área Foliar Total (m}^2\text{)}}{\text{Área da Copa Projetada no Solo (m}^2\text{)}}''') # Exemplo isolado

    st.markdown(r"### 🌳 Estimativa de Absorção/Captura de Carbono")
    st.markdown(r"""A fixação de carbono via fotossíntese é o principal mecanismo de entrada de C nos ecossistemas terrestres.
    - **Estimativa Simplificada (Usada Aqui):** Coeficientes fixos multiplicados por AFT ou ET. Meramente ilustrativo.
    - **Abordagem de Doutorado:**
        1.  **Modelos Ecofisiológicos de Fotossíntese:** Modelo de Farquhar, von Caemmerer & Berry (FvCB) para fotossíntese da folha...
            $A = \min(A_c, A_j, A_p) - R_d$ 
        2.  **Modelos Baseados em Eficiência no Uso da Luz (LUE):** $NPP = APAR \times LUE_{eco}$ ...
        3.  **Balanço de Carbono do Ecossistema:** ... $NPP \approx GPP - R_a$ ...
        4.  **Alocação de Biomassa:** ...
        5.  **Eficiência no Uso da Água (WUE):** $WUE = A / E$ ...
    """)

    st.markdown(r"### 🔬 Análise Estatística e Validação de Modelos (PhD)")
    st.markdown(r""" (conteúdo mantido) """)

    st.markdown(r"### ⚛️ q-Estatística (Estatística de Tsallis) em Pesquisas Ecológicas")
    st.markdown(r"""A q-estatística generaliza a estatística de Boltzmann-Gibbs...
    -   A **q-Exponencial** pode descrever processos... Sua forma funcional pode ser: $f(x; q, \beta) = N \exp_q(-\beta x) = N [1 - (1-q) \beta x]_+^{1/(1-q)}$
    """) # Todas as fórmulas LaTeX aqui estão dentro de markdown r"""..."""

with st.expander("🌳 Estimativa Simplificada de Absorção/Captura de Carbono (Conceitual Detalhado)", expanded=False):
    st.latex(r'''\text{Carbono}_{\text{AFT}} \text{(kg C/dia)} \approx k_{\text{AFT}} \times \text{AFT (m}^2\text{)}''')
    st.latex(r'''\text{Carbono}_{\text{ET/WUE}} \text{(kg C/dia)} \approx k_{\text{ET/WUE}} \times \text{ET (litros/dia)}''')
    st.markdown(r"""onde \( k_{\text{AFT}} \) e \( k_{\text{ET/WUE}} \) ... (conteúdo mantido) """)

st.markdown("---")
st.caption(f"Plataforma de Simulação Avançada - Versão para Discussão em Banca de Doutorado. {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
