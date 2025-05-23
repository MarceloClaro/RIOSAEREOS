import streamlit as st
from PIL import Image
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------------------------------------------------------
# I. FUNDAMENTOS DA QUALIDADE DE CÓDIGO: Inicialização e Configuração
# Aderência às Convenções de Estilo (PEP 8) implicitamente seguida.
# Nomenclatura, Comentários e Documentação serão aplicados ao longo do código.
# -----------------------------------------------------------------------------

# 1. Configurações da Página e Estado da Sessão
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="PhD EvapoCarbon Estimator",
    page_icon="🌳"
)

# Constantes para valores padrão e chaves de estado (PEP 8 para constantes)
DEFAULT_NUM_MEDICOES_EXP = 3
DEFAULT_NUM_ESPECIMES_BIO = 1
SESSION_KEY_RESULTADOS_MODELO = "resultados_modelo"
SESSION_KEY_UPLOADED_IMAGE = "uploaded_image"
SESSION_KEY_ET_COEFFS = "et_model_coeffs"
SESSION_KEY_CARBON_COEFFS = "carbon_model_coeffs"
SESSION_KEY_ESPECIMES_DATA = "especimes_data_list"
SESSION_KEY_CONTRAPROVAS_DATA = "contraprovas_data"
SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR = "num_medicoes_exp_anterior"
SESSION_KEY_NUM_ESPECIMES_BIO_ANTERIOR = "num_especimes_anterior_bio"
SESSION_KEY_NUM_ESPECIMES_COMP_ANTERIOR = "num_especimes_anterior_comp" # Para lógica de reset da contraprova
SESSION_KEY_TEMPO_COLETA_ANT = 'tempo_coleta_exp_horas_str_anterior'

# Inicialização robusta do st.session_state (Princípio de UX: lembrar entradas)
# (Prática da Seção IV.B do documento de melhores práticas)
if SESSION_KEY_RESULTADOS_MODELO not in st.session_state:
    st.session_state[SESSION_KEY_RESULTADOS_MODELO] = []
if SESSION_KEY_UPLOADED_IMAGE not in st.session_state:
    st.session_state[SESSION_KEY_UPLOADED_IMAGE] = None
if SESSION_KEY_ET_COEFFS not in st.session_state:
    st.session_state[SESSION_KEY_ET_COEFFS] = {
        "altura": 0.3, "diametro": 0.2, "copa": 0.1, "lai": 0.2,
        "temperatura": 0.1, "umidade": 0.05, "radiacao": 0.03, "vento": 0.02,
        "fator_escala": 10.0
    }
if SESSION_KEY_CARBON_COEFFS not in st.session_state:
    st.session_state[SESSION_KEY_CARBON_COEFFS] = {
        "k_aft_carbon": 0.005,
        "c_et_wue_carbon": 0.002
    }
if SESSION_KEY_ESPECIMES_DATA not in st.session_state:
    st.session_state[SESSION_KEY_ESPECIMES_DATA] = []
if SESSION_KEY_CONTRAPROVAS_DATA not in st.session_state:
    st.session_state[SESSION_KEY_CONTRAPROVAS_DATA] = {}

# Inicialização SEGURA para o widget problemático (Seção II.A e IV.B)
if SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR not in st.session_state:
    st.session_state[SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR] = DEFAULT_NUM_MEDICOES_EXP
elif not isinstance(st.session_state[SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR], (int, float)) or \
     st.session_state[SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR] < 1:
    st.session_state[SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR] = DEFAULT_NUM_MEDICOES_EXP

if SESSION_KEY_NUM_ESPECIMES_BIO_ANTERIOR not in st.session_state:
    st.session_state[SESSION_KEY_NUM_ESPECIMES_BIO_ANTERIOR] = DEFAULT_NUM_ESPECIMES_BIO
if SESSION_KEY_NUM_ESPECIMES_COMP_ANTERIOR not in st.session_state:
    st.session_state[SESSION_KEY_NUM_ESPECIMES_COMP_ANTERIOR] = 0
if SESSION_KEY_TEMPO_COLETA_ANT not in st.session_state:
    st.session_state[SESSION_KEY_TEMPO_COLETA_ANT] = "24"


# -----------------------------------------------------------------------------
# 2. Funções Científicas e de Cálculo (Modularidade - Seção I.C)
# Docstrings adicionadas para clareza (Seção I.B)
# -----------------------------------------------------------------------------
def calculate_area_foliar_total(folhas_data_list: list, galhos_principais: int, num_folhas_estimado_por_galho: int) -> float:
    """
    Calcula a Área Foliar Total (AFT) estimada em metros quadrados.

    A estimativa é baseada na área média de folhas de amostra fornecidas,
    multiplicada pelo número estimado de folhas por galho principal e
    pelo número de galhos principais. Trata-se de uma simplificação que
    exigiria calibração e métodos mais sofisticados em pesquisa aprofundada.

    Args:
        folhas_data_list (list): Lista de tuplas, onde cada tupla contém
                                 (largura_cm_str, comprimento_cm_str) para uma folha de amostra.
        galhos_principais (int): Número estimado de galhos estruturais principais.
        num_folhas_estimado_por_galho (int): Número médio estimado de folhas por galho principal.

    Returns:
        float: Área Foliar Total estimada em m². Retorna 0.0 se entradas forem inválidas.
    """
    if not folhas_data_list or galhos_principais <= 0 or num_folhas_estimado_por_galho <= 0:
        return 0.0

    soma_area_folhas_exemplo_m2 = 0.0
    num_folhas_validas = 0
    for idx_folha, (largura_str, comprimento_str) in enumerate(folhas_data_list):
        try:
            # Validação de entrada (Seção II.A)
            largura_cm = float(largura_str)
            comprimento_cm = float(comprimento_str)
            if largura_cm <= 0 or comprimento_cm <= 0:
                st.warning(f"Dimensões da folha de amostra {idx_folha+1} ({largura_cm}cm x {comprimento_cm}cm) devem ser positivas. Ignorando esta folha.")
                continue
            soma_area_folhas_exemplo_m2 += (largura_cm / 100.0) * (comprimento_cm / 100.0)
            num_folhas_validas += 1
        except ValueError: # Tratamento de Erro (Seção II.B)
            st.warning(f"Valor inválido ('{largura_str}' ou '{comprimento_str}') para dimensão da folha de amostra {idx_folha+1}. Ignorando esta folha.")
            continue

    if num_folhas_validas == 0:
        return 0.0

    area_media_folha_m2 = soma_area_folhas_exemplo_m2 / num_folhas_validas
    area_foliar_total_m2 = area_media_folha_m2 * num_folhas_estimado_por_galho * galhos_principais
    return area_foliar_total_m2

def calculate_lai(area_foliar_total_m2: float, area_copa_m2_str: str) -> float:
    """
    Calcula o Índice de Área Foliar (LAI).

    Args:
        area_foliar_total_m2 (float): Área Foliar Total em m².
        area_copa_m2_str (str): Área da copa projetada no solo em m² (como string).

    Returns:
        float: LAI calculado. Retorna 0.0 se a área da copa for inválida ou zero.
    """
    try:
        area_copa_m2 = float(area_copa_m2_str)
        if area_copa_m2 <= 0:
            st.warning(f"Área da copa ({area_copa_m2} m²) deve ser positiva para cálculo do LAI.")
            return 0.0
        lai = area_foliar_total_m2 / area_copa_m2
        return round(lai, 3)
    except ValueError: # Tratamento de Erro (Seção II.B)
        st.warning(f"Valor inválido '{area_copa_m2_str}' para área da copa. Não foi possível calcular o LAI.")
        return 0.0
    except ZeroDivisionError: # Tratamento de Erro (Seção II.B)
        st.warning("Área da copa é zero. Não foi possível calcular o LAI.")
        return 0.0

def predict_evapotranspiration_phd(altura_m: float, diametro_cm: float, area_copa_m2: float, lai: float,
                                   temperatura_c: float, umidade_perc: float, radiacao_mj_m2_dia: float,
                                   vento_m_s: float, coeffs: dict, image_data=None) -> float:
    """
    Prevê a Evapotranspiração (ET) em litros/dia usando um modelo linear ponderado.
    NOTA: Este é um modelo empírico simplificado para fins demonstrativos.
    Pesquisa de doutorado exigiria modelos biofísicos (e.g., Penman-Monteith)
    ou de Machine Learning, devidamente calibrados e validados.

    Args:
        (diversos floats para parâmetros biométricos e climáticos)
        coeffs (dict): Dicionário contendo os pesos e fator de escala para o modelo.

    Returns:
        float: ET estimada em litros/dia.
    """
    diametro_m = diametro_cm / 100.0
    umidade_frac = umidade_perc / 100.0
    # Validação de intervalo para umidade (Seção II.A)
    if not (0 <= umidade_frac <= 1):
        st.error(f"Valor de umidade ({umidade_perc}%) resultou em fração inválida ({umidade_frac}). Verifique a entrada.")
        # Decide como tratar: parar, retornar NaN, ou usar valor padrão. Para simplicidade, pode-se prosseguir com valor clampado.
        umidade_frac = np.clip(umidade_frac, 0, 1)


    et_calc = (
        altura_m * coeffs["altura"] +
        diametro_m * coeffs["diametro"] +
        area_copa_m2 * coeffs["copa"] +
        lai * coeffs["lai"] +
        temperatura_c * coeffs["temperatura"] +
        umidade_frac * coeffs["umidade"] + # Usar umidade_frac validada/clampada
        radiacao_mj_m2_dia * coeffs["radiacao"] +
        vento_m_s * coeffs["vento"]
    ) * coeffs["fator_escala"]
    return round(et_calc, 2)

def estimate_carbon_absorption_phd(area_foliar_total_m2: float, et_litros_dia: float, coeffs_carbon: dict, especie_info=None) -> tuple[float, float]:
    """
    Estima a absorção de carbono (kg C/dia) de forma altamente simplificada.
    NOTA: Coeficientes são hipotéticos. Pesquisa de doutorado exigiria
    modelagem ecofisiológica detalhada e específica para as espécies e local.

    Args:
        area_foliar_total_m2 (float): AFT em m².
        et_litros_dia (float): ET estimada em L/dia.
        coeffs_carbon (dict): Dicionário com coeficientes 'k_aft_carbon' e 'c_et_wue_carbon'.

    Returns:
        tuple[float, float]: Estimativas de carbono via AFT e via ET/WUE, em kg C/dia.
    """
    carbono_via_aft = area_foliar_total_m2 * coeffs_carbon["k_aft_carbon"]
    carbono_via_et_wue = et_litros_dia * coeffs_carbon["c_et_wue_carbon"]
    return round(carbono_via_aft, 4), round(carbono_via_et_wue, 4)

# -----------------------------------------------------------------------------
# 3. Interface Streamlit (Layout e Widgets - Seção IV)
# -----------------------------------------------------------------------------
st.title("🌳 Plataforma Avançada de Estimativa de Evapotranspiração e Análise de Carbono para Ecossistemas Semiáridos")
st.subheader("Foco: Região de Crateús, Ceará, Brasil - Ferramenta de Suporte à Pesquisa de Doutorado")
st.markdown("---")

left_column, right_column = st.columns([2, 3]) # Proporção ajustada para melhor layout

with left_column:
    st.header("⚙️ Entradas e Configurações do Modelo")
    # ... (Seções 1, 2, 3, 4 da left_column como na versão anterior, com chaves atualizadas se necessário)
    with st.expander("🖼️ 1. Identificação Visual do Espécime (Opcional)", expanded=True):
        uploaded_file = st.file_uploader("Carregar imagem do espécime (JPG/PNG)", type=["jpg", "png"], key="img_uploader_main_v6")
        if uploaded_file is not None:
            try: st.session_state[SESSION_KEY_UPLOADED_IMAGE] = Image.open(uploaded_file)
            except Exception as e: st.error(f"Erro ao carregar imagem: {e}")
        if st.session_state[SESSION_KEY_UPLOADED_IMAGE]: st.image(st.session_state[SESSION_KEY_UPLOADED_IMAGE], caption="Imagem Carregada", use_column_width=True)

    with st.expander("🌿 2. Dados Biométricos e Estruturais do Espécime", expanded=True):
        num_especimes_bio_widget = st.number_input("Número de Espécimes para Análise Biometrica:", min_value=1, step=1, value=st.session_state[SESSION_KEY_NUM_ESPECIMES_BIO_ANTERIOR], key="num_especimes_bio_input_v6")
        if st.session_state[SESSION_KEY_NUM_ESPECIMES_BIO_ANTERIOR] != num_especimes_bio_widget or not st.session_state[SESSION_KEY_ESPECIMES_DATA] or len(st.session_state[SESSION_KEY_ESPECIMES_DATA]) != num_especimes_bio_widget:
            st.session_state[SESSION_KEY_ESPECIMES_DATA] = [{}] * num_especimes_bio_widget
            st.session_state[SESSION_KEY_NUM_ESPECIMES_BIO_ANTERIOR] = num_especimes_bio_widget
        
        for i in range(num_especimes_bio_widget): # Usar o valor do widget para o loop
            if not isinstance(st.session_state[SESSION_KEY_ESPECIMES_DATA][i], dict): st.session_state[SESSION_KEY_ESPECIMES_DATA][i] = {}
            current_data = st.session_state[SESSION_KEY_ESPECIMES_DATA][i]
            st.markdown(f"--- \n **Espécime {i+1}**")
            # Nomes de variáveis mais descritivos (Seção I.B)
            especime_id_user_input = st.text_input(f"Identificador do Espécime {i+1}:", value=current_data.get("id_user", f"Espécime_{i+1}"), key=f"id_especime_bio_v6_{i}")
            altura_m_str_input = st.text_input(f"📏 Altura Total (m) - Espécime {i+1}:", value=current_data.get("altura_m_str", "2.5"), key=f"altura_m_bio_v6_{i}")
            # ... (restante dos inputs biométricos, usando _input para clareza) ...
            diametro_cm_str_input = st.text_input(f"📐 Diâmetro (cm) - Espécime {i+1}:", value=current_data.get("diametro_cm_str", "15"), key=f"diametro_cm_bio_v6_{i}")
            area_copa_m2_str_input = st.text_input(f"🌳 Área da Copa Projetada (m²) - Espécime {i+1}:", value=current_data.get("area_copa_m2_str", "3.0"), key=f"area_copa_m2_bio_v6_{i}")
            galhos_principais_input = st.number_input(f"🌿 Galhos Estruturais Principais - Espécime {i+1}:", min_value=1, value=current_data.get("galhos_principais", 5), step=1, key=f"galhos_princ_bio_v6_{i}")
            num_folhas_por_galho_estimado_input = st.number_input(f"🍂 Folhas Médias Estimadas / Galho Principal - Espécime {i+1}:", min_value=1, value=current_data.get("num_folhas_por_galho_estimado", 50), step=5, key=f"num_folhas_galho_est_bio_v6_{i}", help="Parâmetro crucial. Exige amostragem e estudo alométrico em pesquisa real.")
            num_folhas_amostra_input = st.number_input(f"Quantas folhas de amostra para Espécime {i+1}?",min_value=1, max_value=10, value=current_data.get("num_folhas_amostra", 3),step=1, key=f"num_folhas_amostra_bio_v6_{i}")
            
            folhas_data_temp_input = current_data.get("folhas_data_list", [("6","12")] * num_folhas_amostra_input)
            if len(folhas_data_temp_input) != num_folhas_amostra_input: folhas_data_temp_input = [("6","12")] * num_folhas_amostra_input
            new_folhas_data_list_for_specimen_input = []
            cols_folhas_input = st.columns(num_folhas_amostra_input)
            for j_input in range(num_folhas_amostra_input):
                with cols_folhas_input[j_input]:
                    st.markdown(f"🍃 F{j_input+1}"); val_l_input, val_c_input = folhas_data_temp_input[j_input]
                    l_str_input = st.text_input(f"L (cm):", value=val_l_input, key=f"larg_f_bio_v6_{i}_{j_input}")
                    c_str_input = st.text_input(f"C (cm):", value=val_c_input, key=f"comp_f_bio_v6_{i}_{j_input}")
                    new_folhas_data_list_for_specimen_input.append((l_str_input, c_str_input))
            # Atualizar o dicionário no st.session_state
            st.session_state[SESSION_KEY_ESPECIMES_DATA][i] = {"id_user": especime_id_user_input, "altura_m_str": altura_m_str_input, "diametro_cm_str": diametro_cm_str_input, "area_copa_m2_str": area_copa_m2_str_input, "galhos_principais": galhos_principais_input, "folhas_data_list": new_folhas_data_list_for_specimen_input, "num_folhas_amostra": num_folhas_amostra_input, "num_folhas_por_galho_estimado": num_folhas_por_galho_estimado_input}

    with st.expander("🌦️ 3. Variáveis Climáticas Médias do Período de Análise", expanded=True):
        # ... (inputs climáticos como antes, usando _input para clareza e chaves _v6) ...
        temp_c_str_input = st.text_input("🌡️ Temperatura Média do Ar (°C):", st.session_state.get("temp_c_str_input_persist", "28.5"), key="temp_c_clim_v6")
        umid_perc_str_input = st.text_input("💧 Umidade Relativa Média do Ar (%):", st.session_state.get("umid_perc_str_input_persist", "55"), key="umid_perc_clim_v6")
        rad_mj_m2_dia_str_input = st.text_input("☀️ Radiação Solar Global Incidente Média Diária (MJ/m²/dia):", st.session_state.get("rad_mj_m2_dia_str_input_persist", "19.5"), key="rad_mj_clim_v6")
        vento_m_s_str_input = st.text_input("🌬️ Velocidade Média do Vento a 2m de Altura (m/s):", st.session_state.get("vento_m_s_str_input_persist", "2.2"), key="vento_ms_clim_v6")
        # Persistir valores climáticos para melhor UX
        st.session_state.temp_c_str_input_persist = temp_c_str_input
        st.session_state.umid_perc_str_input_persist = umid_perc_str_input
        st.session_state.rad_mj_m2_dia_str_input_persist = rad_mj_m2_dia_str_input
        st.session_state.vento_m_s_str_input_persist = vento_m_s_str_input


    with st.expander("🛠️ 4. Coeficientes do Modelo (Demonstrativo/Ajustável)", expanded=False):
        # ... (inputs dos coeficientes como antes, usando _v6 para chaves) ...
        st.markdown("**Modelo de Evapotranspiração Empírico:**"); st.caption("Estes coeficientes são para o modelo linear simplificado...")
        cols_coeffs_et1 = st.columns(2); st.session_state[SESSION_KEY_ET_COEFFS]["altura"] = cols_coeffs_et1[0].number_input("Peso Altura:", value=st.session_state[SESSION_KEY_ET_COEFFS]["altura"], step=0.01, format="%.2f", key="coeff_alt_cfg_v6"); st.session_state[SESSION_KEY_ET_COEFFS]["diametro"] = cols_coeffs_et1[1].number_input("Peso Diâmetro:", value=st.session_state[SESSION_KEY_ET_COEFFS]["diametro"], step=0.01, format="%.2f", key="coeff_diam_cfg_v6")
        cols_coeffs_et2 = st.columns(2); st.session_state[SESSION_KEY_ET_COEFFS]["copa"] = cols_coeffs_et2[0].number_input("Peso Área Copa:", value=st.session_state[SESSION_KEY_ET_COEFFS]["copa"], step=0.01, format="%.2f", key="coeff_copa_cfg_v6"); st.session_state[SESSION_KEY_ET_COEFFS]["lai"] = cols_coeffs_et2[1].number_input("Peso LAI:", value=st.session_state[SESSION_KEY_ET_COEFFS]["lai"], step=0.01, format="%.2f", key="coeff_lai_cfg_v6")
        cols_coeffs_et3 = st.columns(2); st.session_state[SESSION_KEY_ET_COEFFS]["temperatura"] = cols_coeffs_et3[0].number_input("Peso Temperatura:", value=st.session_state[SESSION_KEY_ET_COEFFS]["temperatura"], step=0.01, format="%.2f", key="coeff_temp_cfg_v6"); st.session_state[SESSION_KEY_ET_COEFFS]["umidade"] = cols_coeffs_et3[1].number_input("Peso Umidade:", value=st.session_state[SESSION_KEY_ET_COEFFS]["umidade"], step=0.01, format="%.3f", key="coeff_umid_cfg_v6")
        cols_coeffs_et4 = st.columns(2); st.session_state[SESSION_KEY_ET_COEFFS]["radiacao"] = cols_coeffs_et4[0].number_input("Peso Radiação:", value=st.session_state[SESSION_KEY_ET_COEFFS]["radiacao"], step=0.001, format="%.3f", key="coeff_rad_cfg_v6"); st.session_state[SESSION_KEY_ET_COEFFS]["vento"] = cols_coeffs_et4[1].number_input("Peso Vento:", value=st.session_state[SESSION_KEY_ET_COEFFS]["vento"], step=0.001, format="%.3f", key="coeff_vento_cfg_v6")
        st.session_state[SESSION_KEY_ET_COEFFS]["fator_escala"] = st.number_input("Fator de Escala ET Geral:", value=st.session_state[SESSION_KEY_ET_COEFFS]["fator_escala"], step=0.1, format="%.1f", key="coeff_escala_et_cfg_v6")
        st.markdown("**Modelo de Absorção de Carbono Simplificado:**"); st.caption("Coeficientes altamente hipotéticos...")
        st.session_state[SESSION_KEY_CARBON_COEFFS]["k_aft_carbon"] = st.number_input("Coef. Carbono via AFT (kg C/m²/dia):", value=st.session_state[SESSION_KEY_CARBON_COEFFS]["k_aft_carbon"], step=0.0001, format="%.4f", key="coeff_k_aft_c_cfg_v6", help="Taxa média de fixação de Carbono por área foliar.")
        st.session_state[SESSION_KEY_CARBON_COEFFS]["c_et_wue_carbon"] = st.number_input("Coef. Carbono via ET/WUE (kg C/L ET):", value=st.session_state[SESSION_KEY_CARBON_COEFFS]["c_et_wue_carbon"], step=0.0001, format="%.4f", key="coeff_c_et_wue_c_cfg_v6", help="Proxy para Eficiência no Uso da Água em termos de Carbono ganho por água perdida.")

    st.markdown("---")
    if st.button("🚀 Executar Simulação e Análise", type="primary", key="run_simulation_main_button_v6", use_container_width=True):
        st.session_state[SESSION_KEY_RESULTADOS_MODELO] = []
        # Validação das entradas climáticas (Seção II.A)
        try:
            temp_val_clim = float(temp_c_str_input)
            umid_val_clim = float(umid_perc_str_input)
            rad_val_clim = float(rad_mj_m2_dia_str_input)
            vento_val_clim = float(vento_m_s_str_input)
            if not (0 < umid_val_clim <= 100):
                st.error("Umidade Relativa Climática deve estar entre 0 (exclusive) e 100%.")
                st.stop() # Interrompe a execução se dados climáticos forem inválidos
        except ValueError:
            st.error("Erro: Verifique se todas as variáveis climáticas são números válidos.")
            st.stop()

        for i, especime_data_dict in enumerate(st.session_state[SESSION_KEY_ESPECIMES_DATA]):
            if not especime_data_dict or not especime_data_dict.get("id_user"):
                st.warning(f"Dados para o Espécime {i+1} incompletos. Pulando este espécime.")
                continue
            try:
                # Validação e conversão de entradas biométricas (Seção II.A)
                altura_m_val = float(especime_data_dict["altura_m_str"])
                diametro_cm_val = float(especime_data_dict["diametro_cm_str"])
                area_copa_m2_val = float(especime_data_dict["area_copa_m2_str"])
                galhos_p_val = int(especime_data_dict["galhos_principais"])
                num_folhas_gh_est_val = int(especime_data_dict["num_folhas_por_galho_estimado"])

                # Validação de intervalo (Seção II.A)
                if not (0.1 <= altura_m_val <= 200 and \
                          0.1 <= diametro_cm_val <= 500 and \
                          0.1 <= area_copa_m2_val <= 1000 and \
                          galhos_p_val > 0 and \
                          num_folhas_gh_est_val > 0):
                    st.warning(f"Espécime {especime_data_dict['id_user']}: Valores biométricos fora do intervalo plausível ou não positivos. Verifique.")
                    # Pode-se optar por pular este espécime ou usar valores padrão
                    # Para este exemplo, pularemos se a validação falhar criticamente.
                    # Ou, você pode tentar prosseguir com AFT/LAI = 0 se fizer sentido.
                    aft_m2_calculada = 0.0
                    lai_calculado = 0.0
                else:
                    aft_m2_calculada = calculate_area_foliar_total(especime_data_dict["folhas_data_list"], galhos_p_val, num_folhas_gh_est_val)
                    lai_calculado = calculate_lai(aft_m2_calculada, especime_data_dict["area_copa_m2_str"])

                et_litros_dia_calculada = predict_evapotranspiration_phd(
                    altura_m_val, diametro_cm_val, area_copa_m2_val, lai_calculado,
                    temp_val_clim, umid_val_clim, rad_val_clim, vento_val_clim,
                    st.session_state[SESSION_KEY_ET_COEFFS]
                )
                carbono_aft_kg_dia_calc, carbono_et_wue_kg_dia_calc = estimate_carbon_absorption_phd(
                    aft_m2_calculada, et_litros_dia_calculada, st.session_state[SESSION_KEY_CARBON_COEFFS]
                )
                st.session_state[SESSION_KEY_RESULTADOS_MODELO].append({
                    "ID Espécime": especime_data_dict["id_user"], "Altura (m)": altura_m_val,
                    "Diâmetro (cm)": diametro_cm_val, "Área Copa (m²)": area_copa_m2_val,
                    "AFT Estimada (m²)": round(aft_m2_calculada, 3), "LAI Estimado": lai_calculado,
                    "ET Modelo (L/dia)": et_litros_dia_calculada,
                    "Carbono (AFT) (kgC/dia)": carbono_aft_kg_dia_calc,
                    "Carbono (ET/WUE) (kgC/dia)": carbono_et_wue_kg_dia_calc
                })
            except KeyError as ke: st.error(f"Erro de chave (campo faltando) ao processar Espécime {especime_data_dict.get('id_user', f'Índice {i}')}: '{ke}'."); continue
            except ValueError as ve: st.error(f"Erro de valor (tipo de dado incorreto) ao processar Espécime {especime_data_dict.get('id_user', f'Índice {i}')}: {ve}."); continue
            except Exception as e: st.error(f"Erro inesperado processando Espécime {especime_data_dict.get('id_user', f'Índice {i}')}: {e}"); continue
        
        if st.session_state[SESSION_KEY_RESULTADOS_MODELO]: st.success(f"Simulação concluída para {len(st.session_state[SESSION_KEY_RESULTADOS_MODELO])} espécime(s). Veja os resultados à direita.")
        else: st.warning("Nenhum espécime pôde ser processado. Verifique as entradas e mensagens de erro.")

with right_column:
    st.header("📊 Resultados da Simulação e Análises")
    if not st.session_state[SESSION_KEY_RESULTADOS_MODELO]:
        st.info("Aguardando execução da simulação. Configure as entradas à esquerda e clique em 'Executar Simulação'.")
    else:
        df_resultados_finais = pd.DataFrame(st.session_state[SESSION_KEY_RESULTADOS_MODELO])
        st.subheader("Resumo dos Resultados do Modelo:")
        st.dataframe(df_resultados_finais.style.format("{:.2f}", subset=pd.IndexSlice[:, ['Altura (m)', 'Diâmetro (cm)', 'Área Copa (m²)', 'AFT Estimada (m²)', 'LAI Estimado', 'ET Modelo (L/dia)']]).format("{:.4f}", subset=pd.IndexSlice[:, ['Carbono (AFT) (kgC/dia)', 'Carbono (ET/WUE) (kgC/dia)']]))
        st.subheader("Visualizações Gráficas dos Resultados do Modelo:")
        # ... (Gráficos como antes, usando df_resultados_finais) ...
        if not df_resultados_finais.empty:
            col_g1, col_g2 = st.columns(2); # Gráficos como antes
            with col_g1:
                if "ET Modelo (L/dia)" in df_resultados_finais.columns:
                    fig_et, ax_et = plt.subplots(); ax_et.bar(df_resultados_finais["ID Espécime"], df_resultados_finais["ET Modelo (L/dia)"], color='skyblue'); ax_et.set_xlabel("ID do Espécime"); ax_et.set_ylabel("ET Estimada (L/dia)"); ax_et.set_title("Evapotranspiração Estimada por Espécime"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); st.pyplot(fig_et)
                    fig_hist_et, ax_hist_et = plt.subplots(); ax_hist_et.hist(df_resultados_finais["ET Modelo (L/dia)"].dropna(), bins=10, color='lightblue', edgecolor='black'); ax_hist_et.set_xlabel("ET Estimada (L/dia)"); ax_hist_et.set_ylabel("Frequência"); ax_hist_et.set_title("Distribuição da ET Estimada"); st.pyplot(fig_hist_et)
            with col_g2:
                if "Carbono (AFT) (kgC/dia)" in df_resultados_finais.columns:
                    fig_c, ax_c = plt.subplots(); ax_c.bar(df_resultados_finais["ID Espécime"], df_resultados_finais["Carbono (AFT) (kgC/dia)"], color='lightgreen'); ax_c.set_xlabel("ID do Espécime"); ax_c.set_ylabel("Carbono Estimado (kgC/dia - via AFT)"); ax_c.set_title("Absorção de Carbono Estimada (via AFT)"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); st.pyplot(fig_c)
                    fig_hist_c, ax_hist_c = plt.subplots(); ax_hist_c.hist(df_resultados_finais["Carbono (AFT) (kgC/dia)"].dropna(), bins=10, color='green', edgecolor='black'); ax_hist_c.set_xlabel("Carbono Estimado (kgC/dia - via AFT)"); ax_hist_c.set_ylabel("Frequência"); ax_hist_c.set_title("Distribuição do Carbono Estimado (via AFT)"); st.pyplot(fig_hist_c)
            if "LAI Estimado" in df_resultados_finais.columns and not df_resultados_finais["LAI Estimado"].empty:
                st.markdown("#### Boxplot do LAI Estimado entre Espécimes"); fig_box_lai, ax_box_lai = plt.subplots(); ax_box_lai.boxplot(df_resultados_finais["LAI Estimado"].dropna(), patch_artist=True, vert=False); ax_box_lai.set_yticklabels(['LAI']); ax_box_lai.set_xlabel('LAI Estimado'); ax_box_lai.set_title('Distribuição do LAI Estimado'); st.pyplot(fig_box_lai)

        st.markdown("---")
        st.subheader("🔬 Contraprova Experimental e Análise Estatística Comparativa")
        
        # --- GERENCIAMENTO DE ESTADO REVISADO PARA O WIDGET DE NÚMERO DE MEDIÇÕES ---
        # 1. Obter valor do estado, com fallback seguro e correção se inválido.
        # Este valor é usado para definir o 'value' do widget.
        valor_para_widget_num_medicoes = st.session_state.get(SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR, DEFAULT_NUM_MEDICOES_EXP)
        if not isinstance(valor_para_widget_num_medicoes, int) or valor_para_widget_num_medicoes < 1:
            valor_para_widget_num_medicoes = DEFAULT_NUM_MEDICOES_EXP
            st.session_state[SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR] = valor_para_widget_num_medicoes # Corrige o estado se estava ruim

        # 2. Renderizar o widget
        num_medicoes_exp_atual_saida_widget = st.number_input(
            "Número de Medições Experimentais por Espécime:",
            min_value=1,
            value=valor_para_widget_num_medicoes, # Usa o valor seguro
            step=1,
            key="num_med_exp_comp_input_widget_v6" # Chave única
        )
        # 3. Atualizar a variável de estado que alimenta este widget para a PRÓXIMA execução.
        # Isso garante que o estado reflita a última entrada válida do usuário.
        st.session_state[SESSION_KEY_NUM_MEDICOES_EXP_ANTERIOR] = num_medicoes_exp_atual_saida_widget
        # --- FIM DA CORREÇÃO DO GERENCIAMENTO DE ESTADO ---

        tempo_coleta_exp_horas_str_input = st.text_input(
            "Tempo de Coleta para Cada Medição Experimental (horas):",
            st.session_state.get(SESSION_KEY_TEMPO_COLETA_ANT, "24"),
            key="tempo_coleta_exp_h_comp_input_v6"
        )
        st.session_state[SESSION_KEY_TEMPO_COLETA_ANT] = tempo_coleta_exp_horas_str_input

        # Lógica de atualização/reset do st.session_state.contraprovas_data
        # Baseado no número de espécimes *atuais* (de df_resultados_finais) e no número de medições *atuais*
        if not df_resultados_finais.empty:
            num_especimes_modelo_render = df_resultados_finais.shape[0]
            # Condição para resetar: se o número de espécimes mudou OU o número de medições mudou
            if st.session_state[SESSION_KEY_NUM_ESPECIMES_COMP_ANTERIOR] != num_especimes_modelo_render or \
               (df_resultados_finais["ID Espécime"].iloc[0] in st.session_state[SESSION_KEY_CONTRAPROVAS_DATA] and \
                len(st.session_state[SESSION_KEY_CONTRAPROVAS_DATA][df_resultados_finais["ID Espécime"].iloc[0]]) != num_medicoes_exp_atual_saida_widget):
                
                st.session_state[SESSION_KEY_CONTRAPROVAS_DATA] = {} 
                for _, row_modelo_init_exp in df_resultados_finais.iterrows():
                    id_especime_init_exp = row_modelo_init_exp["ID Espécime"]
                    st.session_state[SESSION_KEY_CONTRAPROVAS_DATA][id_especime_init_exp] = ["0"] * num_medicoes_exp_atual_saida_widget # Usa valor atual do widget
                st.session_state[SESSION_KEY_NUM_ESPECIMES_COMP_ANTERIOR] = num_especimes_modelo_render
        
        # Renderiza os campos de input para dados experimentais
        for _, row_modelo_render_exp in df_resultados_finais.iterrows():
            id_especime_render_exp = row_modelo_render_exp["ID Espécime"]
            if id_especime_render_exp not in st.session_state[SESSION_KEY_CONTRAPROVAS_DATA] or \
               len(st.session_state[SESSION_KEY_CONTRAPROVAS_DATA].get(id_especime_render_exp, [])) != num_medicoes_exp_atual_saida_widget:
                st.session_state[SESSION_KEY_CONTRAPROVAS_DATA][id_especime_render_exp] = ["0"] * num_medicoes_exp_atual_saida_widget
            
            with st.container(border=True):
                st.markdown(f"**Valores Experimentais para Espécime: {id_especime_render_exp}**")
                medicoes_atuais_str_input_temp_exp = [] 
                cols_med_exp_display_exp = st.columns(num_medicoes_exp_atual_saida_widget)
                for k_render_exp in range(num_medicoes_exp_atual_saida_widget):
                    with cols_med_exp_display_exp[k_render_exp]:
                        default_val_med_render_exp = st.session_state[SESSION_KEY_CONTRAPROVAS_DATA][id_especime_render_exp][k_render_exp]
                        val_exp_input_render_exp = st.text_input(
                            f"Medição {k_render_exp+1} (mL):", value=default_val_med_render_exp,
                            key=f"med_exp_val_v6_{id_especime_render_exp}_{k_render_exp}"
                        )
                        medicoes_atuais_str_input_temp_exp.append(val_exp_input_render_exp)
                st.session_state[SESSION_KEY_CONTRAPROVAS_DATA][id_especime_render_exp] = medicoes_atuais_str_input_temp_exp

        tipo_teste_estatistico_selecionado = st.selectbox(
            "Escolha o Teste Estatístico para Comparação (Modelo vs. Experimental):",
            ("Teste t de Student (1 amostra)", "Teste de Wilcoxon (Signed-Rank)", "Diferença Absoluta e Percentual"),
            key="tipo_teste_stat_phd_select_comp_widget_v6"
        )

        if st.button("🔄 Comparar Modelo com Dados Experimentais", key="run_comparison_main_button_action_v6", use_container_width=True):
            # ... (lógica do botão de comparação como na versão anterior, com tratamento de erro para conversão) ...
            # Esta parte já estava robusta em relação à conversão individual dos valores
            # e tratamento de erros. Apenas garantir que usa as variáveis corretas.
            if not st.session_state.get(SESSION_KEY_CONTRAPROVAS_DATA): st.warning("Por favor, insira os dados experimentais."); st.stop()
            try: tempo_coleta_h_comp = float(tempo_coleta_exp_horas_str_input); assert tempo_coleta_h_comp > 0
            except (ValueError, AssertionError): st.error("Tempo de coleta experimental inválido ou não positivo."); st.stop()

            st.markdown("### Resultados da Comparação Estatística:")
            all_exp_means_list_comp, all_model_preds_list_comp, valid_comp_count = [], [], 0
            for _, row_modelo_comp_final in df_resultados_finais.iterrows():
                id_especime_comp_final = row_modelo_comp_final["ID Espécime"]; et_modelo_val_comp_final = row_modelo_comp_final["ET Modelo (L/dia)"]
                if id_especime_comp_final not in st.session_state.get(SESSION_KEY_CONTRAPROVAS_DATA, {}): st.warning(f"Dados experimentais não configurados para {id_especime_comp_final}. Pulando."); continue
                medicoes_exp_str_list_comp_final = st.session_state[SESSION_KEY_CONTRAPROVAS_DATA][id_especime_comp_final]
                medicoes_exp_ml_float_validas_final = []; valores_exp_invalidos_flag_final = False
                for idx_med_final, med_str_val_final in enumerate(medicoes_exp_str_list_comp_final):
                    try:
                        med_str_cleaned_final = med_str_val_final.strip()
                        if not med_str_cleaned_final: st.warning(f"Medição {idx_med_final+1} (Espécime {id_especime_comp_final}) vazia. Ignorando."); continue
                        med_float_final = float(med_str_cleaned_final); medicoes_exp_ml_float_validas_final.append(med_float_final)
                    except ValueError: st.error(f"Valor '{med_str_val_final}' (Medição {idx_med_final+1}, Espécime {id_especime_comp_final}) inválido."); valores_exp_invalidos_flag_final = True; break
                if valores_exp_invalidos_flag_final: st.warning(f"Análise para {id_especime_comp_final} interrompida."); continue
                if not medicoes_exp_ml_float_validas_final: st.warning(f"Nenhum dado experimental válido para {id_especime_comp_final}. Pulando."); continue
                medicoes_exp_L_dia_final = [(m_final/1000.0)/(tempo_coleta_h_comp/24.0) for m_final in medicoes_exp_ml_float_validas_final]
                if not medicoes_exp_L_dia_final: st.warning(f"Nenhuma medição processável para {id_especime_comp_final}."); continue
                media_exp_L_dia_final = np.mean(medicoes_exp_L_dia_final)
                all_exp_means_list_comp.append(media_exp_L_dia_final); all_model_preds_list_comp.append(et_modelo_val_comp_final); valid_comp_count += 1
                st.markdown(f"#### Análise para Espécime: {id_especime_comp_final}"); st.write(f"- ET Modelo: {et_modelo_val_comp_final:.2f} L/dia"); st.write(f"- ET Média Experimental: {media_exp_L_dia_final:.2f} L/dia ({len(medicoes_exp_L_dia_final)} medições: {[f'{x:.2f}' for x in medicoes_exp_L_dia_final]})")
                p_valor_teste_atual_final = None
                if tipo_teste_estatistico_selecionado == "Teste t de Student (1 amostra)":
                    if len(medicoes_exp_L_dia_final) < 2 or len(set(medicoes_exp_L_dia_final)) < 2 : st.warning("Teste t: dados insuficientes/sem variabilidade.")
                    else: stat_t_final, p_valor_teste_atual_final = stats.ttest_1samp(medicoes_exp_L_dia_final, et_modelo_val_comp_final); st.write(f"  - Teste t: t={stat_t_final:.3f}, p-valor={p_valor_teste_atual_final:.4f}")
                elif tipo_teste_estatistico_selecionado == "Teste de Wilcoxon (Signed-Rank)":
                    if len(medicoes_exp_L_dia_final) < 1: st.warning("Wilcoxon: dados insuficientes.")
                    else:
                        diffs_final = np.array(medicoes_exp_L_dia_final) - et_modelo_val_comp_final
                        if np.all(diffs_final == 0) and len(diffs_final)>0 : st.warning("Wilcoxon: todas as diferenças são zero.")
                        elif len(diffs_final) == 0: st.warning("Wilcoxon: sem dados para diferenças.")
                        else: 
                            try:
                                if len(diffs_final[diffs_final != 0]) == 0 and len(diffs_final) > 0: st.warning("Wilcoxon: todas as diferenças são zero (após remover zeros).")
                                elif len(diffs_final) > 0: stat_w_final, p_valor_teste_atual_final = stats.wilcoxon(diffs_final, alternative='two-sided', zero_method='wilcox'); st.write(f"  - Wilcoxon: W={stat_w_final:.3f}, p-valor={p_valor_teste_atual_final:.4f}")
                                else: st.warning("Wilcoxon: dados insuficientes após processar diferenças.")
                            except ValueError as e_wilcoxon_final: st.warning(f"  - Wilcoxon não calculado: {e_wilcoxon_final}.")
                diferenca_abs_final = abs(media_exp_L_dia_final - et_modelo_val_comp_final); diferenca_perc_final = (diferenca_abs_final / media_exp_L_dia_final) * 100 if media_exp_L_dia_final != 0 else float('inf')
                st.write(f"  - Diferença Absoluta: {diferenca_abs_final:.2f} L/dia"); st.write(f"  - Diferença Percentual: {diferenca_perc_final:.2f}%")
                if p_valor_teste_atual_final is not None:
                    alpha_final = 0.05
                    if p_valor_teste_atual_final < alpha_final: st.error(f"  - Conclusão: Diferença estatisticamente significativa (p < {alpha_final}).")
                    else: st.success(f"  - Conclusão: Diferença não estatisticamente significativa (p ≥ {alpha_final}).")
            if valid_comp_count > 1:
                st.markdown("--- \n ### Análise Global de Desempenho do Modelo (Comparativo)")
                exp_means_np_global_final = np.array(all_exp_means_list_comp); model_preds_np_global_final = np.array(all_model_preds_list_comp)
                rmse_global_final = np.sqrt(mean_squared_error(exp_means_np_global_final, model_preds_np_global_final)); mae_global_final = mean_absolute_error(exp_means_np_global_final, model_preds_np_global_final)
                try: r2_global_final = r2_score(exp_means_np_global_final, model_preds_np_global_final)
                except ValueError: r2_global_final = np.nan
                st.write(f"**Métricas Globais:** RMSE: {rmse_global_final:.3f} L/dia, MAE: {mae_global_final:.3f} L/dia, R²: {r2_global_final:.3f}" if not np.isnan(r2_global_final) else f"RMSE: {rmse_global_final:.3f}, MAE: {mae_global_final:.3f}, R²: N/A")
                fig_scatter_global_final, ax_scatter_global_final = plt.subplots(); ax_scatter_global_final.scatter(model_preds_np_global_final, exp_means_np_global_final, alpha=0.7, edgecolors='k'); 
                min_val_plot_final = min(min(model_preds_np_global_final), min(exp_means_np_global_final)) if valid_comp_count > 0 and len(model_preds_np_global_final)>0 and len(exp_means_np_global_final)>0 else 0
                max_val_plot_final = max(max(model_preds_np_global_final), max(exp_means_np_global_final)) if valid_comp_count > 0 and len(model_preds_np_global_final)>0 and len(exp_means_np_global_final)>0 else 1
                ax_scatter_global_final.plot([min_val_plot_final, max_val_plot_final], [min_val_plot_final, max_val_plot_final], 'r--', label="Linha 1:1 (Ideal)"); ax_scatter_global_final.set_xlabel("ET Modelo (L/dia)"); ax_scatter_global_final.set_ylabel("ET Média Experimental (L/dia)"); ax_scatter_global_final.set_title("Comparação Global: Modelo vs. Experimental"); ax_scatter_global_final.legend(); ax_scatter_global_final.grid(True); st.pyplot(fig_scatter_global_final)
            elif not df_resultados_finais.empty : st.info("Análise global comparativa requer dados experimentais válidos de pelo menos dois espécimes.")


# ---------------------------------------------------------------
# Seções Explicativas (LaTeX corrigido com r"...")
# ---------------------------------------------------------------
st.sidebar.title("Navegação e Informações")
st.sidebar.info(f"""
**Plataforma de Simulação e Análise para Pesquisa de Doutorado**
Foco: Evapotranspiração e Dinâmica de Carbono em Ecossistemas Semiáridos (Crateús, Ceará).
Versão: 1.0.6 (Correção Erro `StreamlitValueBelowMinError` e LaTeX)
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
        1.  **Modelo de Penman-Monteith (FAO-56 PM):** Padrão ouro, combina balanço de energia com termos de transporte aerodinâmico e resistência superficial (condutância estomática, $g_s$). Requer calibração de $g_s$ para espécies locais da Caatinga, considerando fatores como déficit de pressão de vapor (VPD), radiação, umidade do solo.""")
    st.latex(r"ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T+273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}")
    st.markdown(r"""Para ET real ($ET_c$), $ET_c = K_c ET_0$ ou modelagem direta de $g_s$.
        2.  **Modelos de Machine Learning:** Random Forest, Gradient Boosting, Redes Neurais, treinados com dados de ET medidos (e.g., fluxo de seiva, lisímetros, covariância de vórtices) e preditores ambientais/biométricos. Exigem grandes conjuntos de dados para treinamento e validação.
    """)

    st.markdown(r"### 🍂 Área Foliar Total (AFT) e Índice de Área Foliar (LAI)")
    st.markdown(r"""
    - **AFT:** Área total de superfície foliar fotossinteticamente ativa. Crucial para trocas gasosas.
    - **LAI:** AFT por unidade de área de solo ($LAI = AFT/A_{copa}$). Adimensional, indica a densidade do dossel.
    - **Estimativa (Usada Aqui):** Baseada na área média de folhas de amostra e estimativas do número de folhas. Altamente simplificado.
    - **Abordagem de Doutorado:** (conteúdo mantido como antes)
    """)
    st.latex(r'''\text{AFT (m}^2\text{)} = \sum_{\text{folhas}} (\text{área da folha em m}^2\text{)}''') 
    st.latex(r'''\text{LAI} = \frac{\text{Área Foliar Total (m}^2\text{)}}{\text{Área da Copa Projetada no Solo (m}^2\text{)}}''') 

    st.markdown(r"### 🌳 Estimativa de Absorção/Captura de Carbono")
    st.markdown(r"""A fixação de carbono via fotossíntese é o principal mecanismo de entrada de C nos ecossistemas terrestres.
    - **Estimativa Simplificada (Usada Aqui):** Coeficientes fixos multiplicados por AFT ou ET. Meramente ilustrativo.
    - **Abordagem de Doutorado:**
        1.  **Modelos Ecofisiológicos de Fotossíntese:** Modelo de Farquhar, von Caemmerer & Berry (FvCB) para fotossíntese da folha...""")
    st.latex(r"A = \min(A_c, A_j, A_p) - R_d") # CORREÇÃO ESPECÍFICA DA SYNTAXWARNING
    st.markdown(r"""
        2.  **Modelos Baseados em Eficiência no Uso da Luz (LUE):** $NPP = APAR \times LUE_{eco}$ ...
        3.  **Balanço de Carbono do Ecossistema:** ... $NPP \approx GPP - R_a$ (respiração autotrófica).
        4.  **Alocação de Biomassa:** Entender como o carbono fixado é distribuído entre folhas, caules, raízes.
        5.  **Eficiência no Uso da Água (WUE):** $WUE = A / E$ (Fotossíntese / Transpiração). Crucial em ambientes semiáridos. Varia com CO₂, VPD, espécie.
    """)

    st.markdown(r"### 🔬 Análise Estatística e Validação de Modelos (PhD)")
    st.markdown(r"""(conteúdo mantido como antes)""")

    st.markdown(r"### ⚛️ q-Estatística (Estatística de Tsallis) em Pesquisas Ecológicas")
    st.markdown(r"""A q-estatística generaliza a estatística de Boltzmann-Gibbs...
    -   A **q-Exponencial** pode descrever processos... Sua forma funcional pode ser:""")
    st.latex(r"f(x; q, \beta) = N \exp_q(-\beta x) = N [1 - (1-q) \beta x]_+^{1/(1-q)}")


with st.expander("🌳 Estimativa Simplificada de Absorção/Captura de Carbono (Conceitual Detalhado)", expanded=False):
    st.latex(r'''\text{Carbono}_{\text{AFT}} \text{(kg C/dia)} \approx k_{\text{AFT}} \times \text{AFT (m}^2\text{)}''')
    st.latex(r'''\text{Carbono}_{\text{ET/WUE}} \text{(kg C/dia)} \approx k_{\text{ET/WUE}} \times \text{ET (litros/dia)}''')
    st.markdown(r"""onde \( k_{\text{AFT}} \) e \( k_{\text{ET/WUE}} \) ... (conteúdo mantido como antes)""")

st.markdown("---")
st.caption(f"Plataforma de Simulação Avançada - Versão para Discussão em Banca de Doutorado. {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
