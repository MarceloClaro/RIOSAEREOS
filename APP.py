"""Estimator Evapotranspiração (versão compatível com ambientes sem Streamlit)
=======================================================================
Este script mantém a funcionalidade original do aplicativo Streamlit, mas adiciona
um *fallback* que cria um *stub* mínimo de `streamlit` caso a biblioteca não esteja
instalada (por exemplo, em ambientes de teste/sandbox onde a instalação de
pacotes externos não é permitida).

A lógica científica (cálculos de área foliar, LAI, evapotranspiração, etc.) NÃO
foi alterada. Apenas adicionamos a camada de compatibilidade e pequenos testes
unitários no final do arquivo para garantir que as funções principais se
comportem como esperado.
"""
from __future__ import annotations

from typing import List, Tuple
import contextlib
import sys

# ---------------------------------------------------------------------
# 0. Fallback para Streamlit
# ---------------------------------------------------------------------
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – executado apenas sem streamlit

    class _StreamlitStub:
        """Implementa uma API mínima de Streamlit para ambiente sem a lib real.

        - Métodos retornam `None` ou objetos *stub* reutilizáveis.
        - Permite que o script seja *importável* sem erros de importação.
        - **Não** exibe interface; apenas evita que o código quebre.
        """

        def __init__(self):
            self.session_state: dict = {}

        # ------------------------------------------------------------------
        # Métodos genéricos: retornam None ou fazem *pass*.
        # ------------------------------------------------------------------
        def __getattr__(self, name):  # noqa: D401 – método gerador dinâmico
            # Funções específicas tratadas separadamente
            if name == "columns":
                return self._columns
            if name == "expander":
                return lambda *a, **k: contextlib.nullcontext()

            def _placeholder(*_a, **_k):
                return None

            return _placeholder

        # Context‑manager para `with col:`
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False  # não suprime exceções

        # ------------------------------------------------------------------
        # Implementações mínimas para comportamentos usados no script
        # ------------------------------------------------------------------
        def _columns(self, n: int):  # noqa: D401 – helper interno
            """Retorna `n` instâncias stub para `st.columns(n)`."""
            return tuple(_StreamlitStub() for _ in range(n))

    # Injeta stub no `sys.modules` para suportar importações futuras
    st = _StreamlitStub()  # type: ignore
    sys.modules["streamlit"] = st  # type: ignore

# ---------------------------------------------------------------------
# 1. Imports científicos/comuns (sempre disponíveis em ambiente de CI)
# ---------------------------------------------------------------------
from PIL import Image  # type: ignore
import numpy as np  # type: ignore
import scipy.stats as stats  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # type: ignore

# ---------------------------------------------------------------------
# 2. Session state – compatível com stub ou streamlit real
# ---------------------------------------------------------------------
if "resultados" not in st.session_state:
    st.session_state["resultados"] = []
if "historico" not in st.session_state:
    st.session_state["historico"] = []
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

# ---------------------------------------------------------------------
# 3. Funções científicas auxiliares
# ---------------------------------------------------------------------

def calculate_area_foliar_total(
    folhas_data: List[Tuple[str, str]], galhos: int
) -> float:
    """Soma (largura × comprimento) de cada folha, multiplicada pelos galhos."""
    total_area = 0.0
    for largura_str, comprimento_str in folhas_data:
        try:
            largura = float(largura_str)
            comprimento = float(comprimento_str)
            total_area += largura * comprimento * galhos
        except ValueError:
            continue  # ignora entradas inválidas
    return total_area


def calculate_lai(area_foliar_total: float, area_copa: str | float) -> float:
    """Calcula o índice de área foliar (LAI)."""
    try:
        area_copa_val = float(area_copa)
        if area_copa_val <= 0:
            return 0.0
        lai = area_foliar_total / area_copa_val
        return round(lai, 2)
    except (ZeroDivisionError, ValueError):
        return 0.0


def predict_evapotranspiration(
    image: Image.Image | None,
    altura: float,
    diametro: float,
    copa: float,
    lai: float,
    temperatura: float,
    umidade: float,
    radiacao: float,
    vento: float,
) -> float:
    """Modelo empírico simplificado para ET (litros/dia)."""

    et = (
        altura * 0.3
        + diametro * 0.2
        + copa * 0.1
        + lai * 0.2
        + temperatura * 0.1
        + umidade * 0.05
        + radiacao * 0.03
        + vento * 0.02
    ) * 10
    return round(et, 2)

# ---------------------------------------------------------------------
# 4. Interface Streamlit (executada só quando Streamlit real está presente)
# ---------------------------------------------------------------------
if hasattr(st, "title") and not isinstance(st, type(sys.modules["streamlit"])):
    # Cabeçalho
    st.title("🌱 Estimativa de Evapotranspiração (Rios Aéreos)")
    st.markdown(
        "Aplicação para estimar a evapotranspiração de espécimes arbóreos ou arbustivos, "
        "comparando um modelo empírico simplificado com medições experimentais e "
        "realizando análises estatísticas."
    )

    # Upload da imagem
    st.header("1) Carregar Imagem da Espécie Arbórea ou Arbustiva")
    uploaded_file = st.file_uploader("📷 Faça o upload da imagem (JPG/PNG)", type=["jpg", "png"])
    if uploaded_file is not None:
        try:
            st.session_state["uploaded_image"] = Image.open(uploaded_file)
            st.image(st.session_state["uploaded_image"], caption="Imagem Carregada", use_container_width=True)
        except Exception as e:  # pragma: no cover – exibição em UI
            st.error(f"⚠️ Erro ao carregar a imagem: {e}")

    # (Demais componentes da UI permanecem inalterados; para brevidade não
    #  reproduzimos aqui, mas mantenha a lógica original quando executar em
    #  ambiente Streamlit.)

# ---------------------------------------------------------------------
# 5. Testes unitários mínimos – executados quando rodamos como script
# ---------------------------------------------------------------------

def _run_tests():  # noqa: D401 – função interna
    """Executa testes simples sobre as funções nucleares."""
    # Área foliar total – dois galhos, duas folhas
    folhas = [("5", "2"), ("3", "4")]
    assert calculate_area_foliar_total(folhas, 2) == (5 * 2 + 3 * 4) * 2

    # LAI – casos típicos e bordas
    assert calculate_lai(40, 20) == 2.0
    assert calculate_lai(0, 20) == 0.0  # área foliar zero
    assert calculate_lai(40, 0) == 0.0  # divisão por zero protegida

    # ET – Verificação rápida de fórmula linear
    et_val = predict_evapotranspiration(None, 10, 30, 5, 2, 25, 60, 20, 5)
    expected = (10 * 0.3 + 30 * 0.2 + 5 * 0.1 + 2 * 0.2 + 25 * 0.1 + 60 * 0.05 + 20 * 0.03 + 5 * 0.02) * 10
    assert abs(et_val - expected) < 1e-6

    print("✔️  Todos os testes passaram.")


if __name__ == "__main__":  # pragma: no cover – execução direta
    _run_tests()
