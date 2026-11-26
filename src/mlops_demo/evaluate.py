"""Métricas de avaliação. A classe de interesse é o PULSAR (rótulo 1)."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from mlops_demo.config import CLASSE_POSITIVA
from mlops_demo.features import separar_xy


def calcular_metricas(y_real, y_previsto) -> dict:
    """Recall e precisão da classe pulsar, mais a acurácia como referência.

    A acurácia sozinha engana no HTRU2: só ~9% dos candidatos são pulsares, então um modelo que
    chuta tudo como ruído já acerta ~91%. Por isso o contrato olha recall e precisão do pulsar.
    """
    return {
        "recall_pulsar": float(
            recall_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA, zero_division=0)
        ),
        "precisao_pulsar": float(
            precision_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA, zero_division=0)
        ),
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "n_amostras": int(len(y_real)),
    }


def avaliar(modelo, df: pd.DataFrame) -> dict:
    """Avalia um modelo treinado em um DataFrame com as colunas do projeto."""
    X, y = separar_xy(df)
    return calcular_metricas(y, modelo.predict(X))
