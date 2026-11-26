"""Testes de código: funções e pipeline se comportam como o contrato diz (HTRU2, classe pulsar)."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mlops_demo import config
from mlops_demo.evaluate import calcular_metricas
from mlops_demo.features import construir_pipeline, separar_xy
from mlops_demo.gate import decidir

pytestmark = pytest.mark.codigo


def test_separar_xy_devolve_8_atributos_e_alvo(base_treino):
    X, y = separar_xy(base_treino)
    assert X.shape == (len(base_treino), len(config.FEATURES))
    assert list(X.columns) == config.FEATURES
    assert len(y) == len(base_treino)


def test_separar_xy_impoe_a_ordem_das_colunas(base_treino):
    embaralhado = base_treino[list(reversed(base_treino.columns))]
    X, _ = separar_xy(embaralhado)
    assert list(X.columns) == config.FEATURES


def test_separar_xy_falha_quando_falta_coluna(base_treino):
    sem_coluna = base_treino.drop(columns=["perfil_curtose"])
    with pytest.raises(ValueError, match="Colunas ausentes"):
        separar_xy(sem_coluna)


def test_separar_xy_nao_altera_o_dataframe_original(base_treino):
    antes = base_treino.copy()
    separar_xy(base_treino)
    pd.testing.assert_frame_equal(base_treino, antes)


def test_pipeline_tem_padronizacao_e_classificador(params):
    pipeline = construir_pipeline(params)
    assert isinstance(pipeline, Pipeline)
    assert [nome for nome, _ in pipeline.steps] == ["padronizar", "classificador"]


def test_metricas_usam_o_pulsar_como_classe_positiva():
    # modelo que chuta tudo como pulsar (1): recall 1.0 e precisão = proporção real de pulsares
    y_real = [0, 0, 0, 1]
    metricas = calcular_metricas(y_real, [1, 1, 1, 1])
    assert metricas["recall_pulsar"] == 1.0
    assert metricas["precisao_pulsar"] == pytest.approx(0.25)


def test_metricas_mostram_que_acuracia_engana_no_desbalanceamento():
    # chutar tudo como ruído acerta 90% e não encontra nenhum pulsar
    y_real = [0] * 9 + [1]
    metricas = calcular_metricas(y_real, [0] * 10)
    assert metricas["acuracia"] == pytest.approx(0.9)
    assert metricas["recall_pulsar"] == 0.0


# --- o próprio portão de qualidade precisa ser testado ---------------------------------------


def _metricas(recall, precisao):
    return {"recall_pulsar": recall, "precisao_pulsar": precisao, "n_amostras": 100}


def test_gate_aprova_candidato_melhor_e_dentro_dos_limiares():
    assert decidir(_metricas(0.92, 0.82), _metricas(0.90, 0.82))["aprovado"]


def test_gate_reprova_recall_abaixo_do_limiar():
    assert not decidir(_metricas(0.84, 0.99), _metricas(0.80, 0.99))["aprovado"]


def test_gate_reprova_precisao_abaixo_do_limiar():
    # recall alto não compensa uma precisão ruim (cenário 5: quase tudo vira pulsar)
    assert not decidir(_metricas(0.96, 0.33), _metricas(0.92, 0.82))["aprovado"]


def test_gate_reprova_candidato_pior_que_o_campeao():
    assert not decidir(_metricas(0.88, 0.90), _metricas(0.92, 0.82))["aprovado"]
