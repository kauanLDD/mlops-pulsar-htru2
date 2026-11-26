"""Testes de contrato da API: formato de entrada e de saída do /predict (HTRU2)."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import __version__, config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


def test_health_responde_ok_e_informa_os_atributos():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["versao_codigo"] == __version__
    assert corpo["atributos_esperados"] == config.FEATURES


def test_predict_devolve_o_formato_combinado(base_holdout):
    candidato = base_holdout[config.FEATURES].iloc[0].tolist()
    resposta = cliente.post("/predict", json={"atributos": candidato})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_pulsar", "recomendacao", "versao_codigo"}
    assert corpo["classe"] in {"pulsar", "ruido"}
    assert 0.0 <= corpo["probabilidade_pulsar"] <= 1.0
    assert corpo["recomendacao"]


def test_predict_rejeita_quantidade_errada_de_atributos():
    resposta = cliente.post("/predict", json={"atributos": [1.0, 2.0]})
    assert resposta.status_code == 422


def test_predict_rejeita_os_30_atributos_do_projeto_antigo():
    resposta = cliente.post("/predict", json={"atributos": [1.0] * 30})
    assert resposta.status_code == 422


@pytest.mark.parametrize(
    ("coluna", "valor"),
    [("perfil_desvio", -3.0), ("perfil_media", 300.0), ("dmsnr_curtose", -10.0)],
    ids=["desvio-negativo", "perfil-acima-de-8-bits", "curtose-impossivel"],
)
def test_predict_rejeita_valores_fora_da_faixa_fisica(base_holdout, coluna, valor):
    candidato = base_holdout[config.FEATURES].iloc[0].copy()
    candidato[coluna] = valor
    resposta = cliente.post("/predict", json={"atributos": candidato.tolist()})
    assert resposta.status_code == 422


def test_predict_acerta_um_pulsar_e_um_ruido_claros(base_holdout, campeao):
    """Escolhe os casos mais claros (maior confiança do campeão) de cada classe."""
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_POSITIVA)
    p_pulsar = campeao.predict_proba(X)[:, coluna]
    i_pulsar = int(p_pulsar.argmax())
    i_ruido = int(p_pulsar.argmin())
    for indice, esperado in ((i_pulsar, "pulsar"), (i_ruido, "ruido")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.json()["classe"] == esperado
