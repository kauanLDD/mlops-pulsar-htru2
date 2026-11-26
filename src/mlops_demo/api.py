"""API de inferência (FastAPI) servindo o modelo campeão de triagem de pulsares (HTRU2).

Recebe os 8 atributos estatísticos de um candidato (perfil integrado e curva DM-SNR) e responde
se ele deve ir para a inspeção de um astrônomo (pulsar) ou pode ser descartado (ruído/RFI).

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from mlops_demo import __version__, config
from mlops_demo.data import validar_dataframe

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")

N_ATRIBUTOS = len(config.FEATURES)
EXEMPLO_PULSAR = [23.984, 30.686, 5.453, 32.176, 88.471, 65.777, 0.624, 0.044]

RECOMENDACAO = {
    "pulsar": "encaminhar para inspeção visual por um astrônomo",
    "ruido": "descartar: perfil compatível com ruído ou interferência de rádio (RFI)",
}

app = FastAPI(
    title="Triagem de candidatos a pulsar (HTRU2, demo CI/CD/CT)",
    description="Classifica candidatos do survey HTRU como pulsar ou ruído.",
    version=__version__,
)
_modelo = joblib.load(CAMINHO_MODELO)


@app.exception_handler(RequestValidationError)
async def responder_422(_: Request, erro: RequestValidationError) -> JSONResponse:
    """422 sem ecoar a entrada: um NaN ou infinito no corpo quebraria o JSON da resposta (500)."""
    detalhes = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in erro.errors()]
    return JSONResponse(status_code=422, content={"detail": detalhes})


class Candidato(BaseModel):
    """Os 8 atributos do candidato, na ordem de config.FEATURES: média, desvio, curtose e
    assimetria do perfil integrado, seguidos dos mesmos quatro da curva DM-SNR."""

    atributos: list[float] = Field(
        min_length=N_ATRIBUTOS,
        max_length=N_ATRIBUTOS,
        description="Ordem: " + ", ".join(config.FEATURES),
        examples=[EXEMPLO_PULSAR],
    )

    @field_validator("atributos")
    @classmethod
    def respeitar_faixas_do_dominio(cls, valores: list[float]) -> list[float]:
        """Aplica as mesmas regras de domínio da validação dos dados de treino.

        Valores impossíveis (desvio <= 0, perfil acima de 8 bits, NaN, infinito) indicam um
        pipeline de sinal quebrado: melhor recusar (422) do que devolver uma previsão sem sentido.
        """
        linha = pd.DataFrame([valores], columns=config.FEATURES)
        # o alvo é desconhecido na inferência: rótulo fictício só para reaproveitar o validador
        linha[config.ALVO] = config.CLASSE_POSITIVA
        problemas = validar_dataframe(linha)
        if problemas:
            raise ValueError("candidato inválido: " + "; ".join(problemas))
        return valores


class Predicao(BaseModel):
    classe: Literal["pulsar", "ruido"]
    probabilidade_pulsar: float
    recomendacao: str
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "versao_codigo": __version__,
        "atributos_esperados": config.FEATURES,
        "metricas_do_campeao": metricas,
    }


@app.post("/predict", response_model=Predicao)
def prever(candidato: Candidato) -> Predicao:
    entrada = pd.DataFrame([candidato.atributos], columns=config.FEATURES)
    previsto = int(_modelo.predict(entrada)[0])
    # a coluna da classe pulsar (1) em predict_proba segue a ordem de classes_
    coluna = list(_modelo.classes_).index(config.CLASSE_POSITIVA)
    p_pulsar = float(_modelo.predict_proba(entrada)[0][coluna])
    classe = "pulsar" if previsto == config.CLASSE_POSITIVA else "ruido"
    return Predicao(
        classe=classe,
        probabilidade_pulsar=round(p_pulsar, 4),
        recomendacao=RECOMENDACAO[classe],
        versao_codigo=__version__,
    )
