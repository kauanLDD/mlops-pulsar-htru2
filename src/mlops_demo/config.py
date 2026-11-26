"""Configuração central do projeto: caminhos, esquema dos dados e limiares de qualidade."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]

# Dados
DADOS_FONTE = RAIZ / "data" / "fonte"
DATASET_ORIGINAL = DADOS_FONTE / "htru2.csv"

DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_LOTES = DADOS_BRUTOS / "lotes"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"

BASE_TREINO = DADOS_BRUTOS / "base_treino.csv"
BASE_HOLDOUT = DADOS_BRUTOS / "base_holdout.csv"

TREINO_PROCESSADO = DADOS_PROCESSADOS / "treino.csv"
AVALIACAO_PROCESSADA = DADOS_PROCESSADOS / "avaliacao.csv"

# Modelos e relatórios
MODELO_CAMPEAO = RAIZ / "models" / "champion" / "model.joblib"
METRICAS_CAMPEAO = RAIZ / "models" / "champion" / "metrics.json"
MODELO_CANDIDATO = RAIZ / "models" / "candidate" / "model.joblib"
METRICAS_CANDIDATO = RAIZ / "reports" / "candidate_metrics.json"
RELATORIO_GATE = RAIZ / "reports" / "gate.json"
RESUMO_GATE = RAIZ / "reports" / "gate.md"
RELATORIO_DRIFT = RAIZ / "reports" / "drift.md"
PARAMS = RAIZ / "params.yaml"

# Esquema dos dados HTRU2
FEATURES = [
    "perfil_media",
    "perfil_desvio",
    "perfil_curtose",
    "perfil_assimetria",
    "dmsnr_media",
    "dmsnr_desvio",
    "dmsnr_curtose",
    "dmsnr_assimetria",
]

ALVO = "pulsar"

# Classe positiva: candidato identificado como pulsar
CLASSE_POSITIVA = 1

# Atributos afetados pela mudança de escala do receptor
COLUNAS_ESCALA = [
    "perfil_media",
    "perfil_desvio",
    "dmsnr_media",
    "dmsnr_desvio",
]

# Limiares de qualidade (contrato do modelo)
RECALL_MINIMO = 0.85
PRECISAO_MINIMA = 0.80

TOLERANCIA_CAMPEAO = 0.0
PISO_SANIDADE = 0.80

# Detecção de drift
LIMITE_DRIFT_P = 0.01
MIN_FEATURES_DRIFT = 3


def carregar_params() -> dict:
    """Lê o params.yaml (hiperparâmetros e semente)."""
    with open(PARAMS, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
