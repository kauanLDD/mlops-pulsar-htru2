"""Teste de fumaça: chama /health e /predict de uma API em execução (sem dependências externas).

Envia um pulsar e um ruído reais do holdout e confere o formato da resposta. Depois confere que
a API recusa (422) um candidato com atributos faltando e um com valor fisicamente impossível.

Uso: python demo/smoke_test.py http://localhost:8000
"""

import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
HOLDOUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "base_holdout.csv"
ALVO = "pulsar"  # mesmo valor de config.ALVO (este script não importa o pacote de propósito)


def chamar(caminho: str, corpo: dict | None = None) -> tuple[int, dict]:
    dados = json.dumps(corpo).encode() if corpo is not None else None
    requisicao = urllib.request.Request(
        URL + caminho, data=dados, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=10) as resposta:
            return resposta.status, json.loads(resposta.read())
    except urllib.error.HTTPError as erro:
        return erro.code, json.loads(erro.read() or b"{}")


status, saude = chamar("/health")
assert status == 200 and saude["status"] == "ok", saude
ordem = saude["atributos_esperados"]
print(f"health ok | versão: {saude['versao_codigo']} | {len(ordem)} atributos")

# um exemplo TÍPICO de cada classe (curtose do perfil mediana da classe), montado na ordem que a
# própria API informa. A escolha não depende do modelo; o smoke test confere formato e contrato,
# a qualidade das previsões é conferida pelos testes de modelo do CI.
with open(HOLDOUT, encoding="utf-8") as arquivo:
    linhas = list(csv.DictReader(arquivo))
exemplos = {}
for rotulo in ("1", "0"):
    da_classe = sorted(
        (linha for linha in linhas if linha[ALVO] == rotulo),
        key=lambda linha: float(linha["perfil_curtose"]),
    )
    exemplos[rotulo] = da_classe[len(da_classe) // 2]
for rotulo, linha in exemplos.items():
    atributos = [float(linha[coluna]) for coluna in ordem]
    status, predicao = chamar("/predict", {"atributos": atributos})
    assert status == 200, predicao
    assert predicao["classe"] in {"pulsar", "ruido"}, predicao
    assert 0.0 <= predicao["probabilidade_pulsar"] <= 1.0, predicao
    assert predicao["recomendacao"], predicao
    assert predicao["versao_codigo"] == saude["versao_codigo"], predicao
    real = "pulsar" if rotulo == "1" else "ruido"
    print(f"predict ok | exemplo típico de {real:6} | {predicao}")

# contrato de entrada: quantidade errada de atributos
status, _ = chamar("/predict", {"atributos": atributos[:-1]})
assert status == 422, f"esperava 422 com {len(ordem) - 1} atributos, veio {status}"

# contrato de entrada: desvio padrão negativo não existe
impossivel = dict(zip(ordem, atributos, strict=True), perfil_desvio=-1.0)
status, _ = chamar("/predict", {"atributos": list(impossivel.values())})
assert status == 422, f"esperava 422 com perfil_desvio negativo, veio {status}"
print("validação ok | atributos faltando e desvio negativo recusados com 422")
