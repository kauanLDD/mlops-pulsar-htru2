"""Cenários de prática no HTRU2: cada um aplica uma mudança para você ver o pipeline reagir.

Uso (a partir da raiz do projeto, sempre em uma branch nova):
    python demo/cenarios.py listar
    python demo/cenarios.py aplicar 1

Depois de aplicar: git add -A, git commit, git push e abra o Pull Request.

Os números abaixo foram medidos com a base fixa (semente 7) e o params.yaml da adaptação:
campeão inicial com recall 0,92 e precisão 0,82 do pulsar no holdout (contrato: 0,85 e 0,80).
"""

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mlops_demo.data import gerar_lote  # noqa: E402

DESCRICAO = {
    1: (
        "Bug no pré-processamento: dmsnr_assimetria some da matriz X (alguém achou a coluna "
        "redundante). Esperado: CI vermelho no step 'Testes de código'."
    ),
    2: (
        "Dados corrompidos na base de treino: perfil_media nulo, perfil_desvio negativo e "
        "perfil_media acima de 255 (8 bits). Esperado: CI vermelho no step "
        "'Validar os dados brutos'."
    ),
    3: (
        "Regularização excessiva (C = 0.00005): o recall do pulsar cai para ~0,78. "
        "Esperado: CI vermelho no step 'Testes de modelo' (recall < 0,85)."
    ),
    4: (
        "Lote novo com receptor recalibrado (ganho 20% menor em média e desvio do perfil e da "
        "DM-SNR). Esperado: CI verde; CT detecta drift em 4 de 8 atributos, retreina e o portão "
        "aprova."
    ),
    5: (
        "class_weight extremo (peso 50 no pulsar): quase tudo vira pulsar, recall ~0,96 e "
        "precisão ~0,33. Esperado: CI vermelho em 'Testes de modelo' e, com o lote do cenário 4 "
        "na main, CT vermelho no 'Portão de qualidade' (campeão não muda)."
    ),
}


def _substituir(arquivo: Path, antigo: str, novo: str) -> None:
    texto = arquivo.read_text(encoding="utf-8")
    if antigo not in texto:
        raise SystemExit(f"Trecho não encontrado em {arquivo.name}. O cenário já foi aplicado?")
    arquivo.write_text(texto.replace(antigo, novo, 1), encoding="utf-8")
    print(f"alterado: {arquivo.relative_to(RAIZ)}")


def cenario_1() -> None:
    _substituir(
        RAIZ / "src" / "mlops_demo" / "features.py",
        "X = df[FEATURES].copy()",
        'X = df[[c for c in FEATURES if c != "dmsnr_assimetria"]].copy()',
    )


def cenario_2() -> None:
    caminho = RAIZ / "data" / "raw" / "base_treino.csv"
    df = pd.read_csv(caminho)
    df.loc[0:4, "perfil_media"] = None  # falha na leitura do perfil integrado
    df.loc[5:7, "perfil_desvio"] = -3.0  # desvio padrão negativo não existe
    df.loc[8:9, "perfil_media"] = 300.0  # acima do máximo de 8 bits (255)
    df.to_csv(caminho, index=False)
    print(
        "alterado: data/raw/base_treino.csv (5 nulos em perfil_media, 3 desvios negativos em "
        "perfil_desvio, 2 perfil_media acima de 255)"
    )


def cenario_3() -> None:
    _substituir(RAIZ / "params.yaml", "C: 1.0", "C: 0.00005")


def cenario_4() -> None:
    # n=120 e fator 0.8 foram escolhidos de propósito: com lotes maiores (300 ou mais) o campeão
    # promovido cai para precisão < 0,80 no holdout antigo e o CI da main fica vermelho depois.
    caminho = gerar_lote("lote_01", n=120, fator_drift=0.8, semente=1)
    print(
        "criado:",
        Path(caminho).relative_to(RAIZ),
        "(perfil_media, perfil_desvio, dmsnr_media e dmsnr_desvio 20% menores)",
    )


def cenario_5() -> None:
    _substituir(RAIZ / "params.yaml", "class_weight: balanced", "class_weight: {0: 1, 1: 50}")


CENARIOS = {1: cenario_1, 2: cenario_2, 3: cenario_3, 4: cenario_4, 5: cenario_5}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "listar":
        for numero, texto in DESCRICAO.items():
            print(f"{numero}. {texto}")
    elif len(sys.argv) == 3 and sys.argv[1] == "aplicar" and sys.argv[2].isdigit():
        numero = int(sys.argv[2])
        if numero not in CENARIOS:
            raise SystemExit("Cenário inexistente. Use: python demo/cenarios.py listar")
        print(f"Cenário {numero}: {DESCRICAO[numero]}")
        CENARIOS[numero]()
    else:
        print(__doc__)
        sys.exit(2)
