"""Gera um lote novo de candidatos a pulsar em data/raw/lotes/ (com ou sem drift).

O lote sai da reserva do HTRU2 (linhas que não estão no treino nem no holdout). O drift simula
um receptor recalibrado: o ganho muda e a média e o desvio do perfil integrado e da curva DM-SNR
(config.COLUNAS_ESCALA) são multiplicados pelo fator. Curtose e assimetria não mudam com a escala.

Exemplos:
    python demo/gerar_lote.py lote_02                      # sem drift
    python demo/gerar_lote.py lote_03 --drift 0.8 --n 120  # ganho 20% menor
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops_demo.config import ALVO, COLUNAS_ESCALA  # noqa: E402
from mlops_demo.data import gerar_lote, validar_dataframe  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nome", help="nome do arquivo, sem extensão")
    ap.add_argument("--n", type=int, default=120, help="quantidade de candidatos (padrão: 120)")
    ap.add_argument(
        "--drift",
        type=float,
        default=1.0,
        help=f"fator de ganho aplicado a {', '.join(COLUNAS_ESCALA)} (1.0 = sem drift)",
    )
    ap.add_argument("--semente", type=int, default=1)
    args = ap.parse_args()
    if args.drift <= 0:
        ap.error("--drift precisa ser maior que zero (ganho nulo ou negativo não existe)")

    caminho = Path(gerar_lote(args.nome, args.n, args.drift, args.semente))
    lote = pd.read_csv(caminho)
    print(f"criado: {caminho} | {len(lote)} candidatos | {int(lote[ALVO].sum())} pulsares")
    if args.drift != 1.0:
        print(f"drift: {', '.join(COLUNAS_ESCALA)} multiplicados por {args.drift}")

    # o mesmo validador do CI: avisa já aqui se o lote vai deixar o CI vermelho
    problemas = validar_dataframe(lote)
    # o "preparar" do CT divide cada lote ao meio, estratificado: precisa de 2+ de cada classe
    if lote[ALVO].nunique() < 2 or lote[ALVO].value_counts().min() < 2:
        problemas.append("menos de 2 pulsares ou de 2 ruídos: o CT não divide (aumente --n)")
    if problemas:
        print("ATENÇÃO, este lote vai quebrar o pipeline: " + "; ".join(problemas))
        sys.exit(1)
