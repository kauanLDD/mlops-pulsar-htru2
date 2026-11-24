# Divisão de tarefas do grupo

Problema escolhido: triagem de candidatos a pulsar com o dataset **HTRU2** (UCI, CC BY 4.0, 17.898 linhas,
8 atributos). Classe positiva: pulsar (1). Contrato: recall >= 0,85 e precisão >= 0,80.

Cada um adapta os SEUS arquivos seguindo a Parte C do enunciado, roda o CI local e faz o commit com a própria
conta.

## Antes de começar (todo mundo)

1. Aceitar o convite de colaborador do repositório (o Kauan adiciona em Settings > Collaborators).
2. Clonar e preparar o ambiente:

```bash
git clone https://github.com/kauanLDD/mlops-pulsar-htru2.git
cd mlops-pulsar-htru2
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install -e ".[dev,ct]"
git config --local user.name "Seu Nome"
git config --local user.email "seu-email-do-github"
git config --local core.autocrlf false
```


## Parte 1: adaptação (todos na mesma branch, nesta ordem)

Todo mundo trabalha na branch `adaptacao`. Antes de começar a sua parte: `git switch adaptacao && git pull`.
Depois de adaptar os arquivos: `git add` só dos seus arquivos, `git commit` e `git push`.
O CI do PR vai ficar vermelho até o integrante 5 terminar. É normal, os arquivos dependem uns dos outros.

| # | Quem | Etapas | Arquivos | Commit |
|---|---|---|---|---|
| 1 | Kauan | 1 e 2. Dados | `data/fonte/htru2.csv`, `data/raw/base_treino.csv`, `data/raw/base_holdout.csv`, `src/mlops_demo/data.py` | "troca o dataset para o HTRU2" |
| 2 | Integrante 2 | 3 e 4. Contrato e validação | `src/mlops_demo/config.py` (conferir as regras de `validar_dataframe` no data.py) | "contrato do problema de pulsares" |
| 3 | Integrante 3 | 5, 6 e 7. Modelo, métricas e drift | `params.yaml`, `src/mlops_demo/evaluate.py`, `src/mlops_demo/gate.py` | "modelo e metricas da classe pulsar" |
| 4 | Integrante 4 | 8 e 10. API e cenários | `src/mlops_demo/api.py`, `src/mlops_demo/__init__.py`, `pyproject.toml`, `demo/smoke_test.py`, `demo/cenarios.py`, `demo/gerar_lote.py`, `.github/PULL_REQUEST_TEMPLATE.md` | "api e cenarios do HTRU2" |
| 5 | Integrante 5 | 9 e 11. Testes e campeão | pasta `tests/` inteira. Depois rodar `python -m mlops_demo.train --campeao` e `python demo/local.py ci` (tudo `[ok]`) | "testes adaptados e campeao inicial" |
| 6 | Integrante 6 | 12. Documentação | `README.md` | "readme do projeto de pulsares" |

O integrante 1 cria a branch e abre o PR:

```bash
git switch -c adaptacao
# adaptar os arquivos, commit
git push -u origin adaptacao
gh pr create -B main -H adaptacao -t "Adaptação para o HTRU2" -b "Adaptação do projeto para pulsares"
```

Quando o CI do PR ficar verde (depois do integrante 6), o Kauan faz o merge. O CI da `main` depois do merge é a **E1**.

## Parte 2: evidências (depois do merge, cada um na sua branch)

Roteiro igual ao da Parte 3 do README: `git switch main && git pull`, `git switch -c cenario-N`,
`python demo/cenarios.py aplicar N`, commit, push e abrir o PR.

| Evidência | Quem | O que fazer | Resultado esperado |
|---|---|---|---|
| E2 | Integrante 2 | cenário 1, PR, fechar SEM merge | CI vermelho no step "Testes de código" |
| E3 | Integrante 3 | cenário 4, PR. Ver o Summary do CT. Depois fazer o merge, esperar o CT abrir o PR "ct: novo modelo campeão", clicar em "Approve and run workflows" e fazer o merge desse PR também | CT com drift em 4 de 8 atributos e portão APROVADO |
| E4 | Integrante 4 | cenário 5 (só depois do E3 estar na main), PR, fechar SEM merge | CT vermelho no step "Portão de qualidade", precisão ~0,33 |
| E5 | Integrante 5 | `git tag v1.0.0 && git push origin v1.0.0`. Depois o Kauan muda o pacote para Public (Packages > Package settings) | CD verde e imagem no ghcr.io |
| Relatório | Integrante 6 | escrever o `ADAPTACAO.md` (Parte E do enunciado) com os 5 links dos runs do Actions e a contribuição de cada um | commit "relatorio da adaptacao" |

Cada um manda o link do seu run no grupo para o integrante 6 colocar no relatório.

## Entrega no D2L (Kauan)

URL do repositório, as 5 URLs dos runs (E1 a E5) e o print da página do pacote em Packages.
