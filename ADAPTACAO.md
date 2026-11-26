# Relatório da adaptação

Repositório: https://github.com/kauanLDD/mlops-pulsar-htru2

## 1. Problema e dados

O modelo faz a triagem de candidatos a pulsar. Um radiotelescópio gera milhares de candidatos e quase todos são
ruído ou interferência de rádio. O modelo indica quais vale a pena um astrônomo olhar.

- Dataset: **HTRU2** (Lyon et al., 2016), UCI Machine Learning Repository.
- Licença: CC BY 4.0. Não tem dado pessoal, são medidas de sinal de rádio.
- Tamanho: 17.898 linhas, 8 atributos numéricos, 9,2% de pulsares.
- Atributos: média, desvio, curtose e assimetria do perfil de pulso integrado (`perfil_*`) e da curva DM-SNR (`dmsnr_*`).
- Divisão com semente fixa: 3.000 linhas em `base_treino.csv`, 1.000 em `base_holdout.csv`, sem sobreposição. O
  restante fica como reserva para gerar os lotes novos.

## 2. Contrato de qualidade

- Classe positiva: **pulsar** (rótulo 1).
- Erro mais caro: **falso negativo**. Um pulsar descartado na triagem é uma descoberta perdida. Um falso positivo
  custa só alguns minutos de alguém olhando o candidato.
- Métrica principal: **recall da classe pulsar >= 0,85**.
- Piso de precisão: **>= 0,80**, para o modelo não marcar tudo como pulsar e encher a fila de candidatos falsos.
- O candidato não pode ter recall menor que o do campeão no mesmo conjunto.
- A acurácia não entra no contrato: chutar "ruído" sempre já dá 91%.

## 3. O que mudou

| Etapa | Arquivo | Mudança |
|---|---|---|
| 1 e 2 | `data/fonte/htru2.csv`, `data/raw/*.csv`, `data.py` | saiu o `load_breast_cancer`, entrou o HTRU2 com divisão de semente fixa |
| 3 | `config.py` | `FEATURES`, `ALVO = "pulsar"`, `CLASSE_POSITIVA = 1`, `COLUNAS_ESCALA`, limiares 0,85 e 0,80, `PISO_SANIDADE = 0,80`, `MIN_FEATURES_DRIFT = 3` |
| 4 | `data.py` (`validar_dataframe`) | tipo numérico, infinitos, desvio <= 0, média negativa, `perfil_media` > 255 e curtose impossível |
| 5 | `params.yaml` | regressão logística com `class_weight: balanced` (sem peso o recall ficava em ~0,80) |
| 6 | `evaluate.py`, `gate.py` | `pos_label = 1` e métricas `recall_pulsar` e `precisao_pulsar` |
| 7 | `data.py` (`aplicar_drift`) | drift = receptor recalibrado: média e desvio de perfil e DM-SNR multiplicados pelo fator |
| 8 | `api.py`, `demo/smoke_test.py` | os 8 atributos nomeados e validados, resposta `pulsar`/`ruido` com `probabilidade_pulsar` |
| 9 | `tests/` | os quatro arquivos adaptados, com testes novos por camada e o teste direcional adaptado ao domínio |
| 10 | `demo/cenarios.py` | cenários 1 a 5 com as colunas do HTRU2 |
| 11 | `models/champion/` | campeão inicial treinado na base nova e depois promovido pelo CT |
| 12 | `README.md` | descrição do problema de pulsares |

## 4. Evidências

| | O que mostra | Execução |
|---|---|---|
| E1 | CI verde na main depois da adaptação | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45731192680 |
| E2 | CI vermelho em "Testes de código" (cenário 1) | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45743748798 |
| E3 | CT com drift em 4 de 8 atributos e portão aprovando (cenário 4) | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45734745542 |
| E4 | CT vermelho no portão: recall 0,96 e precisão 0,33 (cenário 5) | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45743570763 |
| E5 | CD da tag v1.0.0 | https://github.com/kauanLDD/mlops-pulsar-htru2/actions/runs/45743751808 |

Depois do E3, o CT abriu o PR do novo campeão (recall 0,908 e precisão 0,824), que foi revisado e aceito. No E4 o
campeão não mudou.

## 5. Limites e riscos

O portão só olha números agregados num conjunto de avaliação pequeno (o holdout tem 92 pulsares, então cada acerto
mexe cerca de 1 ponto no recall). Ele não vê se o modelo erra mais num tipo de pulsar, não mede calibração da
probabilidade e, se o drift também chegar na avaliação, compara os modelos num conjunto que já mudou. A precisão
do campeão (0,82) está perto do limite de 0,80.

Em produção monitoraríamos o drift por atributo (como o teste KS já faz), a proporção de candidatos marcados como
pulsar, o recall real com os rótulos que os astrônomos confirmam depois, a distribuição da probabilidade prevista,
os erros 422 da API e a latência.

## 6. Contribuição de cada integrante e uso de IA

| Integrante | Contribuição |
|---|---|
| Kauan | criação do repositório, dados (etapas 1 e 2), merge, E2 e E5 |
| Arthur Nicolas Oliveira | contrato e validação (etapas 3 e 4) |
| Lucas Gabriel Teixeira | modelo, métricas e drift (etapas 5 a 7), E3 |
| gschr4y | API e cenários (etapas 8 e 10), E4 |
| Samuel Malta | testes e campeão inicial (etapas 9 e 11) |

Uso de IA: usamos o Claude Code (Anthropic) para ajudar na escolha do dataset, na organização das tarefas, na
conferência do código e no rascunho deste relatório. Os resultados foram conferidos nos logs do GitHub Actions.
