## O que mudou

<!-- Código, dados (lote novo), hiperparâmetros, limiares ou contrato da API? Em uma frase. -->

**Tipo:** <!-- código | dados | params.yaml | contrato (config.py) | API | cenário de prática (N) -->

## Checklist de ML (triagem de pulsares, HTRU2)

- [ ] `python demo/local.py ci` passou na minha máquina (lint, dados, código, modelo e API)
- [ ] Se mudei `params.yaml` ou o código de treino, conferi o **recall** (>= 0,85) e a **precisão** (>= 0,80) da classe **pulsar** (rótulo 1)
- [ ] Se mudei os limiares ou as colunas em `config.py`, expliquei o motivo abaixo (custo do falso negativo: um pulsar real descartado)
- [ ] Se adicionei um lote em `data/raw/lotes/`, rodei `python -m mlops_demo.data validar` e `python -m mlops_demo.monitor`
- [ ] Se mudei o contrato da API (`api.py`), atualizei `tests/test_api.py` e `demo/smoke_test.py`
- [ ] Nenhum segredo, token ou senha no diff

## Impacto no modelo

<!-- Cole a tabela do portão de qualidade (Summary do CT ou reports/gate.md) e o relatório de drift
     (reports/drift.md) quando o PR mexer em dados ou treino. -->

## Evidência

<!-- Link da execução do Actions (CI e/ou CT) deste PR. Se for uma evidência do trabalho (E1 a E5),
     diga qual e qual resultado era esperado. -->
