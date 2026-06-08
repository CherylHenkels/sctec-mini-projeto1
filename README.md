# Pipeline de Análise Preditiva de Vendas (SCTEC)***

Este projeto consiste em um pipeline automatizado de extração, limpeza, transformação e análise preditiva de dados de vendas. Desenvolvido como o Mini-Projeto Avaliativo do Módulo 1 (Semana 08) do curso de **Desenvolvedor(a) em IA para Análise Preditiva**, o sistema utiliza conceitos avançados de Programação Orientada a Objetos (POO), manipulação de dados com Pandas e NumPy, expressões regulares (Regex) e funções de ordem superior.

## Objetivo do Projeto

O objetivo principal é simular o ciclo de vida de engenharia e análise de dados em um cenário de e-commerce. O sistema realiza o tratamento de dados brutos de vendas, higieniza informações sensíveis, segmenta clientes por comportamento de compra, gera projeções estatísticas de faturamento futuro e exporta relatórios prontos para tomada de decisão.

---

## Tecnologias e Bibliotecas Utilizadas

- **Python 3.10+**
- **Pandas**: Manipulação e análise estruturada de dados (DataFrames).
- **NumPy**: Cálculos estatísticos e projeções matemáticas de tendência.
- **Matplotlib & Seaborn**: Geração de gráficos para visualização de métricas.
- **re (Regular Expressions)**: Limpeza e padronização de strings (IDs de clientes).
- **JSON e OS**: Manipulação de arquivos de configuração e sistema de diretórios.

---

## 
Estrutura do Projeto

Ao ser executado, o script organiza os resultados na seguinte estrutura de pastas:

```text
MINI-PROJETO1/
│
├── README.md
├── salesinsight.py                 # Script principal da aplicação
├── vendas.csv                      # Dataset de vendas
│
├── outputs/
│   ├── estatisticas_gerais.json    # Estatísticas consolidadas
│   ├── metricas_por_mes.csv        # Métricas agregadas por mês
│   ├── segmentacao_clientes.csv    # Resultado da segmentação de clientes
│   │
│   └── graficos/
│       ├── distribuicao_faturamento_histograma.png
│       ├── distribuicao_regioes.png
│       ├── share_faturamento_categoria.png
│       ├── top_produtos.png
│       └── vendas_por_mes.png
│
├── planejamento/
│   └── tarefas-kanban.md           # Planejamento e organização das tarefas
│
└── utils/
    └── gera_dataset.py             # Geração de dados fictícios para testes
```

