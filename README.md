# DataCareer-Predictor

## Descrição do Projeto

O DataCareer-Predictor é um projeto de classificação vocacional focado no mercado de dados brasileiro. Ele emprega o processo de Descoberta de Conhecimento em Bases de Dados (KDD), uma arquitetura de software modular e modelos de Machine Learning, especificamente Random Forest e Multi-Layer Perceptron (MLP), para prever carreiras no setor de dados. O projeto utiliza o dataset 'State of Data' para suas análises e modelagem.

## Visão Geral

### Problema

Com o crescimento exponencial do mercado de dados, profissionais e aspirantes a profissionais enfrentam o desafio de identificar as carreiras mais adequadas às suas habilidades e interesses. A complexidade das diversas funções e a rápida evolução do setor tornam essa decisão ainda mais difícil.

### Solução

Este projeto propõe uma solução baseada em Machine Learning para auxiliar na orientação vocacional dentro do mercado de dados. Através da análise de um extenso conjunto de dados sobre o perfil de profissionais da área, o DataCareer-Predictor busca identificar padrões e construir modelos preditivos capazes de sugerir carreiras com base nas características individuais dos usuários.

### Metodologia

A metodologia adotada segue o processo KDD, que compreende as seguintes etapas:

1.  **Seleção de Dados:** Identificação e filtragem de instâncias relevantes do dataset 'State of Data'.
2.  **Pré-processamento:** Tratamento de ruídos, valores nulos, e transformação de atributos (e.g., experiência profissional).
3.  **Transformação:** Engenharia de características para otimizar a representação dos dados para os modelos de ML.
4.  **Mineração de Dados:** Aplicação de algoritmos de Machine Learning (Random Forest e MLP) para construir os modelos preditivos.
5.  **Avaliação:** Análise do desempenho dos modelos utilizando métricas apropriadas e validação cruzada (Hold-out).

A arquitetura modular do projeto garante a organização e a manutenibilidade do código, separando as responsabilidades de carregamento de dados, pré-processamento, modelagem e visualização em módulos distintos.

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

```
. 
├── Notebooks/              # Notebooks Jupyter para EDA, pré-processamento e treinamento de modelos
│   ├── 01_eda.ipynb        # Análise Exploratória de Dados
│   ├── 02_preprocessing.ipynb # Pré-processamento de dados
│   ├── 03_model_training.ipynb # Treinamento e avaliação dos modelos
│   └── final_submission.ipynb # Notebook final com o fluxo completo
├── data/                   # Armazena o dataset bruto (sods.csv)
├── reports/                # Relatórios e figuras geradas
│   └── figures/            # Figuras e gráficos
├── src/                    # Módulos Python com a lógica do projeto
│   ├── __init__.py         # Inicialização do pacote Python
│   ├── data_loader.py      # Funções para carregamento e seleção de dados
│   ├── models.py           # Implementação e treinamento dos modelos de ML
│   ├── preprocess.py       # Funções para limpeza e transformação de dados
│   └── visualization.py    # Funções para visualização de dados (inferido)
├── .gitignore              # Arquivos e diretórios a serem ignorados pelo Git
├── poetry.lock             # Bloqueio de dependências do Poetry
└── pyproject.toml          # Definições do projeto e dependências (Poetry)
```

## Tecnologias Utilizadas

O projeto foi desenvolvido utilizando as seguintes tecnologias e bibliotecas:

*   **Python** (>=3.12)
*   **Pandas** (>=2.0.1, <3.0.0): Manipulação e análise de dados.
*   **NumPy** (>=1.24.2, <3.0.0): Suporte a operações numéricas.
*   **scikit-learn** (>=1.8.0, <2.0.0): Implementação de algoritmos de Machine Learning (Random Forest, MLP) e utilitários.
*   **Matplotlib** (>=3.10.0, <4.0.0): Geração de gráficos e visualizações.
*   **Seaborn** (>=0.13.2, <0.14.0): Visualização estatística de dados.
*   **Joblib** (>=1.5.3, <2.0.0): Persistência de modelos Python.
*   **Poetry**: Gerenciamento de dependências e empacotamento do projeto.

## Instalação

Para configurar o ambiente de desenvolvimento e executar o projeto, siga os passos abaixo:

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/Code-AldreySandre/DataCareer-Predictor.git
    cd DataCareer-Predictor
    ```

2.  **Instale o Poetry** (se ainda não tiver):

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3.  **Instale as dependências do projeto:**

    ```bash
    poetry install
    ```

4.  **Ative o ambiente virtual do Poetry:**

    ```bash
    poetry shell
    ```

## Uso

Após a instalação, você pode explorar os notebooks Jupyter na pasta `Notebooks/` para entender o fluxo completo do projeto, desde a análise exploratória de dados até o treinamento e avaliação dos modelos.

Para executar os notebooks, certifique-se de que o ambiente virtual do Poetry esteja ativado e inicie o Jupyter Lab ou Jupyter Notebook:

```bash
poetry run jupyter lab
# ou
poetry run jupyter notebook
```

## Resultados

O projeto implementa e avalia modelos de classificação (Random Forest e MLP) para prever carreiras no mercado de dados. Os resultados detalhados, incluindo métricas de desempenho e análises comparativas, podem ser encontrados nos notebooks de treinamento e nos relatórios gerados.

## Contribuição

Contribuições são bem-vindas! Se você deseja contribuir para este projeto, por favor, siga estas diretrizes:

1.  Faça um fork do repositório.
2.  Crie uma nova branch para sua feature (`git checkout -b feature/sua-feature`).
3.  Faça suas alterações e commit (`git commit -m 'feat: Adiciona nova feature'`).
4.  Envie para a branch (`git push origin feature/sua-feature`).
5.  Abra um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` (se existir) para mais detalhes.

## Contato

Para dúvidas ou sugestões, entre em contato com Aldrey Sandre via GitHub.
