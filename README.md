# 🧠 Specializing LLMs for Medical Diagnosis: A Fine-Tuning-Based Approach

Este repositório contém os artefatos desenvolvidos para a pesquisa de TCC intitulada **“Specializing LLMs for Medical Diagnosis: A Fine-Tuning-Based Approach”**, conduzida por **Mário Guerra** como parte da graduação em Engenharia da Computação pela UPE. A proposta central é investigar o impacto da sobreposição de datasets médicos heterogêneos na performance de **Language Models** aplicados ao diagnóstico clínico.

---

## 📌 Hipótese de Pesquisa

> *“A sobreposição de datasets heterogêneos, por meio da síntese de características médicas comuns e representativas, reduz o viés e aumenta a precisão de LLMs no diagnóstico médico.”*

---

## 🗂 Estrutura do Repositório

```
📁 medical_llm/
│
├── 📒 1_data_preprocessing.ipynb     # Pré-processamento, unificação e enriquecimento dos dados
├── 📒 2_data_completion_biogpt.ipynb # Geração de dados sintéticos com BioGPT
├── 📒 3_model_training.ipynb         # Preparação dos pares texto-alvo, tokenização e fine-tuning
├── 📂 data/                          # Datasets originais e processados
├── 📂 utils/                         # Scripts auxiliares (tokenização, limpeza, etc.)
└── README.md                        # Este arquivo
```

---

## 📚 Bases de Dados Utilizadas

A pesquisa integra e sobrepõe datasets distintos do tipo **disease-symptom**, com sintomas codificados como vetores binários (`0` ou `1`) e uma coluna-alvo com o diagnóstico correspondente:

- **OpenML Medical Datasets**  
- **SymCAT** – Diagnósticos mapeados a partir de sintomas com base em prevalência.
- **Disease Symptom Knowledge Database** – Relações estruturadas entre doenças e sintomas.
- *(Suporte secundário: MIMIC-III, eICU, ClinicalTrials – referenciados para futura extensão)*

---

## 🧪 Metodologia

### ✅ 1. Coleta, Limpeza e Unificação
- Normalização de nomes de sintomas e doenças via **SciSpacy** + **UMLS**.
- Engenharia de features para harmonização inter-base.
- Enriquecimento semântico com descrições e fatores de risco (via web scraping da [Mayo Clinic](https://www.mayoclinic.org/)).

### ✅ 2. Preenchimento com BioGPT
- Doenças sem descrição foram completadas via geração sintética com **BioGPT**.
- A geração abrangeu:
  - Definições clínicas
  - Fatores de risco
  - Contexto epidemiológico

### ✅ 3. Preparação dos Dados para Fine-Tuning
- Construção de pares `input-text` → `diagnóstico esperado`.
- Exemplo:
  ```
  Input: Patient presents with fatigue, low-grade fever, muscle pain and dry cough.
  Output: Influenza
  ```
- Tokenização com tokenizer do **BioGPT**.

### ✅ 4. Fine-Tuning do BioGPT
- Treinamento local com instância do modelo usando PyTorch + HuggingFace Transformers.
- Divisão de dados:
  - 70% treino
  - 15% validação
  - 15% teste

---

## 🧠 Arquitetura do Modelo

- **Modelo-base**: [`BioGPT`](https://huggingface.co/microsoft/BioGPT)
- **Paradigma de ajuste**: Fine-tuning supervisionado
- **Hiperparâmetros principais**:
  - Learning rate: `5e-5`
  - Batch size: `16`
  - Epochs: `5`

---

## 📏 Avaliação de Desempenho

- **Métricas aplicadas**:
  - Accuracy
  - F1-Score
  - Recall
  - AUROC
  - ROUGE (em respostas explicativas)

- **Estratégia de Controle**:
  - Comparação com modelo base (BioGPT sem fine-tuning)
  - Testes com casos reais e simulados
  - Validação qualitativa por especialistas médicos (etapa futura)

---

## 📈 Resultados Esperados

A sobreposição de múltiplas fontes clínicas resulta em:
- Maior cobertura de doenças raras
- Redução de viés de prevalência
- Geração de descrições clinicamente coerentes
- Melhora na acurácia de predição em sintomas diversos

---

## 🔬 Conclusão Provisória

A fusão de datasets médicos complementares, aliada ao enriquecimento textual, potencializa a especialização de LLMs para tarefas diagnósticas. A capacidade do modelo de generalizar para contextos clínicos heterogêneos aumenta consideravelmente com a presença de exemplos variados e bem documentados.

---

## 🛠 Tecnologias e Ferramentas

- `Python`, `Pandas`, `NumPy`
- `SciSpacy`, `UMLS`, `requests`, `BeautifulSoup`
- `HuggingFace Transformers`, `BioGPT`
- `Jupyter Notebooks`

---

## 📎 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/MguerraIV/medical_llm.git
   cd medical_llm
   ```

2. Crie um ambiente e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os notebooks na ordem:
   - `1_data_preprocessing.ipynb`
   - `2_data_completion_biogpt.ipynb`
   - `3_model_training.ipynb`

---

## 🧾 Referências Bibliográficas

- Lee, J. et al. (2020). *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. Bioinformatics.
- Chen, Q. et al. (2022). *BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining*.
- Mayo Clinic. [https://www.mayoclinic.org](https://www.mayoclinic.org)
- NEUMANN, Mark et al. ScispaCy: fast and robust models for biomedical natural language processing. In: Proceedings of the 18th BioNLP Workshop and Shared Task, Florence, Italy, 2019. Association for Computational Linguistics, p. 319–327. Disponível em: https://aclanthology.org/W19-5034.
- BODENREIDER, Olivier. The Unified Medical Language System (UMLS): integrating biomedical terminology. Nucleic Acids Research, [S.l.], v. 32, n. Database issue, p. D267–D270, 2004. Oxford University Press. Disponível em: https://doi.org/10.1093/nar/gkh061. 
---

## ✉️ Contato

**Mário Guerra**  
Universidade de Pernambuco – UPE  
📧 [msg4@poli.br](mailto:msg4@poli.br)
