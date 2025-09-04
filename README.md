# 🧠 Transformers para análise de registros clínicos heterogêneos e especialização de modelos de linguagem no diagnóstico médico

Este repositório contém os artefatos desenvolvidos para a pesquisa de TCC intitulada **“Transformers para análise de registros clínicos heterogêneos e especialização de modelos de linguagem no diagnóstico médico”**, conduzida por **Mário Guerra** como parte da graduação em Engenharia da Computação pela UPE. A proposta central é investigar o impacto da sobreposição de datasets médicos heterogêneos na performance de **Language Models** aplicados ao diagnóstico clínico.

---

## 📌 Hipótese de Pesquisa

> *“A sobreposição de datasets heterogêneos, por meio da síntese de características médicas comuns e representativas, reduz o viés e aumenta a precisão de LLMs no diagnóstico médico.”*

---

## 🗂 Estrutura do Repositório

```
📁 medical_llm/
│
├── 📒 1_data_preprocessing.ipynb     # Pré-processamento, unificação e enriquecimento dos dados
├── 📒 2_data_completion.ipynb        # Geração de dados sintéticos e enriquecimento dos dados
├── 📒 3_model_training.py            # Preparação dos pares texto-alvo, tokenização e fine-tuning
├── 📒 4_model_evaluate.ipynb           # Avaliação do modelo treinado em comparação ao modelo base
├── 📂 data/                          # Datasets originais e processados
├── 📂 utils/                         # Scripts auxiliares (tokenização, limpeza, etc.)
├── Requirements.txt                  # Bibliotecas necessárias para o projeto
└── README.md                        # Este arquivo
```

---

## 📚 Bases de Dados Utilizadas

A pesquisa integra e sobrepõe datasets distintos do tipo **disease-symptom**, com sintomas codificados como vetores binários (`0` ou `1`) e uma coluna-alvo com o diagnóstico correspondente:

- **University of Columbia Disease Symptom Database**  - Base de conhecimento com associações entre doenças e sintomas, gerada por um método automatizado com base em informações extraídas de sumários de alta hospitalar em formato textual de pacientes do Hospital Presbiteriano de Nova York, admitidos durante o ano de 2004. 

- **Symbi Predict** – Uma coleção abrangente de dados estruturados que relacionam sintomas a diversas doenças, meticulosamente curada para facilitar pesquisas e o desenvolvimento de análises preditivas em saúde. Inspirado na metodologia empregada por instituições renomadas como os Centers for Disease Control and Prevention (CDC)

- **Disease Symptom Knowledge Database** – O conjunto de dados contém nomes de doenças juntamente com os sintomas apresentados pelo respectivo paciente. Há um total de 773 doenças únicas e 377 sintomas, com aproximadamente 246.000 linhas. O conjunto de dados foi gerado artificialmente, preservando a severidade dos sintomas e a probabilidade de ocorrência das doenças.

- *(Suporte secundário: MIMIC-III, eICU, ClinicalTrials – referenciados para futura extensão)*

---

## 🧪 Metodologia

### ✅ 1. Coleta, Limpeza e Unificação
- Normalização de nomes de sintomas e doenças via **SciSpacy** + **UMLS**.
- Engenharia de features para harmonização inter-base.
- Enriquecimento semântico com descrições e fatores de risco (via web scraping da [Mayo Clinic](https://www.mayoclinic.org/) e **SciSpacy** + **UMLS**).

### ✅ 2. Preenchimento com BioGPT
- Doenças sem descrição foram completadas via geração sintética com **BioGPT**.
- A geração abrangeu:
  - Definições clínicas
  - Fatores de risco

### ✅ 3. Preparação dos Dados para Fine-Tuning
- Construção de pares `input-text` → `diagnóstico esperado`.
- Exemplo:
  ```
  Input: Given the following symptoms, provide the most likely diagnosis, description, and risk factors.\n\nSymptoms: anxiety and nervousness, breathing fast, chest tightness.

  Output: 
    Diagnosis: Panic Disorder.

    Description: A type of anxiety disorder characterized by unexpected panic attacks that last minutes or, rarely, hours. Panic attacks begin with intense apprehension, fear or terror and, often, a feeling of impending doom. Symptoms experienced during a panic attack include dyspnea or sensations of being smothered; dizziness, loss of balance or faintness; choking sensations; palpitations or accelerated heart rate; shakiness; sweating; nausea or other form of abdominal distress; depersonalization or derealization; paresthesias; hot flashes or chills; chest discomfort or pain; fear of dying and fear of not being in control of oneself or going crazy. Agoraphobia may also develop. Similar to other anxiety disorders, it may be inherited as an autosomal dominant trait.

    Risk factors: Symptoms of panic disorder often start in the late teens or early adulthood and affect more women than men. Factors that may increase the risk of developing panic attacks or panic disorder include: Family history of panic attacks or panic disorderMajor life stress, such as the death or serious illness of a loved oneA traumatic event, such as sexual assault or a serious accidentMajor changes in your life, such as a divorce or the addition of a babySmoking or excessive caffeine intakeHistory of childhood physical or sexual abuse.
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
  - Batch size: `3`
  - Epochs: `3`

---

## 📏 Avaliação de Desempenho

A avaliação do modelo treinado é conduzida de forma **híbrida**, combinando métricas quantitativas tradicionais com análise qualitativa de outputs médicos.

---

### **1️⃣ Métricas Aplicadas**

- **Accuracy (Acurácia)**: mede a proporção de diagnósticos corretos fornecidos pelo modelo em relação ao gabarito clínico.
- **F1-Score**: combina precisão e recall para avaliar a performance em classificação de diagnósticos, penalizando falsos positivos e falsos negativos.
- **Recall**: indica a capacidade do modelo de identificar corretamente todos os diagnósticos relevantes presentes nos casos de teste.
- **AUROC (Área sob a Curva ROC)**: avalia a separação entre diagnósticos corretos e incorretos em um cenário probabilístico, útil para análise de sensibilidade do modelo.
- **ROUGE / BLEU (em respostas explicativas)**: mede similaridade textual entre descrições e fatores de risco gerados pelo modelo e os textos de referência, garantindo que a argumentação clínica seja coerente e completa.

---

### **2️⃣ Estratégia de Controle**

- **Comparação com modelo base**:  
  O desempenho do THERA é comparado com o modelo **BioGPT sem fine-tuning**, permitindo quantificar o ganho obtido pelo fine-tuning em tarefas de diagnóstico médico.

- **Testes com casos reais e simulados**:  
  O conjunto de avaliação inclui **casos clínicos reais** e **casos sintéticos gerados para cobrir cenários raros ou críticos**, garantindo robustez do modelo.

- **LLM-as-a-judge**:  
  Além das métricas automáticas, utilizamos uma estratégia de avaliação **qualitativa com LLMs**, nos quais um modelo de linguagem instrucionado analisa os outputs do THERA e do modelo base, fornecendo **feedback sobre coerência clínica, completude e relevância dos fatores de risco**.

- **Limpeza e normalização de outputs**:  
  Antes da avaliação, os outputs são processados para extrair **Diagnosis, Description e Risk Factors**, removendo duplicatas e estruturando os fatores de risco em bullet points, garantindo consistência para métricas automáticas e comparação humana.

---

### **3️⃣ Objetivo da Avaliação**

O objetivo desta abordagem híbrida é **quantificar e qualificar** o desempenho do modelo THERA, identificando:

- Melhorias obtidas pelo fine-tuning em relação ao modelo base.  
- Coerência clínica e completude das descrições médicas.  
- Limites do modelo em cenários complexos ou atípicos.  

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

3. Execute os arquivos na ordem:
   - `1_data_preprocessing.ipynb`
   - `2_data_completion_biogpt.ipynb`
   - `3_model_training.py`
   - `4_model_evaluate.ipynb`

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
