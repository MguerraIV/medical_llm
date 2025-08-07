import os
import pandas as pd
import torch
from transformers import (
    BioGptTokenizer, BioGptForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, pipeline, set_seed
)
from datasets import Dataset

# ---------------------------
# 🔧 Configurações iniciais
# ---------------------------
MODEL_NAME = "microsoft/biogpt"
MODEL_OUTPUT_DIR = "biogpt-finetuned-symptom-diagnosis"
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 6  
EPOCHS = 5      
LEARNING_RATE = 5e-5

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ---------------------------
# 📥 Carregamento e preparação do dataset
# ---------------------------
print("Lendo dataset...")
df = pd.read_csv("./data/merged_dataset.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

COLUNAS = df.columns

def gerar_pares(row):
    sintomas = [col.replace("_", " ") for col in COLUNAS if row[col] == 1]
    input_text = f"The pacient presents the following symptoms: {', '.join(sintomas)}."
    output_text = f'''
        Diagnosis: {row['diseases']}.

        Description: {row['diseases_description']}.

        Risk factors: {row['disease_risk_factors']}.
    '''
    return {"input": input_text.strip(), "output": output_text.strip()}

caso_diagnostico = df.apply(gerar_pares, axis=1).tolist()
dataset = Dataset.from_list(caso_diagnostico)

# ---------------------------
# 🔀 Divisão treino/validação
# ---------------------------
dataset = dataset.train_test_split(test_size=0.15)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ---------------------------
# 🔠 Tokenização
# ---------------------------
print("Carregando tokenizer e modelo base...")
tokenizer = BioGptTokenizer.from_pretrained(MODEL_NAME)
model = BioGptForCausalLM.from_pretrained(MODEL_NAME).to(device)

def tokenize_function(example):
    prompt = example["input"] + "\n" + example["output"]
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_train = train_dataset.map(tokenize_function)
tokenized_eval = eval_dataset.map(tokenize_function)

# ---------------------------
# ⚙️ Configuração de treinamento
# ---------------------------
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),  # usa float16 se possível
    gradient_accumulation_steps=2,   # simula batch_size de 12
    load_best_model_at_end=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ---------------------------
# 🚀 Treinamento
# ---------------------------
print("Iniciando treinamento...")
trainer.train()

# ---------------------------
# 💾 Salvando modelo
# ---------------------------
print("Salvando modelo em:", MODEL_OUTPUT_DIR)
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

# ---------------------------
# 🧪 Teste de inferência
# ---------------------------
print("Testando inferência com modelo fine-tuned...")
generator = pipeline(
    'text-generation',
    model=BioGptForCausalLM.from_pretrained(MODEL_OUTPUT_DIR),
    tokenizer=BioGptTokenizer.from_pretrained(MODEL_OUTPUT_DIR),
    device=0 if torch.cuda.is_available() else -1
)

test_input = "The pacient presents the following symptoms: fever, cough, fatigue."
result = generator(test_input, max_length=100)
print("\n📋 Geração de exemplo:")
print(result[0]["generated_text"])
