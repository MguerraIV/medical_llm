import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    BioGptTokenizer, BioGptForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, pipeline, set_seed
)
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# 🔧 Configurações iniciais
# ---------------------------
MODEL_NAME = "microsoft/biogpt"
MODEL_OUTPUT_DIR = "thera-finetuned-symptom-diagnosis"
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

    input_text = f"Given the following symptoms, provide the most likely diagnosis, description, and risk factors.\n\nSymptoms: {', '.join(sintomas)}."

    output_text = (
        f"Diagnosis: {row['diseases']}\n\n"
        f"Description: {row['diseases_description']}\n\n"
        f"Risk factors: {row['disease_risk_factors']}."
    )

    return {"input": input_text.strip(), "output": output_text.strip()}

caso_diagnostico = df.apply(gerar_pares, axis=1).tolist()
dataset = Dataset.from_list(caso_diagnostico)

# ---------------------------
# 🔀 Divisão treino/validação/teste
# ---------------------------
dataset_split = dataset.train_test_split(test_size=0.15, seed=SEED)
train_dataset = dataset_split["train"]
temp_dataset = dataset_split["test"]

temp_split = temp_dataset.train_test_split(test_size=0.5, seed=SEED)
eval_dataset = temp_split["train"]   # validação
test_dataset = temp_split["test"]    # teste final

print(f"Tamanho treino: {len(train_dataset)} | validação: {len(eval_dataset)} | teste: {len(test_dataset)}")

# ---------------------------
# 🔠 Tokenização com labels só no output
# ---------------------------
print("Carregando tokenizer e modelo base...")
tokenizer = BioGptTokenizer.from_pretrained(MODEL_NAME)
model = BioGptForCausalLM.from_pretrained(MODEL_NAME).to(device)

def tokenize_function(example):
    prompt = example["input"] + "\n\n" + example["output"]

    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    # mascarar input (labels só no output)
    input_ids = tokenizer(example["input"], truncation=True, max_length=MAX_LENGTH)["input_ids"]
    labels = tokenized["input_ids"].copy()
    labels[:len(input_ids)] = [-100] * len(input_ids)
    tokenized["labels"] = labels

    return tokenized

tokenized_train = train_dataset.map(tokenize_function)
tokenized_eval = eval_dataset.map(tokenize_function)
tokenized_test = test_dataset.map(tokenize_function)

# ---------------------------
# ⚙️ Configuração de treinamento
# ---------------------------
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
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
train_result = trainer.train()

# ---------------------------
# 📊 Curvas de treinamento
# ---------------------------
metrics = trainer.state.log_history

train_loss = [m["loss"] for m in metrics if "loss" in m]
eval_loss = [m["eval_loss"] for m in metrics if "eval_loss" in m]
epochs = list(range(1, len(eval_loss)+1))

plt.figure(figsize=(8,6))
plt.plot(range(1, len(train_loss)+1), train_loss, label="Loss de Treino")
plt.plot(epochs, eval_loss, label="Loss de Validação", marker="o")
plt.xlabel("Passos / Épocas")
plt.ylabel("Loss")
plt.title("Curvas de Treinamento e Validação")
plt.legend()
plt.grid(True)
plt.savefig("training_curves.png")
plt.show()

# ---------------------------
# 💾 Salvando modelo
# ---------------------------
print("Salvando modelo em:", MODEL_OUTPUT_DIR)
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

# ---------------------------
# 🧪 Avaliação no teste final
# ---------------------------
print("\n📊 Avaliando no conjunto de teste...")

generator = pipeline(
    'text-generation',
    model=BioGptForCausalLM.from_pretrained(MODEL_OUTPUT_DIR),
    tokenizer=BioGptTokenizer.from_pretrained(MODEL_OUTPUT_DIR),
    device=0 if torch.cuda.is_available() else -1
)

y_true = []
y_pred = []

for ex in test_dataset:
    input_text = ex["input"]
    expected = ex["output"]
    result = generator(input_text, max_length=200, num_return_sequences=1, temperature=0.7)
    generated = result[0]["generated_text"]

    # regra simples: pegar primeira linha com "Diagnosis:"
    diagnosis = None
    for line in generated.splitlines():
        if line.lower().startswith("diagnosis:"):
            diagnosis = line.split(":", 1)[1].strip()
            break

    if diagnosis:
        y_true.append(expected)
        y_pred.append(diagnosis)

# ---------------------------
# 📊 Matriz de confusão
# ---------------------------
if y_true and y_pred:
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12,10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=90)
    plt.title("Matriz de Confusão - Diagnóstico (Teste Final)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
else:
    print("⚠️ Não foi possível extrair diagnósticos suficientes para a matriz de confusão.")
