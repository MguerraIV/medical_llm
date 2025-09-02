import pandas as pd
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from openai import OpenAI
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ---------------------------
# 🔧 Configurações
# ---------------------------
MODEL_NAME = "microsoft/biogpt"
TRAINED_MODEL_DIR = "biogpt-finetuned-symptom-diagnosis"
MAX_LENGTH = 512
BATCH_SIZE = 4
DEVICE = 0 if torch.cuda.is_available() else -1

# LLM-as-a-Judge
openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Coloque sua chave aqui

# ---------------------------
# 📥 Carregamento e preparação do dataset
# ---------------------------
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
eval_dataset = dataset.train_test_split(test_size=0.15)["test"]

# ---------------------------
# 🔠 Carregar modelos e tokenizers
# ---------------------------
tokenizer_base = BioGptTokenizer.from_pretrained(MODEL_NAME)
model_base = BioGptForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_trained = BioGptTokenizer.from_pretrained(TRAINED_MODEL_DIR)
model_trained = BioGptForCausalLM.from_pretrained(TRAINED_MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

generator_base = pipeline('text-generation', model=model_base, tokenizer=tokenizer_base, device=DEVICE)
generator_trained = pipeline('text-generation', model=model_trained, tokenizer=tokenizer_trained, device=DEVICE)

# ---------------------------
# 🧪 Inferência em lote
# ---------------------------
inputs = [x["input"] for x in eval_dataset]
y_true = [x["output"] for x in eval_dataset]

results_base, results_trained = [], []

for inp in tqdm.tqdm(inputs, desc="Gerando respostas"):
    res_base = generator_base(inp, max_length=150)[0]["generated_text"]
    res_trained = generator_trained(inp, max_length=150)[0]["generated_text"]
    results_base.append(res_base)
    results_trained.append(res_trained)

# ---------------------------
# 📌 Função de extração de diagnóstico
# ---------------------------
def extrair_diagnostico(text):
    if "Diagnosis:" in text:
        return text.split("Diagnosis:")[1].split(".")[0].strip()
    return text

y_true_diag = [extrair_diagnostico(x) for x in y_true]
y_base_diag = [extrair_diagnostico(x) for x in results_base]
y_trained_diag = [extrair_diagnostico(x) for x in results_trained]

# ---------------------------
# 📊 Métricas clássicas
# ---------------------------
acc_base = accuracy_score(y_true_diag, y_base_diag)
acc_trained = accuracy_score(y_true_diag, y_trained_diag)
f1_base = f1_score(y_true_diag, y_base_diag, average='macro')
f1_trained = f1_score(y_true_diag, y_trained_diag, average='macro')

print("📊 Resultados métricas clássicas:")
print(f"Base - Acurácia: {acc_base:.3f}, F1-score: {f1_base:.3f}")
print(f"Treinado - Acurácia: {acc_trained:.3f}, F1-score: {f1_trained:.3f}")

# ---------------------------
# ⚡ Perplexidade
# ---------------------------
def calcular_perplexidade(model, tokenizer, texts):
    model.eval()
    perplexities = []
    for text in tqdm.tqdm(texts, desc="Calculando perplexidade"):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        input_ids = encodings.input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    return perplexities

perp_base = calcular_perplexidade(model_base, tokenizer_base, [x["input"] + "\n" + x["output"] for x in eval_dataset])
perp_trained = calcular_perplexidade(model_trained, tokenizer_trained, [x["input"] + "\n" + x["output"] for x in eval_dataset])

# ---------------------------
# 💡 LLM-as-a-Judge
# ---------------------------
def judge_responses(case_text, base_resp, trained_resp):
    prompt = f"""
        Você é um especialista clínico. Avalie as respostas abaixo para o caso:
        {case_text}

        Resposta modelo base: {base_resp}
        Resposta modelo treinado: {trained_resp}

        Critérios:
        1. Coerência com os sintomas
        2. Relevância do diagnóstico
        3. Argumentação baseada em literatura médica

        Dê uma pontuação de 0 a 10 para cada resposta, indique qual é melhor e se há alguma alucinação (afirmação médica incorreta).
        Formato de resposta:
        Modelo Base: X
        Modelo Treinado: Y
        Melhor Resposta: Base ou Treinado
        Alucinação Base: Sim/Não
        Alucinação Treinado: Sim/Não
    """
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

judge_results = []
for case, base_resp, trained_resp in tqdm.tqdm(zip(inputs, results_base, results_trained), total=len(inputs), desc="LLM-as-a-Judge"):
    try:
        result = judge_responses(case, base_resp, trained_resp)
    except Exception as e:
        result = f"Erro: {e}"
    judge_results.append(result)

# ---------------------------
# 💾 Salvar resultados
# ---------------------------
df_results = pd.DataFrame({
    "case": inputs,
    "true_output": y_true,
    "base_model_output": results_base,
    "trained_model_output": results_trained,
    "base_diag": y_base_diag,
    "trained_diag": y_trained_diag,
    "perplexity_base": perp_base,
    "perplexity_trained": perp_trained,
    "judge_evaluation": judge_results
})
df_results.to_csv("./data/evaluation_results_complete.csv", index=False)
print("✅ Avaliação completa salva em 'evaluation_results_complete.csv'")

# ---------------------------
# 📊 Gráficos automáticos
# ---------------------------

# 1️⃣ Métricas clássicas
metrics = ["Acurácia", "F1-score"]
base_scores = [acc_base, f1_base]
trained_scores = [acc_trained, f1_trained]

x = range(len(metrics))
plt.figure(figsize=(8,6))
plt.bar([i-0.15 for i in x], base_scores, width=0.3, label="Modelo Base")
plt.bar([i+0.15 for i in x], trained_scores, width=0.3, label="Modelo Treinado")
plt.xticks(x, metrics)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Comparação de métricas clássicas")
plt.legend()
plt.grid(axis='y')
plt.show()

# 2️⃣ Distribuição de diagnósticos corretos
df_results["base_correct"] = df_results["base_diag"] == df_results["true_output"].apply(extrair_diagnostico)
df_results["trained_correct"] = df_results["trained_diag"] == df_results["true_output"].apply(extrair_diagnostico)

plt.figure(figsize=(8,6))
sns.countplot(data=df_results.melt(value_vars=["base_correct","trained_correct"],
                                   var_name="Modelo", value_name="Correto"),
              x="Modelo", hue="Correto")
plt.title("Distribuição de diagnósticos corretos")
plt.ylabel("Número de casos")
plt.show()

# 3️⃣ Decisão do LLM-as-a-Judge
def extract_best(judge_text):
    if "Melhor Resposta: Base" in judge_text:
        return "Base"
    elif "Melhor Resposta: Treinado" in judge_text:
        return "Treinado"
    else:
        return "Indefinido"

df_results["judge_best"] = df_results["judge_evaluation"].apply(extract_best)

plt.figure(figsize=(6,6))
sns.countplot(data=df_results, x="judge_best", palette="Set2")
plt.title("Decisão do LLM-as-a-Judge")
plt.ylabel("Número de casos")
plt.show()

# 4️⃣ Distribuição de pontuações
def extract_scores(judge_text):
    base_score = re.search(r"Modelo Base:\s*(\d+)", judge_text)
    trained_score = re.search(r"Modelo Treinado:\s*(\d+)", judge_text)
    return int(base_score.group(1)) if base_score else None, int(trained_score.group(1)) if trained_score else None

df_results[["score_base","score_trained"]] = df_results["judge_evaluation"].apply(lambda x: pd.Series(extract_scores(x)))

plt.figure(figsize=(10,5))
sns.histplot(df_results, x="score_base", color="blue", label="Base", kde=True, binwidth=1)
sns.histplot(df_results, x="score_trained", color="orange", label="Treinado", kde=True, binwidth=1)
plt.title("Distribuição de pontuações do LLM-as-a-Judge")
plt.xlabel("Pontuação")
plt.ylabel("Número de casos")
plt.legend()
plt.show()

# 5️⃣ Distribuição de perplexidade
plt.figure(figsize=(10,5))
sns.histplot(df_results, x="perplexity_base", color="blue", label="Base", kde=True)
sns.histplot(df_results, x="perplexity_trained", color="orange", label="Treinado", kde=True)
plt.title("Distribuição de perplexidade por modelo")
plt.xlabel("Perplexidade")
plt.ylabel("Número de casos")
plt.legend()
plt.show()
