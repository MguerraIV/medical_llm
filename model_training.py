import pandas as pd
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
from datasets import Dataset
from transformers import AutoTokenizer
import torch

#importando dataset unido e padronizado
merged_dataset = pd.read_csv("./data/merged_dataset.csv")
merged_dataset.drop(columns=["Unnamed: 0"], inplace=True)

#função para geração dos pares na base
COLUNAS = merged_dataset.columns
def gerar_pares(row):
    # Geração do input com base nos sintomas marcados como 1
    sintomas = [col.replace("_", " ") for col in COLUNAS if row[col] == 1]
    input_text = f"The pacient presents the following symptoms: {', '.join(sintomas)}."

    # Geração do output com diagnóstico + descrição + fatores de risco
    output_text = f'''
        Diagnosis: {row['diseases']}.\n
        Description: {row['diseases_description']}.\n
        Risk factors: {row['disease_risk_factors']}.
    '''
    
    return {"input": input_text, "output": output_text} #retorno do par gerado

#agora é só ler a base e aplicar a geração dos pares
caso_diagnostico = merged_dataset.apply(gerar_pares, axis=1).tolist()

#código exemplo para a utilização do modelo BioGPT
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt") #instanciando o modelo

#movendo o modelo para a gpu do sistema (Nvidia RTX 3050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt") #instanciando o tokenizer
generator = pipeline('text-generation', model=model, tokenizer=tokenizer) #criando o gerador de texto
set_seed(42) #configurando semente aleatória

#cria dataset Hugging Face com os pares
dataset = Dataset.from_list(caso_diagnostico)
print(dataset) #dataset preparado

#separando treino/validação
dataset = dataset.train_test_split(test_size=0.15)
train_dataset = dataset['train']
eval_dataset = dataset['test']

#como o BioGPT é causal LM (autogerativo), vamos concatenar input + output e treinar o modelo para prever
def tokenize_function(example): #função para tokenizar os dados antes do treinamento
    prompt = example["input"] + "\n" + example["output"]
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

#dados tokenizados
tokenized_train = train_dataset.map(tokenize_function)
tokenized_eval = eval_dataset.map(tokenize_function)

#configurando os dados do treinamento
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./biogpt-finetuned",
    evaluation_strategy="epoch", #estratégia de treinamento por épocas
    learning_rate=5e-5, #taxa de aprendizagem do modelo
    per_device_train_batch_size=3, #tamanho do batch de treino
    per_device_eval_batch_size=3, #tamanho do batch de validacao
    num_train_epochs=3, #numero de epocas de treino
    weight_decay=0.01, #taxa de decaimento dos pesos
    save_total_limit=2,
    logging_dir='./logs',
    fp16=True,
    logging_steps=10,
)

#como é causal LM, usamos esse collato, dado uma lista de exemplos, retorna um batch pronto para o modelo
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False #serve para criar tensores compatíveis para o modelo (inputs, labels, masks, etc.) e
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#agora que tudo já foi preparado, vamos realizar o treinamento do modelo
trainer.train()

#salvando o modelo treinado
trainer.save_model("biogpt-finetuned-symptom-diagnosis")
tokenizer.save_pretrained("biogpt-finetuned-symptom-diagnosis")

#teste de inferência do modelo com fine-tuning
generator = pipeline('text-generation', model="biogpt-finetuned-symptom-diagnosis", tokenizer=tokenizer)
generator("The pacient presents the following symptoms: fever, cough, fatigue.", max_length=100)