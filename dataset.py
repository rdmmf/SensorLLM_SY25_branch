import pickle
import json

# Configuration
SAMPLE_SIZE = 100
DATA_IN = "data/train/mhealth_train_data_stage1.pkl"
QA_IN = "data/train/mhealth_train_qa_stage1.json"
DATA_OUT = "data/train/mhealth_subset_data.pkl"
QA_OUT = "data/train/mhealth_subset_qa.json"



# 1. Charger les fichiers
with open(QA_IN, 'r') as f:
    qa_full = json.load(f)
with open(DATA_IN, 'rb') as f:
    data_full = pickle.load(f)

# On vérifie si l'original a déjà la clé 'dataset', sinon on prend l'objet racine
qa_content = qa_full["dataset"] if "dataset" in qa_full else qa_full

# 2. Création des nouveaux ensembles
new_qa_list = []
new_data = []

# Extraction et synchronisation
keys = list(qa_content.keys()) if isinstance(qa_content, dict) else range(len(qa_content))
count = 0

for k in keys:
    if count >= SAMPLE_SIZE:
        break
    try:
        # Extraction du signal
        signal = data_full[k] if isinstance(data_full, dict) else data_full[int(k)]
        
        # Extraction et mise à jour de l'item QA
        item = qa_content[k] if isinstance(qa_content, dict) else qa_content[int(k)]
        
        # SensorLLM attend souvent une liste dans 'dataset'
        new_qa_list.append(item)
        new_data.append(signal)
        count += 1
    except Exception:
        continue

# 3. Sauvegarde avec la clé racine 'dataset'
final_qa_structure = {"dataset": new_qa_list}

with open(DATA_OUT, 'wb') as f:
    pickle.dump(new_data, f)
with open(QA_OUT, 'w') as f:
    json.dump(final_qa_structure, f)

print(f"Succès ! {len(new_data)} exemples encapsulés dans la clé 'dataset'.")