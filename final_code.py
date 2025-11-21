import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "D:\gowtham\Coding Python\DA5401\Da5401_Data Challenge"

with open(f"{DATA_DIR}/train_data.json", encoding='utf-8') as f:
    train_data = json.load(f)
with open(f"{DATA_DIR}/test_data.json", encoding='utf-8') as f:
    test_data = json.load(f)

def extract_text(record):
    parts = []
    if 'metric_name' in record:
        parts.append(str(record['metric_name']))
    for key in ['prompt', 'input', 'query', 'question']:
        if key in record:
            parts.append(str(record[key]))
            break
    for key in ['system_prompt', 'instruction', 'system']:
        if key in record:
            parts.append(str(record[key]))
            break
    for key in ['expected_response', 'response', 'answer']:
        if key in record:
            parts.append(str(record[key]))
            break
    return " ".join(parts)

train_texts = [extract_text(r) for r in train_data]
train_y = np.array([r.get('score', r.get('target', 0)) for r in train_data])

test_texts = [extract_text(r) for r in test_data]

pseudo_df = pd.read_csv("D:\gowtham\Coding Python\DA5401\Da5401_Data Challenge\sample_submission.csv")
pseudo_y = pseudo_df['score'].values

confident = (pseudo_y <= 2) | (pseudo_y >= 8)
conf_idx = np.where(confident)[0]

augment_texts = [test_texts[i] for i in conf_idx]
augment_y = pseudo_y[conf_idx]

combined_texts = train_texts + augment_texts
combined_y = np.concatenate([train_y, augment_y])

print(f"Original: {len(train_texts)}, Augmented: {len(augment_texts)}")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

combined_emb = embedder.encode(combined_texts, batch_size=64, show_progress_bar=False)
test_emb = embedder.encode(test_texts, batch_size=64, show_progress_bar=False)

X_train, X_val, y_train, y_val = train_test_split(combined_emb, combined_y, test_size=0.2, random_state=7)

class RegNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 768)
        self.bn1 = nn.BatchNorm1d(768)
        self.fc2 = nn.Linear(768, 384)
        self.bn2 = nn.BatchNorm1d(384)
        self.fc3 = nn.Linear(384, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
   
    def forward(self, x):
        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = self.relu(self.fc3(x))
        return self.fc4(x).squeeze()

def train_model(X_tr, y_tr, X_v, y_v, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
   
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_v_t = torch.FloatTensor(X_v).to(device)
   
    net = RegNet(X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0008, weight_decay=0.001)
   
    best_score = 999
    no_improve = 0
   
    for epoch in range(60):
        net.train()
        perm = torch.randperm(len(X_tr_t))
       
        for i in range(0, len(X_tr_t), 96):
            batch = perm[i:i+96]
            out = net(X_tr_t[batch])
            loss = nn.MSELoss()(out, y_tr_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
        net.eval()
        with torch.no_grad():
            val_out = net(X_v_t).cpu().numpy()
            val_rmse = sqrt(mean_squared_error(y_v, val_out))
       
        if val_rmse < best_score:
            best_score = val_rmse
            no_improve = 0
            best_state = net.state_dict().copy()
        else:
            no_improve += 1
            if no_improve >= 8:
                break
   
    net.load_state_dict(best_state)
    return net, best_score

seeds = [17, 88, 201]
models = []
scores = []

for s in seeds:
    print(f"Training seed {s}...")
    m, sc = train_model(X_train, y_train, X_val, y_val, s)
    models.append(m)
    scores.append(sc)
    print(f"  Val: {sc:.4f}")

X_test_t = torch.FloatTensor(test_emb).to(device)

predictions = []
for m in models:
    m.eval()
    with torch.no_grad():
        pred = m(X_test_t).cpu().numpy()
        predictions.append(pred)

ensemble_pred = np.mean(predictions, axis=0)

for val in [0, 3, 6, 9]:
    margin = 0.5
    mask = (ensemble_pred >= val-margin) & (ensemble_pred <= val+margin)
    if mask.sum() > 0:
        noise = np.random.normal(0, 0.2, mask.sum())
        ensemble_pred[mask] += noise

ensemble_pred = np.clip(ensemble_pred, 0, 10)
ensemble_pred = np.round(ensemble_pred, 1)

output = pd.DataFrame({
    "ID": np.arange(1, len(ensemble_pred)+1),
    "score": ensemble_pred
})

output.to_csv("submission_g.csv", index=False)

print(f"\nFinal: mean={ensemble_pred.mean():.2f}, std={ensemble_pred.std():.2f}")
print(f"Low (0-3): {(ensemble_pred <= 3).sum()}")
print(f"Mid (4-7): {((ensemble_pred > 3) & (ensemble_pred <= 7)).sum()}")
print(f"High (8-10): {(ensemble_pred >= 8).sum()}")