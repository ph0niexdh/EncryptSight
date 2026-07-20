# Week 3 — ML Model Training Steps (From Scratch)
## Your Datasets: UNSW-NB15 + CICIDS-2017

---

## YOUR DATASETS EXPLAINED

| File | Dataset | Label Column | Classes |
|---|---|---|---|
| `OneDrive_2026-06-25.zip` | **UNSW-NB15** | `attack_cat` | Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |
| `GeneratedLabelledFlows.zip` | **CICIDS-2017** | `Label` | BENIGN, DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, Bot, Web Attack, Infiltration |

**Plan**: Train on UNSW-NB15 (use pre-split train/test files → fastest path). Add CICIDS-2017 to expand coverage.

---

## STEP 1 — INSTALL LIBRARIES

Open terminal/command prompt in your project folder and run:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn joblib
```

Verify:
```python
import lightgbm; print(lightgbm.__version__)   # should print 3.x or 4.x
```

---

## STEP 2 — SET UP FOLDER STRUCTURE

Create this layout manually or run in Python:

```
ETF/
├── data/
│   ├── unsw/          ← extract UNSW-NB15 CSVs here
│   ├── cicids/        ← extract CICIDS-2017 CSVs here
│   └── models/        ← trained model saved here
├── reports/           ← charts and plots saved here
└── train_model.py     ← your main script
```

```python
import os
for folder in ['data/unsw', 'data/cicids', 'data/models', 'reports']:
    os.makedirs(folder, exist_ok=True)
print("Folders ready.")
```

---

## STEP 3 — EXTRACT YOUR ZIP FILES

### From `OneDrive_2026-06-25.zip`:
Extract and place these 2 files inside `data/unsw/`:
- `CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv`
- `CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv`

> Skip the 4 large raw CSVs (UNSW-NB15_1 to 4). The training/testing sets are enough.

### From `GeneratedLabelledFlows.zip`:
Extract all 8 CSV files inside `data/cicids/`

---

## STEP 4 — LOAD UNSW-NB15 (PRIMARY DATASET)

```python
import pandas as pd
import numpy as np

# Load pre-split train and test
train_df = pd.read_csv('data/unsw/UNSW_NB15_training-set.csv')
test_df  = pd.read_csv('data/unsw/UNSW_NB15_testing-set.csv')

print("Train shape:", train_df.shape)
print("Test shape: ", test_df.shape)
print("\nLabel distribution (train):")
print(train_df['attack_cat'].value_counts())
```

Expected output — 10 classes: Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

---

## STEP 5 — LOAD CICIDS-2017 (SECONDARY DATASET)

```python
import glob

cicids_files = glob.glob('data/cicids/*.csv')
dfs = []
for f in cicids_files:
    df = pd.read_csv(f, encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()   # removes leading spaces from column names
    dfs.append(df)

cicids_df = pd.concat(dfs, ignore_index=True)
print("CICIDS shape:", cicids_df.shape)
print("\nLabel distribution:")
print(cicids_df['Label'].value_counts())
```

---

## STEP 6 — CLEAN THE DATA

### UNSW-NB15 Cleaning:

```python
# Check for missing and infinite values
print("Missing:", train_df.isnull().sum().sum())
print("Inf:", np.isinf(train_df.select_dtypes(include=np.number)).sum().sum())

# Drop identifier columns — not features
drop_cols = ['id', 'srcip', 'dstip', 'sport', 'dsport', 'label']
# Note: keep 'attack_cat' as your target label

train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], inplace=True)
test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], inplace=True)

# Drop rows with missing values
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

print("After cleaning — Train:", train_df.shape, "| Test:", test_df.shape)
```

### CICIDS-2017 Cleaning:

```python
# Replace inf values
cicids_df.replace([np.inf, -np.inf], np.nan, inplace=True)
cicids_df.dropna(inplace=True)

# Drop identifier columns
drop_cicids = ['Flow ID', 'Source IP', 'Destination IP', 
               'Source Port', 'Destination Port', 'Timestamp']
cicids_df.drop(columns=[c for c in drop_cicids if c in cicids_df.columns], inplace=True)

print("After cleaning — CICIDS:", cicids_df.shape)
```

---

## STEP 7 — SELECT FEATURES

### For UNSW-NB15:
These are your ETF fingerprinting features (matching background study):

```python
# UNSW-NB15 has 49 columns total
# Exclude: id, src/dst IPs, ports, attack_cat (label), label (binary)

UNSW_FEATURES = [
    'dur',       # Flow Duration
    'spkts',     # Source packet count
    'dpkts',     # Destination packet count
    'sbytes',    # Source bytes (payload size)
    'dbytes',    # Destination bytes
    'rate',      # Bandwidth
    'sttl',      # Source TTL
    'dttl',      # Destination TTL
    'sload',     # Source bits/sec (bandwidth)
    'dload',     # Destination bits/sec
    'sloss',     # Source packet loss
    'dloss',
    'sinpkt',    # Inter-arrival time source (ms) ← key fingerprint feature
    'dinpkt',    # Inter-arrival time destination (ms)
    'sjit',      # Source jitter
    'djit',      # Destination jitter
    'swin',      # Source TCP window size
    'dwin',      # Destination TCP window size
    'tcprtt',    # TCP round-trip time (latency)
    'synack',    # SYN→ACK time
    'ackdat',    # ACK→DATA time
    'smean',     # Mean packet size source
    'dmean',     # Mean packet size destination
    'ct_srv_src',
    'ct_state_ttl',
    'ct_dst_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
    'ct_src_ltm',
    'ct_srv_dst',
]

# Categorical columns needing encoding
CAT_COLS = ['proto', 'service', 'state']

X_train = train_df[UNSW_FEATURES + CAT_COLS].copy()
y_train = train_df['attack_cat'].copy()

X_test = test_df[UNSW_FEATURES + CAT_COLS].copy()
y_test = test_df['attack_cat'].copy()

print("Features:", X_train.shape[1])
print("Train samples:", len(X_train))
```

---

## STEP 8 — ENCODE CATEGORICAL COLUMNS & LABELS

```python
from sklearn.preprocessing import LabelEncoder

# 8a. Encode proto, service, state columns
for col in CAT_COLS:
    le_col = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le_col.fit(combined)
    X_train[col] = le_col.transform(X_train[col])
    X_test[col]  = le_col.transform(X_test[col])

# 8b. Encode target labels (attack_cat → numbers)
le_label = LabelEncoder()
y_train_enc = le_label.fit_transform(y_train)
y_test_enc  = le_label.transform(y_test)

print("Classes:", list(le_label.classes_))
print("Encoded as:", list(range(len(le_label.classes_))))

# Save the label encoder for later use
import joblib
joblib.dump(le_label, 'data/models/label_encoder.pkl')
```

---

## STEP 9 — CHECK CLASS IMBALANCE

```python
import matplotlib.pyplot as plt

counts = y_train.value_counts()
print("Class distribution:\n", counts)
print("\nImbalance ratio (max/min):", round(counts.max() / counts.min(), 1))

# Visualize
plt.figure(figsize=(10, 4))
counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Malware Family Distribution — UNSW-NB15 Training Set')
plt.xlabel('Attack Category')
plt.ylabel('Sample Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/family_distribution.png', dpi=150)
plt.show()
print("Saved → reports/family_distribution.png")
```

---

## STEP 10 — TRAIN LightGBM FROM SCRATCH

```python
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
import time

# 10a. Handle class imbalance with sample weights (no oversampling needed)
sample_weights = compute_sample_weight('balanced', y_train_enc)

# 10b. Create LightGBM Dataset objects
dtrain = lgb.Dataset(X_train, label=y_train_enc, weight=sample_weights)
dval   = lgb.Dataset(X_test,  label=y_test_enc,  reference=dtrain)

# 10c. Hyperparameters — tuned lightweight for edge deployment
params = {
    'objective':    'multiclass',
    'num_class':    len(le_label.classes_),   # 10 classes for UNSW-NB15
    'metric':       ['multi_logloss', 'multi_error'],
    
    # Lightweight (small model, fast inference)
    'num_leaves':   31,       # keep low = smaller model
    'max_depth':    6,        # limits tree depth
    'min_child_samples': 20,
    
    # Training settings
    'learning_rate':    0.05,
    'feature_fraction': 0.8,  # 80% features per tree
    'bagging_fraction': 0.8,  # 80% rows per tree
    'bagging_freq':     5,
    
    # System
    'verbose':      -1,
    'seed':         42,
    'num_threads':  4,
}

# 10d. Train with early stopping
print("Training LightGBM model...")
start_time = time.time()

callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=True),
    lgb.log_evaluation(period=25),
]

model = lgb.train(
    params,
    dtrain,
    num_boost_round=300,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'val'],
    callbacks=callbacks,
)

elapsed = time.time() - start_time
print(f"\nTraining done in {elapsed:.1f} seconds")
print(f"Best iteration: {model.best_iteration}")
```

---

## STEP 11 — EVALUATE THE MODEL

```python
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, f1_score)
import seaborn as sns

# 11a. Predict
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred_proba, axis=1)

# 11b. Measure inference latency (lightweight test)
start = time.time()
for _ in range(1000):
    model.predict(X_test.iloc[:1], num_iteration=model.best_iteration)
latency_ms = (time.time() - start)
print(f"Avg inference latency: {latency_ms:.3f} ms per flow")

# 11c. Metrics
accuracy = accuracy_score(y_test_enc, y_pred)
f1_macro = f1_score(y_test_enc, y_pred, average='macro')
f1_weighted = f1_score(y_test_enc, y_pred, average='weighted')

print(f"\nAccuracy:      {accuracy*100:.2f}%")
print(f"F1 (macro):    {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")

# 11d. Full classification report
print("\n=== Per-Class Report ===")
print(classification_report(y_test_enc, y_pred, target_names=le_label.classes_))

# 11e. Confusion matrix
fig, ax = plt.subplots(figsize=(12, 9))
cm = confusion_matrix(y_test_enc, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_label.classes_,
            yticklabels=le_label.classes_, ax=ax)
ax.set_title('Confusion Matrix — ETF LightGBM (UNSW-NB15)')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150)
plt.show()
print("Saved → reports/confusion_matrix.png")
```

---

## STEP 12 — FEATURE IMPORTANCE (Fingerprint Analysis)

```python
# Which features matter most for identifying malware families?

importance_df = pd.DataFrame({
    'feature':    model.feature_name(),
    'gain':       model.feature_importance(importance_type='gain'),
    'split':      model.feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)

print("Top 15 fingerprinting features:")
print(importance_df.head(15).to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
top15 = importance_df.head(15)
ax.barh(top15['feature'][::-1], top15['gain'][::-1], color='darkorange')
ax.set_title('Top 15 Traffic Fingerprinting Features (by Information Gain)')
ax.set_xlabel('Gain')
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=150)
plt.show()
print("Saved → reports/feature_importance.png")
```

---

## STEP 13 — VISUALIZE TRAFFIC BEHAVIOR

```python
# Build plot dataframe
plot_df = X_train[['sinpkt', 'dinpkt', 'sload', 'smean', 'dur']].copy()
plot_df['family'] = y_train.values

# Cap outliers at 95th percentile for readability
for col in ['sinpkt', 'sload', 'smean', 'dur']:
    cap = plot_df[col].quantile(0.95)
    plot_df[col] = plot_df[col].clip(upper=cap)

# Sample 300 per class
sample = plot_df.groupby('family').apply(
    lambda g: g.sample(min(300, len(g)), random_state=42)
).reset_index(drop=True)

# Chart 1: IAT distribution (beacon pattern)
fig, ax = plt.subplots(figsize=(12, 5))
for fam in sample['family'].unique():
    data = sample[sample['family'] == fam]['sinpkt']
    ax.hist(data, bins=40, alpha=0.5, label=fam, density=True)
ax.set_title('Source Inter-Arrival Time by Malware Family\n(sinpkt — C2 Beacon Pattern)')
ax.set_xlabel('Inter-Arrival Time (ms)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('reports/iat_distribution.png', dpi=150)
plt.show()

# Chart 2: Duration vs Bandwidth
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10.colors
families = sample['family'].unique()
for i, fam in enumerate(families):
    sub = sample[sample['family'] == fam]
    ax.scatter(sub['dur'], sub['sload'], alpha=0.4, s=15,
               label=fam, color=colors[i % len(colors)])
ax.set_title('Flow Duration vs Source Bandwidth per Family')
ax.set_xlabel('Flow Duration (s)')
ax.set_ylabel('Source Load (bits/sec)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('reports/duration_vs_bandwidth.png', dpi=150)
plt.show()

# Chart 3: Payload size violin plot
fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(data=sample, x='family', y='smean', palette='Set2', ax=ax)
ax.set_title('Mean Payload Size Distribution per Malware Family')
ax.set_xlabel('Family')
ax.set_ylabel('Mean Source Packet Size (bytes)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/payload_violin.png', dpi=150)
plt.show()

print("All 3 charts saved to reports/")
```

---

## STEP 14 — SAVE THE MODEL

```python
import os

# Save LightGBM model (lightweight text format)
model.save_model('data/models/etf_lgbm_model.txt')

model_size_kb = os.path.getsize('data/models/etf_lgbm_model.txt') / 1024
print(f"Model saved. Size: {model_size_kb:.1f} KB")   # Expect < 500 KB

# Save label encoder
joblib.dump(le_label, 'data/models/label_encoder.pkl')

# Save feature list (needed for inference later)
import json
with open('data/models/feature_list.json', 'w') as f:
    json.dump(UNSW_FEATURES + CAT_COLS, f)

print("All model artifacts saved to data/models/")
```

---

## STEP 15 — TEST INFERENCE (Single Flow Prediction)

```python
# Simulate predicting on 1 new network flow

sample_flow = X_test.iloc[[0]]   # take one test row as example

pred_proba = model.predict(sample_flow, num_iteration=model.best_iteration)
pred_class = np.argmax(pred_proba, axis=1)[0]
pred_label = le_label.inverse_transform([pred_class])[0]
confidence = pred_proba[0][pred_class] * 100

print(f"Predicted Family: {pred_label}")
print(f"Confidence:       {confidence:.1f}%")
print(f"True Label:       {le_label.inverse_transform([y_test_enc[0]])[0]}")
```

---

## COMPLETE RUN ORDER

Run steps in this order, no skipping:

```
Step 1  → Install libraries
Step 2  → Create folders
Step 3  → Extract ZIPs into correct folders
Step 4  → Load UNSW-NB15
Step 5  → Load CICIDS-2017 (optional, for later)
Step 6  → Clean data
Step 7  → Select features
Step 8  → Encode categorical + labels
Step 9  → Check class distribution → save chart
Step 10 → Train LightGBM ← core step
Step 11 → Evaluate → save confusion matrix
Step 12 → Feature importance → save chart
Step 13 → Visualize traffic → save 3 charts
Step 14 → Save model
Step 15 → Test single prediction
```

---

## EXPECTED RESULTS (UNSW-NB15)

| Metric | Expected |
|---|---|
| Accuracy | 96 – 99% |
| F1 (macro) | 0.90 – 0.96 |
| Inference latency | < 1 ms/flow |
| Model file size | < 1 MB |
| Training time | 30 – 120 seconds |

---

## TROUBLESHOOTING

| Error | Fix |
|---|---|
| `KeyError: 'attack_cat'` | Column might be `'Attack_cat'` — check with `df.columns` |
| `ValueError: could not convert string to float` | A feature column is categorical — add it to `CAT_COLS` and encode it |
| `LightGBMError: num_class should be set` | Check `params['num_class']` matches actual number of unique labels |
| Memory error on large files | Load only train/test CSVs, skip the raw UNSW CSVs 1–4 |
| `inf` values causing NaN | Run `df.replace([np.inf, -np.inf], np.nan, inplace=True)` before training |

---

*ETF Project — Week 3 | UNSW-NB15 + CICIDS-2017 | LightGBM*
