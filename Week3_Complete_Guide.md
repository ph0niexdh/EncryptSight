# Week 3 — Complete Step-by-Step Guide
## Encrypted Traffic Fingerprinting for Malware Family Detection
### LightGBM Model | From Scratch

---

## OVERVIEW

Your pipeline for Week 3:
```
Dataset Download → PCAP/CSV Processing → Feature Extraction → 
EDA & Visualization → Model Training (LightGBM) → Evaluation → Deliverables
```

---

## STEP 1: DATASET COLLECTION (Downloadable as ZIP)

Use the datasets below. They are pre-labelled with malware family tags — critical since you're training from scratch.

### ✅ Primary Dataset — CICIDS-2017 (RECOMMENDED — Start Here)
- **URL**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Direct download**: Files are on Google Drive linked on that page (~8 GB total, CSVs available separately ~250 MB)
- **Format**: Pre-extracted CSV files with 80 features + label column
- **Labels**: BENIGN, DoS, DDoS, Botnet ARES, Infiltration, Web Attack
- **Why use it**: Already flow-aggregated, no PCAP processing needed — plug straight into Python
- **Tip**: Download only the `MachineLearningCVE` folder (CSV files only, ~250 MB) to avoid the full PCAP download

### ✅ Secondary Dataset — UNSW-NB15
- **URL**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Direct download**: https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys
- **Format**: CSV (2 million+ records, 49 features)
- **Labels**: 9 attack categories — Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms + Normal
- **Size**: ~100 MB for CSVs

### ✅ CTU-13 — Botnet-Specific
- **URL**: https://www.stratosphereips.org/datasets-ctu13
- **Direct download**: Each scenario is a separate ZIP (~50–200 MB each)
- **Format**: PCAP + pre-extracted NetFlow CSV (use the Binetflow CSV files)
- **Labels**: Botnet, Normal, Background
- **Recommended**: Download Scenarios 1, 2, 9, 10 only (cover different botnet families)
- **Why**: Real captured botnet traffic, great for C2 beacon pattern features

### ✅ CIC-MalMem-2022 (Best for Malware Family Classification)
- **URL**: https://www.unb.ca/cic/datasets/malmem-2022.html
- **Format**: CSV (pre-extracted features)
- **Labels**: Ransomware, Trojan, Spyware, Benign — exactly what your project needs
- **Size**: ~50 MB

### ✅ BODMAS Dataset (Real Malware Families, Newest)
- **URL**: https://whyisyoung.github.io/BODMAS/
- **Format**: CSV with malware family labels
- **Labels**: 581 malware families grouped into types (ransomware, RAT, banker, etc.)
- **Download**: Linked directly from the GitHub page

### ⚠️ What to Skip for Now
- MALWARE-TRAFFIC-ANALYSIS.NET — PCAPs only, requires manual labelling
- Full CICIDS PCAP files — too large (8 GB+), CSVs already extracted

---

## STEP 2: ENVIRONMENT SETUP

```bash
# Create virtual environment
python -m venv etf_env
source etf_env/bin/activate  # Windows: etf_env\Scripts\activate

# Install all required libraries
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn \
            plotly dpkt scapy pyshark tqdm joblib jupyter
```

**Folder structure to create:**
```
ETF/
├── data/
│   ├── raw/           # Downloaded CSVs go here
│   ├── processed/     # Cleaned + feature-extracted CSVs
│   └── models/        # Saved LightGBM model files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── feature_extractor.py
│   ├── train.py
│   └── evaluate.py
└── reports/           # Deliverable outputs
```

---

## STEP 3: DATA LOADING & INITIAL INSPECTION

```python
# Step 3: Load and inspect CICIDS-2017
import pandas as pd
import numpy as np

# Load all day CSVs and merge
import glob

csv_files = glob.glob('data/raw/CICIDS2017/*.csv')
dfs = []
for f in csv_files:
    df = pd.read_csv(f, encoding='latin-1', low_memory=False)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Clean column names (CICIDS has leading spaces)
data.columns = data.columns.str.strip()

print("Shape:", data.shape)
print("Labels:", data['Label'].value_counts())
print("Missing values:\n", data.isnull().sum().sum())
print("Inf values:", np.isinf(data.select_dtypes(include=np.number)).sum().sum())
```

---

## STEP 4: DATA CLEANING

```python
# Step 4: Clean data

# 4a. Drop duplicates
data.drop_duplicates(inplace=True)

# 4b. Replace inf with NaN, then drop rows with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 4c. Drop non-feature columns
cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 
                'Source Port', 'Destination Port', 'Timestamp']
data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True)

# 4d. Map labels to malware families
# CICIDS uses attack names — group into your 5 families
label_map = {
    'BENIGN': 'Benign',
    'Bot': 'Botnet',
    'DDoS': 'Botnet',
    'DoS GoldenEye': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS slowloris': 'DoS',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Heartbleed': 'Exploit',
    'Infiltration': 'RAT',
    'PortScan': 'Reconnaissance',
    'Web Attack – Brute Force': 'Brute Force',
    'Web Attack – Sql Injection': 'Exploit',
    'Web Attack – XSS': 'Exploit',
}
data['family'] = data['Label'].map(label_map).fillna('Other')
data.drop(columns=['Label'], inplace=True)

print("Family distribution:\n", data['family'].value_counts())
```

---

## STEP 5: FEATURE EXTRACTION (The Core of ETF)

These are your fingerprinting features as defined in the background study.

```python
# Step 5: Feature extraction — these ARE the traffic fingerprints
# CICIDS already has these computed. For raw PCAPs, see Step 5b below.

# Core features aligned with your background study
ETF_FEATURES = [
    # Payload Size features
    'Flow Bytes/s',             # Bandwidth proxy
    'Flow Packets/s',
    'Fwd Packet Length Max',    # Max payload size forward
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',   # Mean payload size
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    
    # Flow Duration
    'Flow Duration',            # Short vs persistent C2 sessions
    
    # Packet Count
    'Total Fwd Packets',
    'Total Backward Packets',
    
    # Inter-Arrival Time (IAT)
    'Flow IAT Mean',            # Beacon interval proxy
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    
    # Byte Ratio (Upload/Download)
    'Fwd Header Length',
    'Bwd Header Length',
    
    # TCP Flags (behavioral fingerprint)
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    
    # Window sizes (TLS behavior proxy)
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    
    # Subflow stats
    'Subflow Fwd Bytes',
    'Subflow Bwd Bytes',
]

# Filter to only columns that exist in your loaded dataset
available_features = [f for f in ETF_FEATURES if f in data.columns]
print(f"Using {len(available_features)} features out of {len(ETF_FEATURES)} defined")

X = data[available_features]
y = data['family']
```

### Step 5b: Feature Extraction from Raw PCAP (if using CTU-13 or MTA.net)

```python
# Only needed if working with raw .pcap files
# Install: pip install dpkt
import dpkt, socket
from collections import defaultdict

def extract_flow_features(pcap_path):
    flows = defaultdict(list)
    
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    continue
                
                transport = ip.data
                # 5-tuple flow key
                flow_key = (
                    socket.inet_ntoa(ip.src),
                    socket.inet_ntoa(ip.dst),
                    transport.sport,
                    transport.dport,
                    ip.p
                )
                flows[flow_key].append({
                    'timestamp': ts,
                    'payload_len': len(transport.data),
                    'total_len': ip.len
                })
            except:
                continue
    
    # Aggregate per-flow statistics
    records = []
    for flow_key, packets in flows.items():
        if len(packets) < 2:
            continue
        
        timestamps = [p['timestamp'] for p in packets]
        payloads = [p['payload_len'] for p in packets]
        iats = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        records.append({
            'src_ip': flow_key[0],
            'dst_ip': flow_key[1],
            'src_port': flow_key[2],
            'dst_port': flow_key[3],
            'protocol': flow_key[4],
            'flow_duration': timestamps[-1] - timestamps[0],
            'packet_count': len(packets),
            'payload_mean': np.mean(payloads),
            'payload_std': np.std(payloads),
            'payload_max': np.max(payloads),
            'payload_min': np.min(payloads),
            'iat_mean': np.mean(iats) if iats else 0,
            'iat_std': np.std(iats) if iats else 0,
            'iat_max': np.max(iats) if iats else 0,
            'total_bytes': sum(payloads),
            'bandwidth_bps': sum(payloads) / max(timestamps[-1] - timestamps[0], 1e-6),
        })
    
    return pd.DataFrame(records)
```

---

## STEP 6: MALWARE FAMILY DISTRIBUTION ANALYSIS

```python
# Step 6: Study malware family distribution
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6a. Count plot
family_counts = y.value_counts()
axes[0].bar(family_counts.index, family_counts.values, color='steelblue', edgecolor='black')
axes[0].set_title('Malware Family Distribution')
axes[0].set_xlabel('Family')
axes[0].set_ylabel('Sample Count')
axes[0].tick_params(axis='x', rotation=45)

# 6b. Pie chart
axes[1].pie(family_counts.values, labels=family_counts.index, autopct='%1.1f%%', startangle=140)
axes[1].set_title('Family Proportions')

plt.tight_layout()
plt.savefig('reports/family_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 6c. Check class imbalance ratio
print("Imbalance ratio (max/min):", family_counts.max() / family_counts.min())
# If > 10, you need class balancing (handled in Step 8)
```

---

## STEP 7: IDENTIFY IMPORTANT FINGERPRINTING FEATURES (EDA)

```python
# Step 7: Feature importance analysis before training

# 7a. Correlation heatmap of key ETF features
fig, ax = plt.subplots(figsize=(12, 10))
corr = X[available_features[:15]].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, 
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('reports/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 7b. Box plots of top features per malware family
key_features = ['Flow Duration', 'Flow IAT Mean', 'Flow Bytes/s', 
                'Fwd Packet Length Mean', 'Total Fwd Packets']

fig, axes = plt.subplots(1, len(key_features), figsize=(20, 5))
for i, feat in enumerate(key_features):
    if feat in X.columns:
        plot_data = X[[feat]].copy()
        plot_data['family'] = y.values
        # Cap outliers at 99th percentile for readability
        cap = plot_data[feat].quantile(0.99)
        plot_data[feat] = plot_data[feat].clip(upper=cap)
        
        sns.boxplot(data=plot_data, x='family', y=feat, ax=axes[i])
        axes[i].set_title(feat)
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/feature_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## STEP 8: DATA PREPROCESSING FOR ML

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 8a. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 8b. Train/val/test split — 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
# 0.176 of 0.85 = ~0.15 of total

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 8c. Handle class imbalance using class_weight (no oversampling needed with LightGBM)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
```

---

## STEP 9: TRAIN LightGBM FROM SCRATCH

```python
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time

# 9a. Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
val_data   = lgb.Dataset(X_val,   label=y_val,   reference=train_data)

# 9b. Hyperparameters — tuned for lightweight deployment
params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': ['multi_logloss', 'multi_error'],
    
    # Lightweight settings (key for edge deployment)
    'num_leaves': 31,           # Keep low → smaller model
    'max_depth': 6,             # Limit tree depth
    'min_child_samples': 20,    # Prevents overfitting on small classes
    'n_estimators': 200,
    
    # Speed & memory
    'learning_rate': 0.05,
    'feature_fraction': 0.8,    # Use 80% of features per tree
    'bagging_fraction': 0.8,    # Row subsampling
    'bagging_freq': 5,
    'histogram_pool_size': 512, # MB — cap memory use
    
    # Output
    'verbose': -1,
    'seed': 42,
    'num_threads': 4,           # Adjust to your CPU
}

# 9c. Train with early stopping
print("Training LightGBM...")
start = time.time()

callbacks = [
    lgb.early_stopping(stopping_rounds=20, verbose=True),
    lgb.log_evaluation(period=20)
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=callbacks,
)

training_time = time.time() - start
print(f"\nTraining complete in {training_time:.1f}s")
print(f"Best iteration: {model.best_iteration}")
```

---

## STEP 10: EVALUATE THE MODEL

```python
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, f1_score)

# 10a. Predict on test set
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred_proba, axis=1)

# 10b. Measure inference latency (lightweight check)
start = time.time()
for _ in range(1000):
    model.predict(X_test.iloc[:1], num_iteration=model.best_iteration)
latency_ms = (time.time() - start) / 1000 * 1000
print(f"Avg inference latency per flow: {latency_ms:.3f} ms")

# 10c. Classification report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 10d. Key metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy:    {accuracy:.4f}")
print(f"F1 (macro):  {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")

# 10e. Confusion matrix visualization
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title('Confusion Matrix — LightGBM ETF Model')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## STEP 11: FEATURE IMPORTANCE (Fingerprint Identification)

```python
# Step 11: Which features matter most for fingerprinting?

# 11a. LightGBM built-in feature importance
importance_df = pd.DataFrame({
    'feature': model.feature_name(),
    'importance_split': model.feature_importance(importance_type='split'),
    'importance_gain': model.feature_importance(importance_type='gain'),
}).sort_values('importance_gain', ascending=False)

print("Top 15 Features by Gain:")
print(importance_df.head(15).to_string(index=False))

# 11b. Plot top 20 features
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

top20 = importance_df.head(20)
axes[0].barh(top20['feature'][::-1], top20['importance_split'][::-1], color='teal')
axes[0].set_title('Feature Importance (Split Count)')
axes[0].set_xlabel('Splits')

axes[1].barh(top20['feature'][::-1], top20['importance_gain'][::-1], color='darkorange')
axes[1].set_title('Feature Importance (Information Gain)')
axes[1].set_xlabel('Gain')

plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## STEP 12: TRAFFIC VISUALIZATION CHARTS

```python
# Step 12: Visualize traffic behavior per malware family

plot_df = X[['Flow Duration', 'Flow IAT Mean', 'Flow Bytes/s', 
             'Fwd Packet Length Mean']].copy()
plot_df['family'] = y.values

# Cap outliers
for col in plot_df.columns[:-1]:
    cap = plot_df[col].quantile(0.95)
    plot_df[col] = plot_df[col].clip(upper=cap)

# Sample for readability
sample = plot_df.groupby('family').apply(
    lambda g: g.sample(min(500, len(g)), random_state=42)
).reset_index(drop=True)

# 12a. IAT distribution per family (beacon detection)
fig, ax = plt.subplots(figsize=(12, 5))
for family in sample['family'].unique():
    subset = sample[sample['family'] == family]['Flow IAT Mean']
    ax.hist(subset, bins=50, alpha=0.5, label=family, density=True)
ax.set_title('Inter-Arrival Time Distribution by Malware Family\n(Reveals C2 Beacon Patterns)')
ax.set_xlabel('Mean IAT (ms)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig('reports/iat_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 12b. Scatter: Flow Duration vs Bandwidth colored by family
fig, ax = plt.subplots(figsize=(10, 7))
palette = {'Benign':'green','Botnet':'red','DoS':'orange',
           'RAT':'purple','Brute Force':'blue','Exploit':'brown'}
for family in sample['family'].unique():
    sub = sample[sample['family'] == family]
    color = palette.get(family, 'gray')
    ax.scatter(sub['Flow Duration'], sub['Flow Bytes/s'],
               alpha=0.4, s=10, label=family, color=color)
ax.set_title('Flow Duration vs Bandwidth — Traffic Clusters')
ax.set_xlabel('Flow Duration (s)')
ax.set_ylabel('Bytes/s (Bandwidth)')
ax.legend()
plt.tight_layout()
plt.savefig('reports/duration_vs_bandwidth.png', dpi=150, bbox_inches='tight')
plt.show()

# 12c. Payload size by family
fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(data=sample, x='family', y='Fwd Packet Length Mean', 
               palette='Set2', ax=ax)
ax.set_title('Payload Size Distribution per Malware Family')
ax.set_xlabel('Family')
ax.set_ylabel('Mean Forward Packet Length (bytes)')
plt.tight_layout()
plt.savefig('reports/payload_violin.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## STEP 13: SAVE MODEL (Lightweight Deployment)

```python
import joblib

# 13a. Save LightGBM model (very compact — typically < 5 MB)
model.save_model('data/models/etf_lightgbm.txt')
print(f"Model size: {os.path.getsize('data/models/etf_lightgbm.txt') / 1024:.1f} KB")

# 13b. Save label encoder
joblib.dump(le, 'data/models/label_encoder.pkl')

# 13c. Save feature list (needed for inference)
import json
with open('data/models/feature_list.json', 'w') as f:
    json.dump(available_features, f)

print("Model artifacts saved. Ready for deployment.")
```

---

## WEEK 3 DELIVERABLES — WHAT TO SUBMIT

| Deliverable | What It Is | Files Generated |
|---|---|---|
| Dataset Report | Which datasets you used, size, label distribution | `reports/family_distribution.png` + written report |
| Traffic Fingerprinting Report | Feature table + why each feature matters | `reports/feature_importance.png` |
| Malware Family Analysis | Distribution analysis + imbalance discussion | `reports/family_distribution.png` |
| Feature Description Table | Table of all 30+ features with definitions | See Step 5 above |
| Traffic Visualization Charts | IAT, bandwidth, payload plots | `reports/iat_distribution.png`, `duration_vs_bandwidth.png`, `payload_violin.png` |
| 5-slide Progress Presentation | Summary of Week 3 work | Use pptx skill |

---

## QUICK REFERENCE — DATASET PRIORITY ORDER

1. **CICIDS-2017 CSV** (start here — already processed, ~250 MB)
2. **UNSW-NB15** (add for more malware variety)
3. **CTU-13 Binetflow CSVs** (add for botnet-specific C2 features)
4. Combine all three → merge on common feature columns → unified dataset

---

## EXPECTED MODEL PERFORMANCE (CICIDS-2017)

| Metric | Expected Range |
|---|---|
| Accuracy | 97–99% |
| F1 (macro) | 0.93–0.97 |
| False Positive Rate | < 2% |
| Inference Latency | < 1 ms/flow |
| Model File Size | < 5 MB |

---

*Week 3 guide — ETF Project | LightGBM from scratch*
