import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

files = [r"C:\Users\ahmed\Downloads\balanced_dataset1.csv"]

print("File Loding...")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)


print(df["Label"].unique())


df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"] != "Label"]



df["Label_bin"] = df["Label"].apply(lambda x: 0 if x == "Benign" else 1)

drop_cols = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "SimillarHTTP", "Inbound"]
X = df.drop(columns=drop_cols + ["Label", "Label_bin"], errors="ignore")
y = df["Label_bin"]

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)
X = X.clip(lower=-1e15, upper=1e15)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

print("Before SMOTE:", pd.Series(y_train).value_counts().to_dict())

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE :", pd.Series(y_train_sm).value_counts().to_dict())


print("Lasso full training shape:", X_train_sm.shape)
print(pd.Series(y_train_sm).value_counts())

lasso = LassoCV(cv=3, random_state=42, max_iter=3000)
lasso.fit(X_train_sm, y_train_sm)
selector = SelectFromModel(lasso, prefit=True, max_features=46, threshold=-np.inf)

X_train_sel = selector.transform(X_train_sm)
X_test_sel = selector.transform(X_test.values)

print("Selected full training shape:", X_train_sel.shape)
print(pd.Series(y_train_sm).value_counts())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)


print("Modils are being trained...\n")
from sklearn.svm import LinearSVC

base_svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)

svm_model = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
svm_model.fit(X_train_scaled, y_train_sm)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train_sm)

svm_prob = svm_model.predict_proba(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)


def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 50)
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred, zero_division=0))

    return {
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1
    }


def belief_from_probs(probs):
    p_attack = probs[:, 1]
    p_benign = probs[:, 0]
    uncertainty = 1 - np.abs(p_attack - p_benign)
    m_attack = p_attack * (1 - uncertainty * 0.1)
    m_benign = p_benign * (1 - uncertainty * 0.1)
    m_u = 1 - (m_attack + m_benign)
    return m_benign, m_attack, m_u

def dempster_combination(m1_b, m1_a, m1_u, m2_b, m2_a, m2_u):
    K = (m1_b * m2_a) + (m1_a * m2_b)
    denominator = 1 - K
    denominator = np.where(denominator <= 0, 1e-9, denominator)
    f_b = (m1_b * m2_b + m1_b * m2_u + m1_u * m2_b) / denominator
    f_a = (m1_a * m2_a + m1_a * m2_u + m1_u * m2_a) / denominator
    f_u = (m1_u * m2_u) / denominator
    total = f_b + f_a + f_u
    return f_b/total, f_a/total, f_u/total

svm_mb, svm_ma, svm_mu = belief_from_probs(svm_prob)
rf_mb, rf_ma, rf_mu = belief_from_probs(rf_prob)

dst_b, dst_a, dst_u = dempster_combination(svm_mb, svm_ma, svm_mu, rf_mb, rf_ma, rf_mu)
dst_pred_base = (dst_a > dst_b).astype(int)

dst_pred = np.where(rf_prob[:, 1] >= 0.80, 1, 
           np.where(rf_prob[:, 1] <= 0.20, 0, dst_pred_base)).astype(int)


svm_results = evaluate_model("SVM (Individual)", y_test, svm_model.predict(X_test_scaled))
rf_results = evaluate_model("Random Forest (Individual)", y_test, rf_model.predict(X_test_scaled))
dst_results = evaluate_model("SVM + RF + DST (Hybrid System)", y_test, dst_pred)

print("\n--- Summary Table ---")
results_summary = pd.DataFrame([svm_results, rf_results, dst_results])
print(results_summary[["Model", "Accuracy", "F1-score", "Precision","Recall"]])

print("\n--- Fusion Debug (Top 10) ---")
fusion_debug = pd.DataFrame({
    "svm_attack": svm_ma[:10], "rf_attack": rf_ma[:10], "dst_attack": dst_a[:10],
    "final_pred": dst_pred[:10], "true_label": y_test.iloc[:10].values
})
print(fusion_debug)

rf_wrong = (rf_model.predict(X_test_scaled) != y_test.values)
dst_wrong = (dst_pred != y_test.values)
print(" Error analysis...\n")
print(f"RF wrong: {rf_wrong.sum()}")
print(f"DST wrong: {dst_wrong.sum()}")
print(f"DST fixes RF errors: {(rf_wrong & ~dst_wrong).sum()}")