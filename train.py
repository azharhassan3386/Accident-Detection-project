"""Train XGBoost with video-wise (unseen video) cross-validation.
Extra: --drop se scene-shortcut features band karo, AUC aur threshold sweep dikhao.
Dropped features training mein 0 kar diye jate hain, is liye model unpar split nahi
karta aur detect.py mein koi change nahi chahiye."""
import argparse
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, average_precision_score, roc_auc_score)
from xgboost import XGBClassifier
from pipeline import FEATURE_NAMES

p = argparse.ArgumentParser()
p.add_argument("--features", default="features.csv")
p.add_argument("--model", default="model.json")
p.add_argument("--drop", default="n_tracks", help="comma separated; '' = kuch nahi")
a = p.parse_args()

drop = [d for d in a.drop.split(",") if d]
df = pd.read_csv(a.features)
X, y, g = df[FEATURE_NAMES].values.astype(float), df["label"].values, df["video"].values
for d in drop:
    X[:, FEATURE_NAMES.index(d)] = 0.0
print("Dropped features:", drop or "none")
print("Windows: accident =", int((y == 1).sum()), " normal =", int((y == 0).sum()))
spw = (y == 0).sum() / max((y == 1).sum(), 1)


def make():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                         scale_pos_weight=spw, eval_metric="logloss")


folds = min(5, len(set(g)))
prob = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
    m = make().fit(X[tr], y[tr])
    prob[te] = m.predict_proba(X[te])[:, 1]

print("\nCross-validation (unseen videos)")
print("ROC-AUC :", round(roc_auc_score(y, prob), 3), "(0.5 = andaza, 1.0 = perfect)")
print("PR-AUC  :", round(average_precision_score(y, prob), 3),
      "(random baseline =", round(float(y.mean()), 3), ")")
print("\nThreshold  Precision  Recall   F1")
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    pr = (prob >= t).astype(int)
    print(f"  {t:.1f}       {precision_score(y, pr, zero_division=0):.3f}      "
          f"{recall_score(y, pr, zero_division=0):.3f}   {f1_score(y, pr, zero_division=0):.3f}")
print("\nConfusion matrix (threshold 0.5):\n", confusion_matrix(y, (prob >= 0.5).astype(int)))

final = make().fit(X, y)
final.save_model(a.model)
imp = sorted(zip(FEATURE_NAMES, final.feature_importances_), key=lambda t: -t[1])[:8]
print("\nTop features:", [(n, round(float(v), 3)) for n, v in imp])
print("saved", a.model)