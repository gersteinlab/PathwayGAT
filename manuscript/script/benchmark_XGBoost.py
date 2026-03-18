import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df_gene = pd.read_table('../data/TCGA_COAD_gene.txt', sep='\t', header=0)
df_microbe = pd.read_table('../data/TCGA_COAD_microbe.txt', sep='\t', header=0)
df_meta = pd.read_table('../data/COAD_sample_metadata.processed.txt', sep='\t', header=0)

X = np.concatenate([df_gene.iloc[:, 1:], df_microbe.iloc[:, 1:]], axis=1)
encoder = LabelEncoder()
y = encoder.fit_transform(df_meta['sample_type2'])

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss"
)

param_grid = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 200]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=cv,
    scoring={"accuracy":"accuracy", "auc":"roc_auc"},
    refit="auc",
    return_train_score=False
)

grid.fit(X, y)

results = pd.DataFrame(grid.cv_results_)
results.to_csv('benchmark/JHU_COAD_XGBoost_result.txt', sep='\t', header=True, index=False)
