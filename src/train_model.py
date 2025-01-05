import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


from feature_func import (
    get_path,
    RANDOM_SEED
)

FILEPATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(FILEPATH, '../')
DATA_PATH = get_path(BASE_PATH, "data")
ORIGINAL_DATA_PATH = get_path(DATA_PATH, "original")
FINAL_DATA_PATH = get_path(DATA_PATH, "final")
FINAL_FEATURES_PATH = get_path(FINAL_DATA_PATH, 'final_features.csv')
# Model stuff.
MODEL_PATH = get_path(BASE_PATH, "models")
FINAL_MODEL_PATH = get_path(MODEL_PATH, "final_model")
FINAL_MODEL_FILE = get_path(FINAL_MODEL_PATH, 'model.pkl')


if __name__ == '__main__':
    all_features = pd.read_csv(FINAL_FEATURES_PATH)
    all_features.set_index(['gameId', 'playId', 'frameId'], inplace=True)

    X, y = all_features.drop('target', axis=1), all_features.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,
        stratify=y_test
    )

    models = {}
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models['XGBClassifier'] = XGBClassifier(
        max_depth=8,
        learning_rate=0.01,
        gamma=0.1,
        reg_alpha=0.1,
        n_estimators=10000,
        eval_metric="aucpr",
        early_stopping_rounds=200,
    )

    eval_set = [(X_val, y_val)]
    models['XGBClassifier'].fit(X_train, y_train, eval_set=eval_set)

    with open(FINAL_MODEL_FILE, 'wb') as f:
        pickle.dump(models['XGBClassifier'], f)
