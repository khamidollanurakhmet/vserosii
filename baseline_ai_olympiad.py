# -*- coding: utf-8 -*-
"""
- Запустите файл: `python baseline_ai_olympiad.py --example iris`
"""

# REQUIREMENTS (рекомендуется установить в виртуальном окружении):
# pip install numpy pandas scikit-learn xgboost lightgbm joblib
# optional: pip install torch torchvision torchaudio optuna

from __future__ import annotations

# ---------------------- Импорты ----------------------
import os
import sys
import math
import time
import json
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator

# optional imports with graceful fallback
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import joblib
except Exception:
    joblib = None

# ---------------------- Конфиг ----------------------
@dataclass
class Config:
    seed: int = 42
    device: str = 'cpu'
    num_workers: int = 4
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    early_stopping_patience: int = 5
    save_dir: str = 'models'

CFG = Config()

# ---------------------- Утилиты ----------------------

def set_seed(seed: int = 42) -> None:
    """Фиксируем сиды для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def prepare_device(preferred: Optional[str] = None) -> str:
    """Вернуть строку устройства. Если CUDA доступна, использовать её."""
    if preferred is not None:
        return preferred
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------------- Датасеты / загрузчики ----------------------

def get_classification_sklearn_dataset(name: str = 'iris') -> Tuple[np.ndarray, np.ndarray]:
    """Загрузить простой датасет sklearn по имени для примера."""
    name = name.lower()
    if name == 'iris':
        data = datasets.load_iris()
        X, y = data.data, data.target
    elif name == 'wine':
        data = datasets.load_wine()
        X, y = data.data, data.target
    elif name == 'breast_cancer' or name == 'cancer':
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def prepare_dataloaders_from_arrays(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1, cfg: Config = CFG):
    """Разбить массивы на train/val/test и вернуть numpy наборы.
    (Для PyTorch используйте отдельный конвертер.)
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=cfg.seed, stratify=y)
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_rel, random_state=cfg.seed, stratify=y_train_full)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ---------------------- Метрики ----------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
    try:
        if y_proba is not None and y_proba.shape[1] >= 2:
            # multiclass roc_auc requires one-vs-rest or average
            metrics['roc_auc_ovr'] = float(roc_auc_score(pd.get_dummies(y_true), y_proba, average='macro', multi_class='ovr'))
    except Exception:
        pass
    return metrics

# ---------------------- Sklearn модели (быстрый старт) ----------------------

def train_sklearn_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, model: BaseEstimator) -> Tuple[BaseEstimator, Dict[str, float]]:
    """Обучить sklearn-модель и вернуть обученную модель + метрики на валидации."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_val)
        except Exception:
            y_proba = None
    metrics = classification_metrics(y_val, y_pred, y_proba)
    return model, metrics

# ---------------------- Boosting (XGBoost / LightGBM) ----------------------

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Optional[dict] = None, num_round: int = 100) -> Tuple[Any, Dict[str, float]]:
    if xgb is None:
        raise ImportError('xgboost не установлен')
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    default = {'objective': 'multi:softprob' if len(np.unique(y_train)) > 2 else 'binary:logistic', 'eval_metric': 'mlogloss' if len(np.unique(y_train)) > 2 else 'logloss', 'verbosity': 0}
    if params:
        default.update(params)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(default, dtrain, num_boost_round=num_round, evals=watchlist, early_stopping_rounds=CFG.early_stopping_patience, verbose_eval=False)
    y_proba = bst.predict(dval)
    if y_proba.ndim > 1:
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = (y_proba > 0.5).astype(int)
    metrics = classification_metrics(y_val, y_pred, y_proba if y_proba.ndim > 1 else None)
    return bst, metrics


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Optional[dict] = None, num_round: int = 100) -> Tuple[Any, Dict[str, float]]:
    if lgb is None:
        raise ImportError('lightgbm не установлен')
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    default = {'objective': 'multiclass' if len(np.unique(y_train)) > 2 else 'binary', 'metric': 'multi_logloss' if len(np.unique(y_train)) > 2 else 'binary_logloss', 'verbose': -1}
    if params:
        default.update(params)
    booster = lgb.train(default, lgb_train, num_boost_round=num_round, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=CFG.early_stopping_patience, verbose_eval=False)
    y_proba = booster.predict(X_val)
    if isinstance(y_proba, list) or (isinstance(y_proba, np.ndarray) and y_proba.ndim > 1):
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = (np.array(y_proba) > 0.5).astype(int)
    metrics = classification_metrics(y_val, y_pred, y_proba if np.array(y_proba).ndim > 1 else None)
    return booster, metrics

# ---------------------- PyTorch: простой MLP и тренировка ----------------------

if TORCH_AVAILABLE:
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_sizes: List[int] = [64, 32], num_classes: int = 2, dropout: float = 0.1):
            super().__init__()
            layers = []
            in_dim = input_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, num_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    def train_torch_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: Config = CFG, hidden_sizes: List[int] = [64, 32]) -> Tuple[nn.Module, Dict[str, float]]:
        device = torch.device(cfg.device)
        num_classes = len(np.unique(y_train))
        model = SimpleMLP(input_dim=X_train.shape[1], hidden_sizes=hidden_sizes, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        best_val = -1e9
        best_model = None
        epochs_no_improve = 0

        for epoch in range(cfg.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            preds = []
            probs = []
            trues = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    p = torch.softmax(logits, dim=1).cpu().numpy()
                    preds.append(np.argmax(p, axis=1))
                    probs.append(p)
                    trues.append(yb.numpy())
            y_pred = np.concatenate(preds)
            y_proba = np.vstack(probs)
            y_true = np.concatenate(trues)
            metrics = classification_metrics(y_true, y_pred, y_proba)
            score = metrics.get('f1_macro', metrics.get('accuracy', 0.0))
            if score > best_val:
                best_val = score
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                break
        # load best model
        if best_model is not None:
            model.load_state_dict(best_model)
        return model, metrics

# ---------------------- Энсамблирование ----------------------

def build_voting_ensemble(models: List[Tuple[str, BaseEstimator]]) -> VotingClassifier:
    """Создать простое голосующее объединение sklearn-моделей."""
    vc = VotingClassifier(estimators=models, voting='soft')
    return vc

# ---------------------- Сохранение / загрузка моделей ----------------------

def save_model_sklearn(model: BaseEstimator, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    if joblib is not None:
        joblib.dump(model, path)
    else:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(model, f)


def load_model_sklearn(path: str) -> BaseEstimator:
    if joblib is not None:
        return joblib.load(path)
    else:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

# ---------------------- Пример гиперпараметрической сетки (GridSearch) ----------------------

def example_gridsearch(X: np.ndarray, y: np.ndarray):
    pipe = Pipeline([('scale', StandardScaler()), ('clf', RandomForestClassifier(random_state=CFG.seed))])
    param_grid = {'clf__n_estimators': [50, 100], 'clf__max_depth': [None, 10]}
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring='f1_macro', n_jobs=1)
    gs.fit(X, y)
    return gs

# ---------------------- CLI / пример запуска ----------------------

def run_example(dataset_name: str = 'iris'):
    set_seed(CFG.seed)
    CFG.device = prepare_device()
    print(f"Запуск примера для датасета {dataset_name}, устройство: {CFG.device}")
    X, y = get_classification_sklearn_dataset(dataset_name)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataloaders_from_arrays(X, y, test_size=0.2, val_size=0.1)

    # Стандартный sklearn baseline
    pipe = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    model, metrics = train_sklearn_model(X_train, y_train, X_val, y_val, pipe)
    print('Sklearn baseline metrics:', metrics)

    # Быстрый RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=CFG.seed)
    rf, rf_metrics = train_sklearn_model(X_train, y_train, X_val, y_val, rf)
    print('RandomForest metrics:', rf_metrics)

    # Если доступен xgboost — попробовать
    if xgb is not None:
        try:
            bst, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, num_round=50)
            print('XGBoost metrics:', xgb_metrics)
        except Exception as e:
            print('XGBoost failed:', e)
    else:
        print('XGBoost не установлен — пропускаю')

    # Если доступен lightgbm — попробовать
    if lgb is not None:
        try:
            booster, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val, num_round=50)
            print('LightGBM metrics:', lgb_metrics)
        except Exception as e:
            print('LightGBM failed:', e)
    else:
        print('LightGBM не установлен — пропускаю')

    # PyTorch example
    if TORCH_AVAILABLE:
        try:
            torch_cfg = CFG
            torch_cfg.epochs = 50
            model_torch, torch_metrics = train_torch_model(X_train, y_train, X_val, y_val, cfg=torch_cfg)
            print('PyTorch MLP metrics:', torch_metrics)
        except Exception as e:
            print('PyTorch training failed:', e)
    else:
        print('PyTorch не установлен — пропускаю')

    # Финальная оценка лучшей sklearn модели на тесте
    y_test_pred = model.predict(X_test)
    final_metrics = classification_metrics(y_test, y_test_pred)
    print('Final test metrics (baseline):', final_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline для олимпиады по ИИ — единый файл')
    parser.add_argument('--example', type=str, default='iris', help='название примера: iris|wine|breast_cancer')
    return parser.parse_args()


def main():
    args = parse_args()
    run_example(args.example)


if __name__ == '__main__':
    main()
