"""Shared data checks and evaluation. Class 1 always means fraud."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix, precision_recall_curve)
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
SEED = 42

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def load_data(path):
    raw = pd.read_csv(path)
    if set(raw.columns) != set(FEATURES + ['Class']):
        raise ValueError('CSV must contain Time, V1–V28, Amount, Class only.')
    if not all(pd.api.types.is_numeric_dtype(t) for t in raw.dtypes):
        raise ValueError('All columns must be numeric.')
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Missing or infinite values found.')
    if set(raw.Class.unique()) != {0, 1} or (raw.Amount < 0).any() or (raw.Time < 0).any():
        raise ValueError('Need binary Class and nonnegative Amount/Time.')
    clean = raw.drop_duplicates().copy()
    if clean.duplicated(FEATURES, keep=False).any():
        raise ValueError('Identical features with conflicting labels; investigate.')
    audit = {'raw_rows': len(raw), 'retained_rows': len(clean),
             'duplicates_removed': len(raw)-len(clean),
             'fraud_before': int(raw.Class.sum()), 'fraud_after': int(clean.Class.sum())}
    return clean, audit

def split_data(df):
    # Same outer split as Sai's existing notebook: 25% development, 75% test.
    dev, test = train_test_split(df, train_size=.25, stratify=df.Class, random_state=SEED)
    train, val = train_test_split(dev, test_size=.20, stratify=dev.Class, random_state=SEED)
    if train.Class.value_counts().min() < 5:
        raise ValueError('Need at least five training examples per class for CV.')
    return train, val, test

def choose_threshold(y, scores):
    # F2 weights recall more than precision. Choose using validation ONLY.
    p, r, thresholds = precision_recall_curve(y, scores)
    f2 = 5*p[:-1]*r[:-1] / np.maximum(4*p[:-1]+r[:-1], 1e-12)
    # Highest threshold wins exact ties, to reduce unnecessary alerts.
    best = np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])

def metrics(y, scores, threshold):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'average_precision': float(average_precision_score(y, scores)),
            'roc_auc': float(roc_auc_score(y, scores)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'threshold': float(threshold), 'alert_rate': float(pred.mean())}

def simulated_cost(y, scores, amount, threshold, prevention=.8, review_cost=5., false_cost=0.):
    y, amount = np.asarray(y), np.asarray(amount)
    alerts = np.asarray(scores) >= threshold
    baseline = float(amount[y == 1].sum())
    captured = float(amount[(y == 1) & alerts].sum())
    cost = baseline - prevention*captured + review_cost*alerts.sum() + false_cost*((y == 0)&alerts).sum()
    return {'prevention_rate': prevention, 'review_cost_per_alert': review_cost,
            'false_alert_extra_cost': false_cost, 'baseline_loss': baseline,
            'detected_fraud_amount': captured, 'simulated_cost': float(cost),
            'simulated_net_savings': float(baseline-cost)}
