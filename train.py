"""Run once on creditcard.csv. All choices are frozen before test scoring."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay, ConfusionMatrixDisplay
from core import FEATURES, SEED, save_json, load_data, split_data, choose_threshold, metrics, simulated_cost
from neural import train_nn, predict_nn

def run(data_path, out, quick=False):
    out = Path(out).resolve()
    from datetime import datetime

    out = out / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    df, audit = load_data(data_path)
    train, val, test = split_data(df)
    save_json(out/'audit.json', audit)
    for name, frame in [('train', train), ('validation', val), ('test', test)]:
        pd.DataFrame({'original_row_index': frame.index, 'Class': frame.Class}).to_csv(out/f'{name}_indices.csv', index=False)
    xt, xv, xe = [f[FEATURES] for f in (train, val, test)]
    yt, yv, ye = [f.Class for f in (train, val, test)]
    print(f'Rows: train={len(train)}, validation={len(val)}, test={len(test)}', flush=True)
    # No scaling leakage: each CV model fits independently; NN scaler sees training only.
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED))
    lr.fit(xt, yt)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {'n_estimators': [50, 100], 'max_depth': [8, 12],
         'min_samples_leaf': [1, 3], 'class_weight': [None, 'balanced_subsample']},
        n_iter=2 if quick else 4, scoring='average_precision',
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=2, pre_dispatch=2, random_state=SEED, error_score='raise', verbose=1)
    search.fit(xt, yt)
    rf = search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(out/'rf_cv_results.csv', index=False)
    scaler = StandardScaler().fit(xt)
    st, sv, se = [scaler.transform(x).astype(np.float32) for x in (xt, xv, xe)]
    models = {'logistic_regression': lr, 'random_forest': rf}
    validation_scores = {name: model.predict_proba(xv)[:, 1] for name, model in models.items()}
    architectures = [(32,16)] if quick else [(32,16), (64,32)]
    networks, histories = {}, {}
    for hidden in architectures:
        name = f'pytorch_{hidden[0]}_{hidden[1]}'
        model, history, weight = train_nn(st, yt, sv, yv, hidden, epochs=2 if quick else 20)
        networks[name] = (model, hidden)
        histories[name] = history
        pd.DataFrame(history).to_csv(out/f'{name}_training.csv', index=False)
        validation_scores[name] = predict_nn(model, sv)
        torch.save(model.state_dict(), out/f'{name}.pt')
    thresholds = {name: choose_threshold(yv, p) for name, p in validation_scores.items()}
    val_ap = {name: float(average_precision_score(yv, p)) for name, p in validation_scores.items()}
    selected = max(val_ap, key=val_ap.get)
    # Save selection BEFORE evaluating the test partition. Never select the test winner.
    metadata = {'seed': SEED, 'features': FEATURES, 'selected_model': selected,
                'thresholds': thresholds, 'validation_ap': val_ap,
                'architectures': {n:list(h) for n,(_,h) in networks.items()},
                'rf_best_params': search.best_params_, 'rf_cv_ap': float(search.best_score_),
                'nn_pos_weight': weight, 'mode': 'quick_pipeline_check' if quick else 'portfolio',
                'split': {'train':len(train), 'validation':len(val), 'test':len(test)},
                'split_policy': '20% train / 5% validation / 75% historical test; stratified seed 42',
                'test_caveat': 'Same outer test as prior notebook if input/order unchanged; previously examined, not a pristine independent holdout.',
                'dataset_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                'versions': {p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch','shap','mlflow','fastapi']}}
    save_json(out/'metadata.json', metadata)
    joblib.dump({'models': models, 'scaler': scaler}, out/'models.joblib')
    all_test_scores = {n:m.predict_proba(xe)[:,1] for n,m in models.items()}
    all_test_scores.update({n:predict_nn(m, se) for n,(m,_) in networks.items()})
    rows=[]
    fig, ax = plt.subplots(figsize=(7,5))
    for name, scores in all_test_scores.items():
        rows.append({'model':name, 'selected_on_validation':name==selected,
                     'validation_ap':val_ap[name], **metrics(ye, scores, thresholds[name])})
        PrecisionRecallDisplay.from_predictions(ye, scores, name=name, ax=ax)
    ax.set_title('Historical test partition: precision–recall')
    fig.tight_layout(); fig.savefig(out/'precision_recall.png', dpi=150); plt.close(fig)
    results = pd.DataFrame(rows)
    results.to_csv(out/'test_metrics.csv', index=False)
    p = all_test_scores[selected]
    pred = p >= thresholds[selected]
    ConfusionMatrixDisplay.from_predictions(ye, pred, labels=[0,1], display_labels=['Legitimate','Fraud'], colorbar=False)
    plt.title(f'Validation-selected model: {selected}')
    plt.tight_layout(); plt.savefig(out/'confusion_matrix.png', dpi=150); plt.close()
    pd.DataFrame({'original_row_index': test.index, 'actual':ye.to_numpy(),
                  'score':p, 'predicted':pred.astype(int), 'Amount':test.Amount.to_numpy()}).to_csv(out/'test_predictions.csv', index=False)
    costs = [simulated_cost(ye,p,test.Amount,thresholds[selected],prevention=r, review_cost=c, false_cost=f)
             for r in [.5,.8] for c in [1.,5.] for f in [0.,5.]]
    pd.DataFrame(costs).to_csv(out/'simulated_cost_sensitivity.csv', index=False)
    # SHAP explains the Random Forest, even if a different model is served.
    print('Computing SHAP on up to 100 validation transactions...', flush=True)
    import shap
    sample = xv.sample(min(100,len(xv)), random_state=SEED)
    explainer = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
    raw = explainer.shap_values(sample)
    values = raw[1] if isinstance(raw,list) else np.asarray(raw)[:,:,1]
    expected = float(np.asarray(explainer.expected_value)[1])
    np.testing.assert_allclose(expected+values.sum(axis=1), rf.predict_proba(sample)[:,1], atol=1e-5)
    pd.DataFrame(values, columns=FEATURES, index=sample.index).to_csv(out/'rf_shap_values.csv', index_label='original_row_index')
    importance = pd.Series(np.abs(values).mean(axis=0), index=FEATURES).sort_values()
    importance.to_csv(out/'rf_shap_importance.csv', header=['mean_absolute_shap'])
    importance.tail(15).plot.barh(figsize=(8,5), title='Random Forest: mean absolute SHAP (validation sample)')
    plt.tight_layout(); plt.savefig(out/'rf_shap_global.png', dpi=150); plt.close()
    # Explain the highest RF score in the sample; this is not necessarily a true fraud.
    i = int(np.argmax(rf.predict_proba(sample)[:,1]))
    shap.plots.waterfall(shap.Explanation(values=values[i], base_values=expected,
                         data=sample.iloc[i].to_numpy(), feature_names=FEATURES), show=False)
    plt.tight_layout(); plt.savefig(out/'rf_shap_local.png', dpi=150, bbox_inches='tight'); plt.close()
    # Example API request contains one development row, never a label.
    save_json(out/'example_request.json', {'transactions':[xt.iloc[0].to_dict()]})
    # Benchmark model inference only: excludes HTTP, network and cold start.
    from predict import Predictor
    predictor = Predictor(out)
    batch = xv.iloc[:min(100,len(xv))]
    expected_scores = validation_scores[selected][:len(batch)]
    np.testing.assert_allclose(predictor.predict(batch), expected_scores, rtol=1e-5, atol=1e-6)
    durations=[]
    for _ in range(20):
        begin=time.perf_counter(); predictor.predict(batch)
        durations.append((time.perf_counter()-begin)*1000)
    save_json(out/'inference_benchmark.json', {'batch_size':len(batch), 'repeats':20,
        'median_batch_ms':float(np.median(durations)), 'p95_batch_ms':float(np.percentile(durations,95)),
        'scope':'Local model inference only; not API latency or production throughput'})
    # Real MLflow runs: SQLite metadata plus local artifacts; no cloud account needed.
    mlflow.set_tracking_uri('sqlite:///'+(out/'mlflow.db').as_posix())
    if mlflow.get_experiment_by_name('fraud_detection') is None:
        mlflow.create_experiment('fraud_detection', artifact_location=(out/'mlruns').as_uri())
    mlflow.set_experiment('fraud_detection')
    for row in rows:
        with mlflow.start_run(run_name=row['model']) as active:
            mlflow.log_params({'seed':SEED, 'model':row['model'], 'mode':metadata['mode'],
                               'selected':row['selected_on_validation'], 'train_rows':len(train)})
            if row['model']=='random_forest': mlflow.log_params(search.best_params_)
            if row['model'] in networks:
                mlflow.log_params({'hidden_layers':str(networks[row['model']][1]),
                                   'dropout':.2,'lr':.001,'pos_weight':weight,'optimizer':'Adam'})
                for h in histories[row['model']]:
                    mlflow.log_metrics({'train_loss':h['train_loss'], 'epoch_validation_ap':h['validation_ap']}, step=h['epoch'])
            mlflow.log_metrics({k:float(v) for k,v in row.items() if k not in ['model','selected_on_validation']})
            mlflow.log_artifact(str(out/'metadata.json'))
            if row['model'] in networks: mlflow.log_artifact(str(out/f"{row['model']}.pt"))
            else: mlflow.log_artifact(str(out/'models.joblib'))
            mlflow.log_artifact(str(out/'test_metrics.csv'))
    chosen = results.loc[results.model==selected].iloc[0]
    report = (f'# Run results\n\nMode: {metadata["mode"]}.\n\n'
              f'Selected on validation AP: **{selected}**. Test AP: {chosen.average_precision:.4f}; '
              f'precision: {chosen.precision:.4f}; recall: {chosen.recall:.4f}; F1: {chosen.f1:.4f}.\n\n'
              'The test partition has been used in the previous project. These are retrospective benchmark results, '
              'not independent prospective performance. AP means average precision, not trapezoidal PR-AUC.\n\n'
              'SHAP explains the Random Forest only. Cost results are simulations. '
              'Serving and container execution require separate verification.\n')
    (out/'RESULTS.md').write_text(report, encoding='utf-8')
    # This draft uses measured training/evaluation evidence only; no unverified deployment claim.
    (out/'RESUME_AFTER_REVIEW.txt').write_text(
        ('QUICK RUN: DO NOT USE THESE METRICS ON A RESUME.\n' if quick else '')+
        'Credit Card Fraud Detection — PyTorch, scikit-learn, SHAP, MLflow, FastAPI\n'
        f'- Compared logistic regression, a Random Forest tuned with {search.n_iter} configurations and 5-fold stratified CV, '
        f'and {len(networks)} custom PyTorch MLP architecture(s) using validation average precision.\n'
        f'- Evaluated the validation-selected {selected} on {len(test):,} historical test transactions: '
        f'AP {chosen.average_precision:.3f}, precision {chosen.precision:.1%}, recall {chosen.recall:.1%}; '
        'selected the decision threshold using validation F2.\n'
        '- Generated Random Forest SHAP explanations and logged model experiments in MLflow.\n'
        'Only after running API/Docker successfully, add: Served predictions locally through a Dockerized FastAPI endpoint.\n', encoding='utf-8')
    save_json(out/'completion.json', {'status':'training_evaluation_complete', 'seconds':time.perf_counter()-started,
                                    'api_verified':False,'docker_verified':False,'cloud_deployed':False})
    print(results.to_string(index=False))
    print('Completed. Results:', out, flush=True)
    return out

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', default='data/creditcard.csv')
    parser.add_argument('--out', default='artifacts')
    parser.add_argument('--quick', action='store_true', help='2 epochs, 2 RF candidates; pipeline debugging only')
    args=parser.parse_args()
    run(args.data,args.out,args.quick)
