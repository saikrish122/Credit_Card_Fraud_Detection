# Sai: start here

This is a standalone extension of your credit-card fraud project. Keep the old notebook unchanged. Extract the entire ZIP before opening the new notebook. Do not run files inside the ZIP viewer.

No API key, paid LLM, or GPU is required. Internet is needed to install packages. Use Python 3.12 (3.11 is also a reasonable option); avoid reusing the old environment with broken PyTorch DLLs.

## 1. Open the extracted fraud_mlops folder in VS Code

Put your existing Kaggle **creditcard.csv** inside **data/**. Its columns must be Time, V1–V28, Amount, Class. Do not put the CSV on GitHub.

In the VS Code PowerShell terminal, run these commands one at a time:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands use the environment directly; PowerShell activation is unnecessary. If `py -3.12` fails, install Python 3.12 first. If packages were installed while VS Code had an old kernel open, restart the kernel and select this folder's `.venv`.

## 2. Train once

Choose ONE route:

- Open **Fraud_Detection_Extension.ipynb**, select `.venv` as the kernel, and Run All. The notebook uses the companion Python modules; it is not a standalone upload.
- Or use the terminal:

```powershell
.\.venv\Scripts\python.exe train.py --data data/creditcard.csv --out artifacts
```

The portfolio run uses 4 Random Forest configurations × 5 CV folds and up to 20 epochs per neural architecture. Runtime depends on your computer; no completion time is guaranteed. It prints progress. Do not stop it just because a Random Forest fit takes time.

For an optional installation/pipeline check first:

```powershell
.\.venv\Scripts\python.exe train.py --data data/creditcard.csv --out artifacts_quick --quick
```

Quick-run metrics are NOT the portfolio results. The normal command still uses `artifacts`. An existing output directory is deliberately rejected to avoid mixing results. For a failed run, rename that directory to `artifacts_failed` before retrying. Do not repeatedly tune against test scores.

## 3. Read the actual outputs

A successful run ends with `artifacts/completion.json`. Inspect:

- `RESULTS.md` and `test_metrics.csv`: actual model results.
- `metadata.json`: architecture, hyperparameters, versions, dataset hash and split.
- `rf_cv_results.csv`: 5-fold search evidence.
- `pytorch_*_training.csv`: loss and validation AP by epoch.
- `rf_shap_global.png`, `rf_shap_local.png`: explanations of the Random Forest.
- `confusion_matrix.png`, `precision_recall.png`: selected-model mistakes and comparisons.
- `simulated_cost_sensitivity.csv`: assumptions-based financial scenarios, not measured savings.
- `RESUME_AFTER_REVIEW.txt`: draft bullets using YOUR run's metrics. Review before copying.

## 4. Run the API

```powershell
$env:TEST_ARTIFACT_DIR = (Resolve-Path artifacts).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Keep that terminal running. Open **http://127.0.0.1:8000/docs**. In another terminal in this folder:

```powershell
$body = Get-Content artifacts/example_request.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

Press Ctrl+C in the API terminal when finished. The API serves the validation-selected model, which may be the Random Forest, logistic regression, or PyTorch network. A neural network does not have to win for its implementation to be useful evidence.

## 5. View MLflow

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

Open **http://127.0.0.1:5000**. The code creates actual experiment runs with parameters, metrics and model artifacts. These are local experiments, not a cloud registry/deployment.

## 6. Verify Docker (optional installation, required before claiming Docker execution)

Install and start Docker Desktop if you do not have it. Stop the standalone API first so port 8000 is available. In this folder:

```powershell
docker compose up --build
```

Repeat the PowerShell prediction request above and check http://127.0.0.1:8000/health. Then stop:

```powershell
docker compose down
```

The Compose service mounts the trained `artifacts` folder read-only. The dataset is not copied into the image. Container build/run was not tested in the author's execution environment because Docker was unavailable. A Dockerfile alone is not proof that you deployed anything.

## 7. Submit evidence before Tuesday, 27 October 2026

1. Run the normal training command successfully.
2. Inspect the confusion matrix and all four models' scores, including weak results.
3. Run the API test and save a screenshot of a successful request.
4. Verify Docker if you want a Dockerized-serving resume bullet.
5. Copy a small set of genuine plots/results to a `results/` folder in your GitHub repository. The supplied `.gitignore` excludes CSV/model artifacts globally; use `git add -f results/test_metrics.csv` only for the specific small results file you want to publish. Never force-add the dataset or predictions.
6. Update the README with your metrics and exact commands that passed. Upload code and notebook together.
7. Review `RESUME_AFTER_REVIEW.txt`; remove any feature you did not run or cannot explain yet. No score increase is guaranteed.

If an error occurs, send the complete traceback and command used. If the run completes, send **test_metrics.csv**, **metadata.json**, and **completion.json** so the final resume wording can be checked against evidence.
