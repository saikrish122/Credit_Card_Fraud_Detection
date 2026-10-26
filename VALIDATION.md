# Validation performed on 25 October 2026

## Verified in Linux / Python 3.12

- Ran a complete training/evaluation pipeline on a private synthetic fixture: 4,000 rows with 30 features, binary labels, nonnegative Time/Amount. This is software validation only, not credit-card fraud performance evidence.
- Completed 4 Random Forest configurations × 5-fold CV, logistic regression, and both PyTorch architectures with early stopping.
- Generated metric tables, PR/confusion plots, SHAP global/local plots and financial sensitivity scenarios.
- Verified SHAP additivity against the Random Forest's output.
- Verified saved-model predictions against in-memory predictions; also exercised API model loading and inference for all four model branches.
- Completed real local MLflow SQLite experiment logging with parameters, metrics, model artifacts and four finished runs.
- Passed four unittest methods, including deduplication/conflicting-label checks, split isolation, metric/cost arithmetic, and FastAPI inference/validation. API testing includes feature-order invariance, empty batches, missing features, extra label fields and negative Amount.
- Validated notebook JSON/schema and compiled its Python cells and Python modules.
- Synthetic scores, synthetic model artifacts and synthetic data are not included in this deliverable.

## Not verified here

- Training or model performance on the user's real `creditcard.csv`: dataset bytes were not available.
- Windows installation/execution: user instructions were prepared, but execution here was on Linux.
- Docker build/run: no Docker executable/daemon available here.
- Cloud deployment, live business impact, production reliability, temporal generalization, or the resume-review score.

See requirements-tested.txt for direct dependency versions used. requirements.txt provides bounded ranges rather than a fully locked transitive environment. Freeze your successful local environment for reproducibility. Retain the original artifact location when using the local MLflow UI: MLflow artifact URIs use absolute paths and do not automatically relocate when folders move.

An initial MLflow-skinny-only installation lacked SQLite tracking support; the package requires the full **mlflow** distribution, and the full pipeline was rerun successfully after installing it.
