# Learn the project in this order

1. **Problem and split:** Class 1 means fraud. Explain why AP, precision and recall matter more than overall accuracy. Explain why the historical 25/75 split was preserved, and why the test is not pristine.
2. **Data leakage:** In `core.py`, find deduplication and the two splits. In `train.py`, find where StandardScaler is fitted. No validation/test row is used to learn scaling statistics.
3. **Random Forest:** Explain 4 random parameter settings, 5 stratified folds, and `scoring='average_precision'`. CV chooses RF parameters, not the test score.
4. **Neural network:** Read `FraudNet`. Explain input size 30, two hidden layers, ReLU, dropout and output size 1. Follow zero_grad → forward → loss → backward → optimizer.step.
5. **Imbalance:** `pos_weight` gives rare positive cases more loss weight; it does not create examples. It can affect score calibration and false alarms.
6. **Early stopping:** Save the state with best validation AP. Stop after four non-improving epochs. Explain why `.eval()` turns off dropout.
7. **Threshold:** F2 favors recall. Validation selects the threshold; test reports mistakes afterward. The threshold need not be 0.5.
8. **Explainability:** SHAP describes RF contributions, not causes or explanations of the neural network. V features have no readable original business names.
9. **MLflow:** Parameters describe a run, metrics describe measured performance, artifacts store files. This project tracks locally.
10. **API and Docker:** `predict.py` reuses preprocessing; `app.py` accepts JSON; Docker packages dependencies. Running locally is not the same as deploying to cloud/production.

Before an interview, explain one successful prediction and one false positive, show where the best model was selected, and honestly describe which parts you implemented with assistance. You do not need to memorize every line, but you should understand the results and main decisions before claiming proficiency.
