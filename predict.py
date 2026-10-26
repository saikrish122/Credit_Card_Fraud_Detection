"""Exactly the same preprocessing and threshold used during evaluation."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from neural import FraudNet, predict_nn

class Predictor:
    def __init__(self, artifact_dir):
        folder=Path(artifact_dir)
        self.metadata=json.loads((folder/'metadata.json').read_text())
        self.name=self.metadata['selected_model']
        self.features=self.metadata['features']
        self.threshold=self.metadata['thresholds'][self.name]
        # Load only artifacts you trained/trust: joblib uses Python pickle.
        saved=joblib.load(folder/'models.joblib')
        self.scaler=saved['scaler']
        if self.name.startswith('pytorch_'):
            self.model=FraudNet(len(self.features), self.metadata['architectures'][self.name])
            self.model.load_state_dict(torch.load(folder/f'{self.name}.pt',map_location='cpu',weights_only=True))
        else:
            self.model=saved['models'][self.name]

    def predict(self, frame):
        if not isinstance(frame,pd.DataFrame) or set(frame.columns)!=set(self.features):
            raise ValueError('Provide exactly the expected named features.')
        x=frame[self.features].astype(float)
        if len(x)==0 or not np.isfinite(x.to_numpy()).all():
            raise ValueError('Need nonempty finite numeric features.')
        if (x[['Time','Amount']]<0).any().any():
            raise ValueError('Time and Amount cannot be negative.')
        if self.name.startswith('pytorch_'):
            return predict_nn(self.model,self.scaler.transform(x).astype(np.float32))
        return self.model.predict_proba(x)[:,1]
