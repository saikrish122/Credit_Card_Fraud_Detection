"""Synthetic checks verify code behavior, never portfolio performance."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import FEATURES, load_data, split_data, choose_threshold, metrics, simulated_cost

class CoreTests(unittest.TestCase):
    def test_cost_arithmetic(self):
        # One detected fraud, one missed fraud, one false alert.
        result=simulated_cost([1,1,0],[.9,.1,.8],[100,50,20],.5)
        self.assertEqual(result['simulated_net_savings'],70.)
    def test_threshold_and_confusion(self):
        y=np.array([0,0,1,1]); s=np.array([.1,.2,.7,.8])
        t=choose_threshold(y,s)
        m=metrics(y,s,t)
        self.assertEqual((m['tp'],m['tn'],m['fp'],m['fn']),(2,2,0,0))
    def test_dedup_and_disjoint_splits(self):
        x,y=make_classification(n_samples=2000,n_features=30,weights=[.9,.1],random_state=42)
        frame=pd.DataFrame(x,columns=FEATURES)
        frame['Time']=np.arange(len(frame)); frame['Amount']=np.abs(frame.Amount)
        frame['Class']=y
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.csv'
            pd.concat([frame,frame.iloc[:5]]).to_csv(path,index=False)
            clean,audit=load_data(path)
            self.assertEqual(audit['duplicates_removed'],5)
            train,val,test=split_data(clean)
            self.assertFalse(set(train.index)&set(val.index))
            self.assertFalse(set(train.index)&set(test.index))
            self.assertFalse(set(val.index)&set(test.index))
            self.assertEqual(len(train)+len(val)+len(test),len(clean))
            conflict=frame.iloc[[0]].copy(); conflict['Class']=1-conflict.Class
            pd.concat([frame,conflict]).to_csv(path,index=False)
            with self.assertRaises(ValueError): load_data(path)

@unittest.skipUnless(os.getenv('TEST_ARTIFACT_DIR'), 'Set TEST_ARTIFACT_DIR after training')
class ServingTests(unittest.TestCase):
    def test_api_and_saved_scores(self):
        from fastapi.testclient import TestClient
        folder=Path(os.environ['TEST_ARTIFACT_DIR']).resolve()
        os.environ['ARTIFACT_DIR']=str(folder)
        from app import app
        from predict import Predictor
        payload=json.loads((folder/'example_request.json').read_text())
        predictor=Predictor(folder)
        expected=predictor.predict(pd.DataFrame(payload['transactions']))[0]
        with TestClient(app) as client:
            self.assertEqual(client.get('/health').status_code,200)
            response=client.post('/predict',json=payload)
            self.assertEqual(response.status_code,200)
            self.assertAlmostEqual(response.json()['results'][0]['fraud_score'],expected,places=6)
            self.assertEqual(client.post('/predict',json={'transactions':[]}).status_code,422)
            self.assertEqual(client.post('/predict',json={'transactions':[{'Amount':1}]}).status_code,422)
            bad={'transactions':[dict(payload['transactions'][0],Amount=-1)]}
            self.assertEqual(client.post('/predict',json=bad).status_code,422)
            extra={'transactions':[dict(payload['transactions'][0],Class=0)]}
            self.assertEqual(client.post('/predict',json=extra).status_code,422)
            # Column order should never change predictions.
            reversed_row=dict(reversed(list(payload['transactions'][0].items())))
            response=client.post('/predict',json={'transactions':[reversed_row]})
            self.assertAlmostEqual(response.json()['results'][0]['fraud_score'],expected,places=6)

if __name__=='__main__': unittest.main()
