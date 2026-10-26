"""Local portfolio prediction API. Run: python -m uvicorn app:app"""
from contextlib import asynccontextmanager
import os
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from predict import Predictor

@asynccontextmanager
async def lifespan(app):
    # Fail at startup if the artifacts are missing or incompatible.
    app.state.predictor=Predictor(os.getenv('ARTIFACT_DIR','artifacts'))
    yield

app=FastAPI(title='Fraud Detection Portfolio API', lifespan=lifespan)
class Request(BaseModel):
    model_config=ConfigDict(extra='forbid')
    transactions:list[dict[str,float]]=Field(min_length=1,max_length=1000)

@app.get('/health')
def health():
    return {'status':'ready','model':app.state.predictor.name}

@app.get('/schema')
def schema():
    return {'features':app.state.predictor.features,
            'note':'V1–V28 are precomputed dataset PCA features; raw card details are not accepted.'}

@app.post('/predict')
def predict(body:Request):
    predictor=app.state.predictor
    begin=time.perf_counter()
    if any(set(row)!=set(predictor.features) for row in body.transactions):
        raise HTTPException(status_code=422,detail='Every transaction must have exactly the /schema features.')
    try:
        scores=predictor.predict(pd.DataFrame(body.transactions))
    except (ValueError,TypeError) as e:
        raise HTTPException(status_code=422,detail=str(e)) from e
    return {'model':predictor.name,'threshold':predictor.threshold,
            'results':[{'fraud_score':float(p),'flagged':bool(p>=predictor.threshold)} for p in scores],
            'inference_ms':(time.perf_counter()-begin)*1000,
            'note':'Scores are uncalibrated; this is a demonstration, not a payment decision service.'}
