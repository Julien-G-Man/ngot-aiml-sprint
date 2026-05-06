# app/main.py
import time, logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import AnalyseRequest, AnalyseResponse
from .llm_service import analyse_complaint

logging.basicConfig(level=logging.INFO)
app = FastAPI(title='Telecomm Complaint Analyser', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.get('/health')
async def health():
    return {'status': 'healthy', 'version': '1.0.0'}

@app.post('/analyse', response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest):
    t0 = time.time()
    try:
        analysis = analyse_complaint(req.complaint_text)
        return AnalyseResponse(
            customer_id=req.customer_id, analysis=analysis,
            model_used='gpt-4o-mini + pydantic-function-calling',
            latency_ms=round((time.time()-t0)*1000, 1),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
