# app/main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import RecommendRequest, RecommendResponse
from app.recommender import ProductRecommender

app = FastAPI(
    title="Product Recommender API",
    description="Semantic product recommendations powered by vector similarity search.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
allow_headers=['*'])

recommender = ProductRecommender()


@app.on_event('startup')
async def startup():
    recommender.load()


@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Product Recommender API'}


@app.get('/health')
async def health():
    return {'status': 'healthy' if recommender.is_ready else 'degraded', 'api_version': '1.0.0'}


@app.post('/recommend', response_model=RecommendResponse, tags=['Recommendations'])
async def recommend(request: RecommendRequest):
    """
    Find products that best match a customer query using semantic similarity.
    Works for natural language queries like 'something cozy for rainy weather'.
    """
    if not recommender.is_ready:
        raise HTTPException(status_code=503, detail='Recommender not loaded.')
    try:
        return recommender.recommend(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
