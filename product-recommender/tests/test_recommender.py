import pytest 
from fastapi.testclient import TestClient 
from unittest.mock import patch, MagicMock 
from app.main import app 
from app.schemas import RecommendResponse, ProductRecommendation 
  
client = TestClient(app) 
  
MOCK_REC = ProductRecommendation( 
    product_id='P002', name='Rain Poncho', description='Waterproof poncho', 
    category='Outdoor', price_ghs=45.0, similarity_score=0.89, rank=1, 
) 
MOCK_RESPONSE = RecommendResponse( 
    recommendations=[MOCK_REC], query='rainy day gear', 
    total_returned=1, model_used='text-embedding-3-small', 
) 
  
  
class TestHealthEndpoint: 
    def test_health_returns_200(self): 
        assert client.get('/health').status_code == 200 
  
  
class TestRecommendEndpoint: 
    @patch('app.main.recommender') 
    def test_valid_query_returns_200(self, mock_rec): 
        mock_rec.is_ready = True 
        mock_rec.recommend.return_value = MOCK_RESPONSE 
        r = client.post('/recommend', json={'query': 'rainy day gear'}) 
        assert r.status_code == 200 
        data = r.json() 
        assert 'recommendations' in data 
        assert len(data['recommendations']) > 0 
        assert 'similarity_score' in data['recommendations'][0] 
  
    def test_short_query_rejected(self): 
        r = client.post('/recommend', json={'query': 'hi'}) 
        assert r.status_code == 422 
  
    def test_top_k_too_large_rejected(self): 
        r = client.post('/recommend', json={'query': 'nice clothes', 'top_k': 999}) 
        assert r.status_code == 422 

    @patch('app.main.recommender') 
    def test_unloaded_recommender_returns_503(self, mock_rec): 
        mock_rec.is_ready = False 
        r = client.post('/recommend', json={'query': 'nice clothes'}) 
        assert r.status_code == 503