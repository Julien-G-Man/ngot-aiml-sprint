import pytest 
from fastapi.testclient import TestClient 
from unittest.mock import patch, MagicMock 
from app.main import app 
from app.schemas import MedicalAnswer, SourceDocument 
from datetime import datetime 
  
client = TestClient(app) 
  
MOCK_ANSWER = MedicalAnswer( 
    answer='Artemether-Lumefantrine is the first-line treatment.', 
    sources=[SourceDocument(filename='malaria.txt', chunk_text='First-line: AL', relevance_score=0.92)], 
    question='What is first-line malaria treatment?', 
    num_sources_used=1, 
    latency_ms=450.0, 
    model_used='gpt-4o-mini', 
    answer_timestamp=datetime.utcnow(), 
) 
  
  
class TestHealthEndpoint: 
    def test_health_returns_200(self): 
        r = client.get('/health') 
        assert r.status_code == 200 
  
    def test_health_has_required_fields(self): 
        r = client.get('/health') 
        d = r.json() 
        assert 'status' in d 
        assert 'index_loaded' in d 
        assert 'vector_count' in d 
  

class TestAskEndpoint: 
    @patch('app.main.rag_engine') 
    def test_valid_query_returns_200(self, mock_engine): 
        mock_engine.is_loaded = True 
        mock_engine.query.return_value = MOCK_ANSWER 
        r = client.post('/ask', json={'question': 'What is malaria treatment?'}) 
        assert r.status_code == 200 
        data = r.json() 
        assert 'answer' in data 
        assert 'sources' in data 
        assert len(data['sources']) > 0 
  
    def test_short_question_rejected(self): 
        r = client.post('/ask', json={'question': 'hi'}) 
        assert r.status_code == 422   # Question too short (< 10 chars) 
  
    def test_top_k_out_of_range_rejected(self): 
        r = client.post('/ask', json={'question': 'What is malaria treatment?', 'top_k': 99}) 
        assert r.status_code == 422 
  
    @patch('app.main.rag_engine') 
    def test_unloaded_engine_returns_503(self, mock_engine): 
        mock_engine.is_loaded = False 
        r = client.post('/ask', json={'question': 'What is malaria treatment?'}) 
        assert r.status_code == 503