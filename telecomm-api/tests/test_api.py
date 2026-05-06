# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from ..app.main import app
from ..app.schemas import ComplaintAnalysis

client = TestClient(app)

MOCK_ANALYSIS = ComplaintAnalysis(
    category='network_outage', priority='high', sentiment_score=-0.8,
    account_at_risk=True, key_issue='No signal for 3 days',
    suggested_resolution='Run line diagnostic, offer credit',
    requires_escalation=True,
)

@patch('app.main.analyse_complaint', return_value=MOCK_ANALYSIS)
def test_valid_complaint_returns_200(mock):
    r = client.post('/analyse', json={
        'complaint_text': 'No signal for 3 days, absolutely furious, cancelling.',
        'customer_id': 'C001',
    })
    assert r.status_code == 200
    assert r.json()['analysis']['category'] == 'network_outage'
    assert r.json()['analysis']['account_at_risk'] is True

def test_short_complaint_rejected():
    r = client.post('/analyse', json={'complaint_text': 'bad', 'customer_id':
'C001'})
    assert r.status_code == 422

def test_health_returns_ok():
    assert client.get('/health').json()['status'] == 'healthy'
