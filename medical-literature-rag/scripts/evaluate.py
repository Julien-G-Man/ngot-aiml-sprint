# Evaluate RAG quality with a test set 
  
import json 
import time 
import httpx 
from pathlib import Path 
  
API_URL = 'http://localhost:8001' 

# Evaluation test set: question + expected keywords in answer 
TEST_SET = [ 
    { 
        'question': 'What is the first-line treatment for uncomplicated malaria in Ghana?', 
        'expected_keywords': ['artemether', 'lumefantrine', 'AL', 'coartem'], 
        'category': 'factual', 
    }, 
    { 
        'question': 'What blood pressure should we target in elderly patients over 65?', 
        'expected_keywords': ['140/90', '65', 'elderly'], 
        'category': 'factual', 
    }, 
    { 
        'question': 'How often should HbA1c be measured when diabetes is stable?', 
        'expected_keywords': ['6 months', 'stable', 'every'], 
        'category': 'factual', 
    }, 
    { 
        'question': 'What drug is used for malaria in the first trimester of pregnancy?', 
        'expected_keywords': ['quinine', 'first trimester', 'pregnancy'], 
        'category': 'factual', 
    }, 
    { 
        'question': 'Why should malaria medication be taken with food?', 
        'expected_keywords': ['fatty', 'absorption', 'food'], 
        'category': 'reasoning', 
    }, 
] 
  
results = [] 
print('Running RAG evaluation...') 
print('='*60) 
  
for i, test in enumerate(TEST_SET): 
    start = time.time() 
    resp = httpx.post( 
        f'{API_URL}/ask', 
        json={'question': test['question'], 'top_k': 3}, 
        timeout=30, 
    ) 
    latency = (time.time() - start) * 1000 
    data = resp.json() 
  
    answer_lower = data['answer'].lower() 
    # Simple keyword-based faithfulness check 
    hits = sum(1 for kw in test['expected_keywords'] if kw.lower() in answer_lower) 
    precision = hits / len(test['expected_keywords']) 
  
    result = { 
        'question':   test['question'][:50] + '...', 
        'category':   test['category'], 
        'precision':  round(precision, 2), 
        'latency_ms': round(latency, 0), 
        'num_sources': data['num_sources_used'], 
        'passed':     precision >= 0.5, 
    } 
    results.append(result) 
    status_icon = '✅' if result['passed'] else '❌' 
    print(f'{status_icon} [{i+1}] precision={precision:.0%}  latency={latency:.0f}ms  Q: {test["question"][:50]}') 
  
avg_precision = sum(r['precision'] for r in results) / len(results) 
avg_latency   = sum(r['latency_ms'] for r in results) / len(results) 
pass_rate     = sum(1 for r in results if r['passed']) / len(results) 
  
print('='*60) 
print(f'Average Precision:  {avg_precision:.0%}') 
print(f'Average Latency:    {avg_latency:.0f} ms') 
print(f'Pass Rate:          {pass_rate:.0%} ({sum(1 for r in results if 
r["passed"])}/{len(results)} tests)') 
  
Path('metrics').mkdir(exist_ok=True) 
with open('metrics/rag_evaluation.json', 'w') as f: 
    json.dump({'results': results, 'summary': {'avg_precision': avg_precision, 
'avg_latency_ms': avg_latency, 'pass_rate': pass_rate}}, f, indent=2) 
print('Results saved to metrics/rag_evaluation.json')