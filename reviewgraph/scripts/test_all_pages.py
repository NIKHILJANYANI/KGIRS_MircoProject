import urllib.request
import json

base = 'http://127.0.0.1:8000/api'

print('=== 1. HEALTH ===')
with urllib.request.urlopen(f'{base}/health') as r:
    print('Status:', r.status, json.loads(r.read().decode())['status'])

print('\n=== 2. SEARCH ALL 4 MODES ===')
for mode in ['keyword', 'semantic', 'graph', 'hybrid']:
    req = urllib.request.Request(
        f'{base}/search',
        data=json.dumps({'query': 'best quality product', 'mode': mode, 'page_size': 5}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as r:
        res = json.loads(r.read().decode())
        count = len(res.get('results', []))
        lat = res.get('latency_ms')
        print(f"Mode [{mode}]: {count} results, latency: {lat}ms")
        if res.get('results'):
            top = res['results'][0]
            print(f"   Top 1: {top['title']} (id={top['id']}) score={top['score']}")

print('\n=== 3. INVESTIGATION ===')
with urllib.request.urlopen(f'{base}/products/P001/investigation') as r:
    inv = json.loads(r.read().decode())
    print('Product P001 suspiciousness score:', inv.get('suspiciousness', {}).get('score'))
    print('Total reviews:', inv.get('total_reviews'))
    summary = inv.get('explanation', {}).get('explanation_summary') or ''
    print('Explanation summary:', summary[:90] + '...')

print('\n=== 4. KNOWLEDGE GRAPH ===')
with urllib.request.urlopen(f'{base}/products/P001/graph') as r:
    graph = json.loads(r.read().decode())
    print('Graph P001 nodes:', graph.get('node_count'), 'edges:', graph.get('edge_count'))

print('\n=== 5. REVIEW DETAILS (SYNTH_R071 - Fake Review) ===')
with urllib.request.urlopen(f'{base}/reviews/SYNTH_R071') as r:
    rev = json.loads(r.read().decode())
    print('Review ID:', rev['review']['review_id'])
    print('Rating:', rev['review']['rating'])
    print('Predicted class:', rev['review']['predicted_class'])
    print('Model score:', rev['review']['model_score'])
    print('Live prediction:', rev['live_prediction'])
    print('Signals count:', len(rev['signals']))

print('\n=== 6. REVIEW DETAILS (SYNTH_R001 - Genuine Review) ===')
with urllib.request.urlopen(f'{base}/reviews/SYNTH_R001') as r:
    rev = json.loads(r.read().decode())
    print('Review ID:', rev['review']['review_id'])
    print('Rating:', rev['review']['rating'])
    print('Predicted class:', rev['review']['predicted_class'])
    print('Model score:', rev['review']['model_score'])
    print('Signals count:', len(rev['signals']))
