import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import pytest
import backend.main as main

client = TestClient(main.app)


def test_health_endpoint():
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'ok'}


def test_chat_endpoint(monkeypatch):
    # Fake similar prompts returned by Qdrant
    fake_prompts = [f'prompt {i}' for i in range(5)]

    def fake_retrieve(text: str, k: int):
        return fake_prompts[:k]

    captured = []

    def fake_call(prompt: str) -> str:
        captured.append(prompt)
        return 'dummy'

    monkeypatch.setattr(main, 'retrieve_similar', fake_retrieve)
    monkeypatch.setattr(main, 'call_ollama', fake_call)
    monkeypatch.setattr(main.client, 'collection_exists', lambda *_: True)

    payload = {'prompt': 'How do I bake bread?', 'top_k': 5}
    resp = client.post('/chat', json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data['similar'] == fake_prompts
    assert data['original_answer'] == 'dummy'
    assert data['final_answer'] == 'dummy'

    # first call should be original prompt
    assert captured[0] == payload['prompt']
    # second call should include merged context
    assert 'How do I bake bread?' in captured[1]
    for p in fake_prompts:
        assert p in captured[1]

