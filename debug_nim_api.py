#!/usr/bin/env python3
"""
Debug script to test the NIM embedding API and diagnose issues.
"""

import requests
import json
import sys
import os

# API Configuration
LOCAL_API_URL = "http://localhost:8000/v1/embeddings"
CLOUD_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"
MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
EXPECTED_EMBEDDING_DIM = 2048  # Expected embedding dimension

def test_local_api():
    """Test the local NIM API with various payloads."""
    print("=" * 80)
    print("🧪 Testing Local NIM API")
    print("=" * 80)
    print(f"\nEndpoint: {LOCAL_API_URL}")
    print(f"Model: {MODEL}")
    print(f"Expected Embedding Dimension: {EXPECTED_EMBEDDING_DIM}")
    print()
    
    # Test 1: Minimal payload
    print("\n📝 Test 1: Minimal payload with input_type")
    payload1 = {
        "input": ["test"],
        "model": MODEL,
        "input_type": "query"
    }
    print(f"Payload: {json.dumps(payload1, indent=2)}")
    
    try:
        response = requests.post(
            LOCAL_API_URL,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=payload1,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                emb = data['data'][0].get('embedding', [])
                dim_match = "✅" if len(emb) == EXPECTED_EMBEDDING_DIM else "⚠️"
                print(f"✅ Success! Embedding dimension: {len(emb)} {dim_match}")
                if len(emb) != EXPECTED_EMBEDDING_DIM:
                    print(f"   ⚠️  Warning: Expected {EXPECTED_EMBEDDING_DIM} dimensions")
            else:
                print(f"⚠️  Unexpected response format")
        else:
            print(f"❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Without input_type
    print("\n📝 Test 2: Payload without input_type")
    payload2 = {
        "input": ["test"],
        "model": MODEL
    }
    print(f"Payload: {json.dumps(payload2, indent=2)}")
    
    try:
        response = requests.post(
            LOCAL_API_URL,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=payload2,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                emb = data['data'][0].get('embedding', [])
                dim_match = "✅" if len(emb) == EXPECTED_EMBEDDING_DIM else "⚠️"
                print(f"✅ Success! Embedding dimension: {len(emb)} {dim_match}")
                if len(emb) != EXPECTED_EMBEDDING_DIM:
                    print(f"   ⚠️  Warning: Expected {EXPECTED_EMBEDDING_DIM} dimensions")
            else:
                print(f"⚠️  Unexpected response format")
        else:
            print(f"❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Check if API is OpenAI-compatible
    print("\n📝 Test 3: OpenAI-compatible format")
    payload3 = {
        "input": "test",  # Single string instead of list
        "model": MODEL
    }
    print(f"Payload: {json.dumps(payload3, indent=2)}")
    
    try:
        response = requests.post(
            LOCAL_API_URL,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=payload3,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                emb = data['data'][0].get('embedding', [])
                dim_match = "✅" if len(emb) == EXPECTED_EMBEDDING_DIM else "⚠️"
                print(f"✅ Success! Embedding dimension: {len(emb)} {dim_match}")
                if len(emb) != EXPECTED_EMBEDDING_DIM:
                    print(f"   ⚠️  Warning: Expected {EXPECTED_EMBEDDING_DIM} dimensions")
            else:
                print(f"⚠️  Unexpected response format")
        else:
            print(f"❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Batch request
    print("\n📝 Test 4: Batch request")
    payload4 = {
        "input": ["test 1", "test 2", "test 3"],
        "model": MODEL,
        "input_type": "passage"
    }
    print(f"Payload: {json.dumps(payload4, indent=2)}")
    
    try:
        response = requests.post(
            LOCAL_API_URL,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            json=payload4,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                print(f"✅ Success! Got {len(data['data'])} embeddings")
                if len(data['data']) > 0:
                    emb = data['data'][0].get('embedding', [])
                    dim_match = "✅" if len(emb) == EXPECTED_EMBEDDING_DIM else "⚠️"
                    print(f"   Embedding dimension: {len(emb)} {dim_match}")
                    if len(emb) != EXPECTED_EMBEDDING_DIM:
                        print(f"   ⚠️  Warning: Expected {EXPECTED_EMBEDDING_DIM} dimensions")
            else:
                print(f"⚠️  Unexpected response format")
        else:
            print(f"❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_local_api()

