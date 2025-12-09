#!/usr/bin/env python3
"""
CUDA Diagnostic Script
Tests CUDA availability and initialization to help debug issues.
"""

import sys

print("=" * 80)
print("🔍 CUDA Diagnostic Test")
print("=" * 80)

# Test 1: Import torch
print("\n1️⃣  Testing PyTorch import...")
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"   ❌ Failed to import torch: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n2️⃣  Checking CUDA availability...")
try:
    cuda_available = torch.cuda.is_available()
    print(f"   torch.cuda.is_available(): {cuda_available}")
    if not cuda_available:
        print("   ⚠️  CUDA is not available!")
        print("   This could mean:")
        print("   - No CUDA-capable GPU")
        print("   - CUDA drivers not installed")
        print("   - PyTorch built without CUDA support")
        sys.exit(0)
except Exception as e:
    print(f"   ❌ Error checking CUDA: {e}")
    sys.exit(1)

# Test 3: CUDA device count
print("\n3️⃣  Checking CUDA device count...")
try:
    device_count = torch.cuda.device_count()
    print(f"   Device count: {device_count}")
except Exception as e:
    print(f"   ❌ Error getting device count: {e}")
    sys.exit(1)

# Test 4: Get device properties
print("\n4️⃣  Getting device properties...")
try:
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"      Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"      Compute Capability: {props.major}.{props.minor}")
except Exception as e:
    print(f"   ❌ Error getting device properties: {e}")

# Test 5: Create tensor on GPU
print("\n5️⃣  Testing tensor creation on GPU...")
try:
    device = torch.device('cuda:0')
    x = torch.ones(10, 10, device=device)
    print(f"   ✅ Created tensor on GPU: shape {x.shape}")
    del x
except Exception as e:
    print(f"   ❌ Failed to create tensor on GPU: {e}")
    sys.exit(1)

# Test 6: Test CUDA operations
print("\n6️⃣  Testing CUDA operations...")
try:
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"   ✅ Matrix multiplication successful: {c.shape}")
    del a, b, c
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ❌ CUDA operation failed: {e}")
    sys.exit(1)

# Test 7: Check CUDA version
print("\n7️⃣  CUDA version info...")
try:
    print(f"   CUDA version (PyTorch): {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
except Exception as e:
    print(f"   ⚠️  Could not get CUDA version info: {e}")

# Test 8: Test sentence-transformers if available
print("\n8️⃣  Testing sentence-transformers with CUDA...")
try:
    from sentence_transformers import SentenceTransformer
    print(f"   ✅ sentence-transformers imported successfully")
    
    # Try to load a small model on GPU
    print("   Loading a test model (this may take a moment)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    print(f"   ✅ Model loaded on GPU successfully!")
    
    # Test embedding
    test_text = ["Hello world"]
    embedding = model.encode(test_text)
    print(f"   ✅ Embedding generated: dimension {len(embedding[0])}")
    
    del model
    torch.cuda.empty_cache()
    
except ImportError:
    print(f"   ⚠️  sentence-transformers not installed")
except Exception as e:
    print(f"   ❌ sentence-transformers test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ CUDA diagnostic completed!")
print("=" * 80)

