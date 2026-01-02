# Batch Size & Memory Configuration Guide

This guide explains how to choose optimal batch sizes for embedding extraction with `nvidia/llama-embed-nemotron-8b`.

---

## TL;DR — Quick Reference

| Max Text Length | Recommended Batch Size | GPU Memory Required |
|-----------------|------------------------|---------------------|
| 512 | 128-256 | ~20 GB |
| 1024 | 64-128 | ~25 GB |
| 2048 | 64-96 | ~35 GB |
| 4096 | 32-64 | ~45 GB |
| 8192 | 16-32 | ~55 GB |
| 16384 | 8-16 | ~70 GB |
| 32768 | 4-8 | ~90 GB |

**For B300 GPUs (192GB HBM3e):** You can use larger batches than listed above.

---

## Understanding Memory Usage

### Memory Components

When running the embedding model, GPU memory is consumed by:

| Component | Size | Scales With |
|-----------|------|-------------|
| **Model weights** | ~16 GB | Fixed (8B params × 2 bytes fp16) |
| **KV Cache** | Variable | `batch × seq_len × hidden × layers` |
| **Attention scores** | Variable | `batch × heads × seq_len²` ⚠️ |
| **Activations** | Variable | `batch × seq_len × hidden` |
| **Input/Output tensors** | Variable | `batch × seq_len × vocab/hidden` |

### The Quadratic Attention Problem

The **attention mechanism** is the primary memory bottleneck:

```
Memory ∝ batch_size × num_heads × sequence_length²
```

This means doubling sequence length **quadruples** attention memory:

| Sequence Length | Attention Elements (per sample) | Memory (fp16) |
|-----------------|--------------------------------|---------------|
| 512 | 262K | 0.5 MB |
| 2048 | 4M | 8 MB |
| 8192 | 67M | 134 MB |
| 32768 | 1B | 2 GB |

With batch_size=32 at seq_len=32768: **64 GB just for attention!**

---

## Detailed Memory Calculations

### Formula

```
Total GPU Memory ≈ 
    Model Weights (16 GB)
  + Batch × Seq × Hidden × 4 (input/output tensors)
  + Batch × Heads × Seq² × 2 (attention scores)
  + Batch × Seq × Hidden × Layers × 2 (activations)
  + Overhead (~2-4 GB)
```

### Model Specifications (llama-embed-nemotron-8b)

| Parameter | Value |
|-----------|-------|
| Parameters | 8B |
| Hidden size | 4096 |
| Num layers | 32 |
| Num attention heads | 32 |
| Precision | float16 |

### Example Calculations

**Scenario 1: batch=32, seq_len=8192**
```
Model weights:     16 GB
Attention:         32 × 32 × 8192² × 2 bytes = 137 GB  ❌ OOM!
```

**Scenario 2: batch=8, seq_len=8192**
```
Model weights:     16 GB
Attention:         8 × 32 × 8192² × 2 bytes = 34 GB
Activations:       ~15 GB
Total:             ~65 GB ✓
```

**Scenario 3: batch=4, seq_len=32768**
```
Model weights:     16 GB
Attention:         4 × 32 × 32768² × 2 bytes = 275 GB  ❌ OOM!
```

> ⚠️ **Note:** Flash Attention or other optimizations can significantly reduce these numbers, but the quadratic scaling remains.

---

## Recommended Configurations

### For A100 80GB / H100 80GB

```bash
# Short texts (chat, instructions)
python extract_embeddings_parallel_shards.py \
    --splits v1:chat v2:chat \
    --batch-size 64 \
    --max-text-length 4096

# Medium texts (code, math)
python extract_embeddings_parallel_shards.py \
    --splits v1:code v1:math \
    --batch-size 32 \
    --max-text-length 8192

# Long texts (documents, proofs)
python extract_embeddings_parallel_shards.py \
    --splits v3-math-proofs:lean \
    --batch-size 8 \
    --max-text-length 16384
```

### For B300 192GB / GB200 288GB

```bash
# Short texts
python extract_embeddings_parallel_shards.py \
    --splits v1:chat v2:chat \
    --batch-size 128 \
    --max-text-length 4096

# Medium texts
python extract_embeddings_parallel_shards.py \
    --splits v1:code v1:math \
    --batch-size 64 \
    --max-text-length 8192

# Long texts
python extract_embeddings_parallel_shards.py \
    --splits v3-math-proofs:lean \
    --batch-size 16 \
    --max-text-length 32768
```

---

## Category-Specific Recommendations

### General Splits (Chat, STEM, Safety)

These typically have shorter texts (< 4K tokens on average):

```bash
python extract_embeddings_parallel_shards.py \
    --splits \
    v1:chat v1:stem v1:tool_calling \
    v2:stem v2:chat \
    llama-sft:science llama-sft:chat llama-sft:safety \
    v3-science:MCQ v3-science:RQA \
    v3-instruction-chat:chat_if v3-instruction-chat:structured_outputs \
    --batch-size 80 \
    --max-text-length 8192 \
    --num-gpus 8
```

### Code Splits

Code can be longer; use moderate settings:

```bash
python extract_embeddings_parallel_shards.py \
    --splits v1:code v2:code llama-sft:code \
    --batch-size 48 \
    --max-text-length 16384 \
    --num-gpus 8
```

### Math Splits

Math proofs (especially Lean) can be very long:

```bash
python extract_embeddings_parallel_shards.py \
    --splits v1:math v2:math llama-sft:math v3-math-proofs:lean \
    --batch-size 32 \
    --max-text-length 16384 \
    --num-gpus 8
```

---

## Troubleshooting OOM Errors

### Symptoms

```
CUDA out of memory. Tried to allocate X GiB
torch.cuda.OutOfMemoryError: CUDA out of memory
RuntimeError: CUDA error: out of memory
```

### Solutions

1. **Reduce batch size first** (most effective)
   ```bash
   --batch-size 16  # instead of 64
   ```

2. **Reduce max text length** (if acceptable)
   ```bash
   --max-text-length 8192  # instead of 32768
   ```

3. **Use fewer GPUs per model** (if multi-GPU inference)
   ```bash
   --num-gpus 4  # instead of 8
   ```

4. **Clear GPU cache before running**
   ```bash
   # Kill any processes using GPUs
   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}
   
   # Or use fuser
   sudo fuser -k /dev/nvidia*
   ```

5. **Monitor memory during extraction**
   ```bash
   watch -n 1 nvidia-smi
   ```

### Finding Optimal Batch Size

Start small and increase until you hit ~80% memory usage:

```bash
# Start with small batch
python extract_embeddings_parallel_shards.py --splits v1:chat --batch-size 8 --dry-run

# Monitor memory usage
watch -n 1 nvidia-smi

# Gradually increase: 8 → 16 → 32 → 64 → ...
```

---

## Performance Tips

### Maximize Throughput

1. **Use the largest batch that fits in memory**
   - Larger batches = better GPU utilization
   - Target 80-90% memory usage

2. **Match batch size to your text lengths**
   - Short texts → large batch
   - Long texts → small batch

3. **Use all available GPUs**
   ```bash
   --num-gpus 8  # Use all 8 GPUs
   ```

### Typical Throughput (B300 192GB)

| Configuration | Throughput |
|---------------|------------|
| batch=128, seq=2048 | ~500 samples/s |
| batch=64, seq=4096 | ~300 samples/s |
| batch=32, seq=8192 | ~150 samples/s |
| batch=16, seq=16384 | ~50 samples/s |
| batch=8, seq=32768 | ~15 samples/s |

---

## Memory Optimization Techniques

### Gradient Checkpointing (Not applicable for inference)

For training, this helps. For inference (embedding extraction), it's not used.

### Flash Attention

The model may use Flash Attention if available, which reduces memory from O(n²) to O(n):

```python
# Check if Flash Attention is available
python -c "from flash_attn import flash_attn_func; print('Flash Attention available')"
```

### Mixed Precision

The script uses fp16 by default:
```python
model = AutoModel.from_pretrained(..., torch_dtype=torch.float16)
```

For newer GPUs (A100+), bf16 may be more stable:
```python
model = AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)
```

---

## Quick Decision Tree

```
Q: What's your max text length?

├─ ≤ 2048 tokens
│   └─ Use batch_size 64-128
│
├─ 2048-8192 tokens
│   └─ Use batch_size 32-64
│
├─ 8192-16384 tokens
│   └─ Use batch_size 16-32
│
└─ 16384+ tokens
    └─ Use batch_size 4-16
```

---

## See Also

- [README.md](../README.md) — Main documentation
- [TROUBLESHOOT.md](TROUBLESHOOT.md) — GPU troubleshooting guide
- [extract_embeddings_parallel_shards.py](../extract_embeddings_parallel_shards.py) — Main extraction script

