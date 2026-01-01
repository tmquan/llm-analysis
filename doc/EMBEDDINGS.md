# Multi-GPU Parallel Embedding Extraction

## Configuration

```
╭───────────────────────────────────────────────────────────╮
│ Multi-GPU Parallel Embedding Extraction                   │
│ Shard-Based Work Distribution for Maximum GPU Utilization │
╰───────────────────────────────────────────────────────────╯
```

- **Mode:** Extract ALL available dataset splits
- **Note:** Excluding multilingual splits

---

## Dataset Scan Results

### ✅ Successfully Loaded Datasets

| Dataset | Splits Found | Split Names |
|---------|--------------|-------------|
| v1 | 5 | chat, code, math, stem, tool_calling |
| v2 | 4 | stem, chat, math, code |
| llama-sft | 5 | code, math, science, chat, safety |
| v3-science | 2 | MCQ, RQA |
| v3-instruction-chat | 2 | chat_if, structured_outputs |
| v3-math-proofs | 1 | lean |

**Total:** 6 datasets, 19 splits

### ⚠️ Excluded Splits (v2 multilingual)

The following multilingual splits were excluded from v2:
- multilingual_ja
- multilingual_de
- multilingual_it
- multilingual_es
- multilingual_fr

### ❌ Failed to Load

| Dataset | Error |
|---------|-------|
| llama-rl | An error occurred while generating the dataset |
| v3-rl-blend | The read operation timed out |
| v3-agentic | An error occurred while generating the dataset |

---

## Summary

| Status | Count |
|--------|-------|
| ✅ Loaded | 6 datasets |
| ⚠️ Excluded (multilingual) | 5 splits |
| ❌ Failed | 3 datasets |

**Available for embedding extraction:** 19 splits across 6 datasets

