# NVIDIA Nemotron Datasets Summary

**Download Location:** `/raid/datasets`

## Download Results

📊 **Results: 6 successful, 5 failed**

---

## ✅ Successfully Downloaded Datasets

### Nemotron v1
- **Total samples:** 25,659,642
- **Splits:**
  | Split | Samples |
  |-------|---------|
  | chat | 746,622 |
  | code | 1,896,395 |
  | math | 2,044,407 |
  | stem | 20,662,167 |
  | tool_calling | 310,051 |

### Nemotron v2
- **Total samples:** 6,341,414
- **Splits:**
  | Split | Samples |
  |-------|---------|
  | stem | 355,000 |
  | chat | 627,720 |
  | math | 239,467 |
  | code | 175,000 |
  | multilingual_ja | 975,202 |
  | multilingual_de | 1,015,314 |
  | multilingual_it | 1,016,503 |
  | multilingual_es | 935,704 |
  | multilingual_fr | 1,001,504 |

### Llama-Nemotron SFT
- **Total samples:** 32,955,418
- **Splits:**
  | Split | Samples |
  |-------|---------|
  | code | 10,108,883 |
  | math | 22,066,397 |
  | science | 708,920 |
  | chat | 39,792 |
  | safety | 31,426 |

### v3-science (Nemotron-Science-v1)
- **Total samples:** 226,334
- **Splits:**
  | Split | Samples |
  |-------|---------|
  | MCQ | 174,155 |
  | RQA | 52,179 |

### v3-instruction-chat (Nemotron-Instruction-Following-Chat-v1)
- **Total samples:** 430,978
- **Splits:**
  | Split | Samples |
  |-------|---------|
  | chat_if | 426,009 |
  | structured_outputs | 4,969 |

### v3-math-proofs (Nemotron-Math-Proofs-v1)
- **Total samples:** 1,376,663
- **Splits:**
  | Split | Samples |
  |-------|---------|
  | lean | 1,376,663 |

---

## ❌ Failed Downloads (Data Corruption Issues)

The following datasets have data corruption issues in the HuggingFace repository:

| Dataset | HuggingFace ID | Issue |
|---------|----------------|-------|
| Llama-Nemotron RL | `nvidia/Llama-Nemotron-Post-Training-Dataset` (RL config) | ArrowInvalid error |
| v3-rl-blend | `nvidia/Nemotron-3-Nano-RL-Training-Blend` | ArrowInvalid error |
| v3-agentic | `nvidia/Nemotron-Agentic-v1` | Schema casting error |
| v3-competitive-programming | `nvidia/Nemotron-Competitive-Programming-v1` | Data generation error |
| v3-math | `nvidia/Nemotron-Math-v2` | Data generation error |

**Note:** These errors are caused by corrupted parquet files or schema mismatches on NVIDIA's HuggingFace repository. Report issues at: https://huggingface.co/nvidia

---

## Total Statistics

| Category | Datasets | Total Samples |
|----------|----------|---------------|
| Successfully Downloaded | 6 | ~67 million |
| Failed | 5 | N/A |

**Grand Total (downloaded):** ~67,000,000 samples

