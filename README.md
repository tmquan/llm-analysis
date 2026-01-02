# LLM Analysis Toolkit

A comprehensive toolkit for downloading, processing, and analyzing NVIDIA's Nemotron post-training datasets, with multi-GPU embedding extraction and visualization capabilities.

---

## Overview

This repository provides tools for:

- **📥 Dataset Management** — Download and manage NVIDIA Nemotron post-training datasets (v1, v2, v3, Llama-SFT, Llama-RL)
- **🔮 Embedding Extraction** — Multi-GPU parallel embedding extraction using `nvidia/llama-embed-nemotron-8b`
- **✅ Validation** — Verify extracted embeddings against source datasets
- **📊 Visualization** — UMAP dimensionality reduction and interactive Plotly visualizations
- **📄 PDF Processing** — Download NVIDIA GPU datasheets and convert to PNG/JSON/XML formats

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n nemotron python=3.12
conda activate nemotron

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install -r requirements.txt
```

### 2. Authentication

```bash
# HuggingFace (for gated datasets)
huggingface-cli login
# Or: export HF_TOKEN=hf_xxxxx

# NVIDIA NGC API (for NIM embeddings)
export NGC_API_KEY=nvapi-xxxxx

# NVIDIA Build API (for PDF parsing)
export NVIDIA_API_KEY=nvapi-xxxxx

# Adobe PDF Services (for PDF to XML)
export ADOBE_CLIENT_ID=xxxxx
export ADOBE_CLIENT_SECRET=xxxxx
```

### 3. Download Datasets

```bash
# Download all datasets
python download_nemotron_datasets.py --all

# Or download specific versions
python download_nemotron_datasets.py --v1 --v2
python download_nemotron_datasets.py --llama-sft
python download_nemotron_datasets.py --v3
```

### 4. Extract Embeddings

```bash
# Extract embeddings from all splits using 8 GPUs
python extract_embeddings_parallel_shards.py --all --num-gpus 8

# Or specific splits
python extract_embeddings_parallel_shards.py --splits v1:chat v2:math --num-gpus 4
```

### 5. Validate & Visualize

```bash
# Validate embeddings
python validate_embeddings.py --compare-source

# Visualize with UMAP
python visualize_embeddings.py embeddings.json --interactive umap.html
```

---

## Available Datasets

### Nemotron v1
| Split | Samples |
|-------|---------|
| chat | 746,622 |
| code | 1,896,395 |
| math | 2,044,407 |
| stem | 20,662,167 |
| tool_calling | 310,051 |
| **Total** | **25,659,642** |

### Nemotron v2
| Split | Samples |
|-------|---------|
| stem | 355,000 |
| chat | 627,720 |
| math | 239,467 |
| code | 175,000 |
| multilingual_* | 4,944,227 |
| **Total** | **6,341,414** |

### Llama-Nemotron SFT
| Split | Samples |
|-------|---------|
| code | 10,108,883 |
| math | 22,066,397 |
| science | 708,920 |
| chat | 39,792 |
| safety | 31,426 |
| **Total** | **32,955,418** |

### Nemotron v3 Collection
- `nvidia/Nemotron-3-Nano-RL-Training-Blend`
- `nvidia/Nemotron-Science-v1`
- `nvidia/Nemotron-Instruction-Following-Chat-v1`
- `nvidia/Nemotron-Math-Proofs-v1`
- `nvidia/Nemotron-Agentic-v1`
- `nvidia/Nemotron-Competitive-Programming-v1`
- `nvidia/Nemotron-Math-v2`

---

## Scripts Reference

### Dataset Management

| Script | Description |
|--------|-------------|
| `download_nemotron_datasets.py` | Download Nemotron datasets from HuggingFace |
| `download_nemotron_datasets.ipynb` | Interactive notebook for dataset exploration |

```bash
# Download all datasets
python download_nemotron_datasets.py --all

# Download specific datasets
python download_nemotron_datasets.py --v1 --v2 --llama-sft

# Custom directories
python download_nemotron_datasets.py \
  --datasets-dir /raid/datasets \
  --checkpoints-dir /raid/checkpoints
```

### Embedding Extraction

| Script | Description |
|--------|-------------|
| `extract_embeddings_parallel_shards.py` | Multi-GPU parallel embedding extraction |

**Features:**
- Shard-based work distribution for optimal GPU utilization
- Automatic dataset shard discovery
- Checkpointing to avoid reprocessing
- Progress tracking with Rich UI
- Memory-efficient batch processing

```bash
# Extract all available splits
python extract_embeddings_parallel_shards.py --all --num-gpus 8

# Extract specific splits
python extract_embeddings_parallel_shards.py \
  --splits v1:chat v2:math llama-sft:safety \
  --num-gpus 4 \
  --batch-size 32 \
  --max-text-length 8192

# Custom directories
python extract_embeddings_parallel_shards.py --all \
  --datasets-dir /data/datasets \
  --checkpoints-dir /data/checkpoints \
  --embeddings-dir /data/embeddings
```

#### Batch Processing by Category

For large-scale processing, you may want to extract embeddings by category to manage resources and prioritize certain data types.

**Split Categories:**

| Category | Splits |
|----------|--------|
| **General** | `v1:chat`, `v1:stem`, `v1:tool_calling`, `v2:stem`, `v2:chat`, `llama-sft:science`, `llama-sft:chat`, `llama-sft:safety`, `v3-science:MCQ`, `v3-science:RQA`, `v3-instruction-chat:chat_if`, `v3-instruction-chat:structured_outputs` |
| **Code** | `v1:code`, `v2:code`, `llama-sft:code` |
| **Math** | `v1:math`, `v2:math`, `llama-sft:math`, `v3-math-proofs:lean` |

**1. General splits (without code and math):**

```bash
python extract_embeddings_parallel_shards.py \
    --splits \
    v1:chat v1:stem v1:tool_calling \
    v2:stem v2:chat \
    llama-sft:science llama-sft:chat llama-sft:safety \
    v3-science:MCQ v3-science:RQA \
    v3-instruction-chat:chat_if v3-instruction-chat:structured_outputs \
    --skip-validation \
    --batch-size 80 \
    --max-text-length 32768 \
    --num-gpus 8 \
    --datasets-dir /raid/datasets \
    --checkpoints-dir /raid/checkpoints \
    --embeddings-dir /raid/embeddings
```

**2. Code only:**

```bash
python extract_embeddings_parallel_shards.py \
    --splits \
    v1:code \
    v2:code \
    llama-sft:code \
    --skip-validation \
    --batch-size 80 \
    --max-text-length 32768 \
    --num-gpus 8 \
    --datasets-dir /raid/datasets \
    --checkpoints-dir /raid/checkpoints \
    --embeddings-dir /raid/embeddings
```

**3. Math only:**

```bash
python extract_embeddings_parallel_shards.py \
    --splits \
    v1:math \
    v2:math \
    llama-sft:math \
    v3-math-proofs:lean \
    --skip-validation \
    --batch-size 80 \
    --max-text-length 32768 \
    --num-gpus 8 \
    --datasets-dir /raid/datasets \
    --checkpoints-dir /raid/checkpoints \
    --embeddings-dir /raid/embeddings
```

**Output Format:** Parquet files organized by dataset/split:
```
embeddings_output/
├── v1/
│   ├── chat/
│   │   ├── v1-chat-00000-of-00001.parquet
│   │   └── ...
│   ├── code/
│   └── ...
├── v2/
└── llama-sft/
```

### Validation

| Script | Description |
|--------|-------------|
| `validate_embeddings.py` | Validate generated embeddings |

```bash
# Basic validation
python validate_embeddings.py

# Compare with source datasets
python validate_embeddings.py --compare-source

# Verbose output with sample data
python validate_embeddings.py --verbose --check-nans
```

### Visualization

| Script | Description |
|--------|-------------|
| `visualize_embeddings.py` | UMAP visualization for embeddings |
| `visualize_nemotron_datasets.ipynb` | Interactive visualization notebook |

```bash
# Create static plot
python visualize_embeddings.py embeddings.json --output umap.png

# Create interactive HTML
python visualize_embeddings.py embeddings.json --interactive umap.html

# Customize UMAP parameters
python visualize_embeddings.py embeddings.json \
  --n-neighbors 30 \
  --min-dist 0.05 \
  --output umap_detailed.png
```

### PDF Processing

| Script | Description |
|--------|-------------|
| `download_gpu_datasheets.py` | Download NVIDIA GPU datasheets |
| `pdf_to_png.py` | Convert PDF pages to PNG images |
| `pdf_to_json.py` | Extract structured content using NVIDIA Nemotron Parse API |
| `pdf_to_xml.py` | Extract content using Adobe PDF Services API |

```bash
# Download GPU datasheets (A100, H100, H200, GH200, B200, GB200)
python download_gpu_datasheets.py

# Convert PDFs to PNGs (300 DPI)
python pdf_to_png.py

# Extract layout with NVIDIA API (requires NVIDIA_API_KEY)
python pdf_to_json.py

# Extract structure with Adobe API (requires Adobe credentials)
python pdf_to_xml.py
```

**Supported GPUs:**
- A100 80GB (SXM/PCIe)
- H100 80GB (SXM/NVL)
- H200 141GB (SXM/NVL)
- GH200 Grace Hopper Superchip
- B200/B300 Blackwell
- GB200/GB300 Blackwell Ultra

---

## Directory Structure

```
llm-analysis/
├── datasets/                   # Downloaded HuggingFace datasets
│   ├── nemotron-v1/
│   ├── nemotron-v2/
│   ├── nemotron-v3/
│   └── llama-nemotron/
├── checkpoints/                # HuggingFace cache and model checkpoints
├── embeddings_output/          # Extracted embeddings (parquet)
├── data/
│   ├── pdf/                    # Downloaded GPU datasheets
│   ├── png/                    # Converted PNG images
│   ├── json/                   # Nemotron Parse API output
│   └── xml/                    # Adobe PDF Services output
├── download_nemotron_datasets.py
├── download_nemotron_datasets.ipynb
├── extract_embeddings_parallel_shards.py
├── validate_embeddings.py
├── visualize_embeddings.py
├── visualize_nemotron_datasets.ipynb
├── download_gpu_datasheets.py
├── pdf_to_png.py
├── pdf_to_json.py
├── pdf_to_xml.py
├── requirements.txt
└── README.md
```

---

## Configuration

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `HF_TOKEN` | HuggingFace access token | Gated datasets |
| `NGC_API_KEY` | NVIDIA NGC API key | NIM embeddings |
| `NVIDIA_API_KEY` | NVIDIA Build API key | Nemotron Parse |
| `ADOBE_CLIENT_ID` | Adobe client ID | PDF Services |
| `ADOBE_CLIENT_SECRET` | Adobe client secret | PDF Services |

### Custom Paths

All scripts support custom paths via CLI arguments:

```bash
--datasets-dir /path/to/datasets       # HuggingFace datasets cache
--checkpoints-dir /path/to/checkpoints # Model checkpoints cache
--embeddings-dir /path/to/embeddings   # Embedding output directory
```

---

## Dependencies

### Core
- `torch`, `torchvision` — PyTorch with CUDA support
- `transformers`, `sentence-transformers` — HuggingFace models
- `datasets` — HuggingFace datasets library

### Visualization
- `umap-learn` — Dimensionality reduction
- `matplotlib`, `plotly` — Static and interactive plots
- `pandas`, `seaborn` — Data manipulation

### PDF Processing
- `PyMuPDF (fitz)` — PDF rendering
- `Pillow` — Image manipulation
- `beautifulsoup4` — HTML parsing
- `pdfservices-sdk` — Adobe PDF Services (Python 3.10+)

### Utilities
- `rich` — Beautiful terminal output
- `tqdm` — Progress bars
- `requests` — HTTP requests

See `requirements.txt` for complete list with versions.

---

## Performance Tips

### Multi-GPU Embedding Extraction

```bash
python download_nemotron_datasets.py --all \
  --datasets-dir /raid/datasets \
  --checkpoints-dir /raid/checkpoints 

python extract_embeddings_parallel_shards.py --all \
  --num-gpus 8 \
  --batch-size 100 \
  --max-text-length 8192 \
  --datasets-dir /raid/datasets \
  --checkpoints-dir /raid/checkpoints \
  --embeddings-dir /raid/embeddings
```

### Long-Running Jobs

```bash
# Use screen or tmux for large extractions
screen -S embedding-extraction
python extract_embeddings_parallel_shards.py --all --num-gpus 8

# Detach: Ctrl+A, D
# Reattach: screen -r embedding-extraction
```

---

## Troubleshooting

> 📖 **See also:** [doc/BATCHSIZES.md](doc/BATCHSIZES.md) for detailed memory/batch size guidance, [doc/TROUBLESHOOT.md](doc/TROUBLESHOOT.md) for GPU issues.

### Dataset Download Issues

```bash
# Check HuggingFace authentication
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

### Embedding Extraction Issues

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.device_count())"

# Validate existing embeddings
python validate_embeddings.py --verbose
```

### API Issues

```bash
# Test NVIDIA API
curl -H "Authorization: Bearer $NGC_API_KEY" \
  https://integrate.api.nvidia.com/v1/models

# Test Adobe API credentials
python -c "from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials; print('OK')"
```

---

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **NVIDIA** — Nemotron datasets and NIM embedding models
- **HuggingFace** — Datasets library and model hub
- **Adobe** — PDF Services API
