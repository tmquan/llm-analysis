# NVIDIA NIM Embedding Extraction Toolkit

Extract high-quality embeddings from NVIDIA's Nemotron datasets using the LLama-3.2-NV-EmbedQA-1B-v2 model, with support for both cloud and local deployment.

---

## 🚀 Quick Start

### 1. Setup Conda Environment

```bash
# Activate your conda environment
conda activate nemotron  # or your preferred environment

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Your NGC API Key

Get a free API key from NVIDIA: https://org.ngc.nvidia.com/setup/api-key

```bash
# Set API key (add to ~/.bashrc for persistence)
export NGC_API_KEY=nvapi-xxxxx
```

### 3. Test the Cloud API

```bash
conda run -n nemotron python extract_emb.py --cloud --testm
```

### 4. Extract Your First Embeddings

```bash
conda run -n nemotron python extract_emb.py --cloud \
  --llama-sft-math \
  --num-samples 200 \
  --output math_embeddings.json
```

---

## 📋 Common Use Cases

### Extract All Samples from All v1 Splits (Complete Dataset)
```bash
python extract_emb.py --cloud \
  --v1-all \
  --num-samples -1 \
  --output Nemotron-Post-Training-Dataset-v1.json
```
This extracts **all samples** from all Nemotron v1 splits (chat, code, math, stem, tool_calling).

### Extract All Samples from All v2 Splits
```bash
python extract_emb.py --cloud \
  --v2-all \
  --num-samples -1 \
  --output Nemotron-Post-Training-Dataset-v2.json
```

### Extract All Llama-Nemotron SFT Data
```bash
python extract_emb.py --cloud \
  --llama-sft-all \
  --num-samples -1 \
  --output Llama-Nemotron-SFT-Complete.json
```

### Compare Math vs Code vs Science (200 samples each)
```bash
python extract_emb.py --cloud \
  --llama-sft-math \
  --llama-sft-code \
  --llama-sft-science \
  --num-samples 200 \
  --output stem_comparison.json
```

### Extract Specific Split with Custom Settings
```bash
python extract_emb.py --cloud \
  --v1-chat \
  --num-samples 500 \
  --batch-size 64 \
  --max-text-length 4096 \
  --output chat_detailed.json
```

### Extract for Quick Exploration (50 samples per split)
```bash
python extract_emb.py --cloud \
  --v1-all \
  --v2-all \
  --num-samples 50 \
  --output exploration.json
```

### Extract Safety Data for Analysis
```bash
python extract_emb.py --cloud \
  --llama-sft-safety \
  --num-samples -1 \
  --output safety_complete.json
```

m## 🎨 Visualize with UMAP

After extracting embeddings, create beautiful visualizations:m

```bash
# Install visualization dependencies
pip install umap-learn matplotlib plotlym

# Create static plot
python visualize_embeddings.py embeddings.json --output umap_plot.png

# Create interactive HTML (with hover text!)
python visualize_embeddings.py embeddings.json --interactive umap.html

# Customize UMAP parameters
python visualize_embeddings.py embeddings.json \
  --n-neighbors 30 \
  --min-dist 0.05 \
  --output umap_detailed.png
```

## 📊 Available Dataset Flags

### Nemotron v1 (Original)
```bash
--v1-chat          # Chat conversations
--v1-code          # Code examples
--v1-math          # Math problems
--v1-stem          # STEM topics
--v1-tool          # Tool calling
--v1-all           # All v1 splits
```

### Nemotron v2 (Updated)
```bash
--v2-chat          # Chat conversations (v2)
--v2-code          # Code examples (v2)
--v2-math          # Math problems (v2)
--v2-stem          # STEM topics (v2)
--v2-tool          # Tool calling (v2)
--v2-all           # All v2 splits
```

### Llama-Nemotron SFT
```bash
--llama-sft-math      # Math dataset
--llama-sft-code      # Code dataset
--llama-sft-science   # Science dataset
--llama-sft-chat      # Chat dataset
--llama-sft-safety    # Safety dataset
--llama-sft-all       # All SFT splits
```

### Extract Everything
```bash
--all              # All datasets (⚠️ very large!)
```

## ⚙️ Configuration Options

### API Mode Selection

| Option | Description | Requirements |
|--------|-------------|--------------|
| `--cloud` | NVIDIA Cloud API | NGC API key, internet |
| `--local` | Local Docker container | GPU, Docker, CUDA |
| `--api-url URL` | Custom endpoint | Custom server |
| `--api-key KEY` | API key inline | Alternative to env var |

**Recommendation:** Use `--cloud` for ease of use and reliability.

### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-samples N` | 100 | Samples per split (-1 = all) |
| `--batch-size N` | 32 | Batch size for API requests |
| `--max-text-length N` | 4096 | Maximum text length in chars |
| `--input-type TYPE` | query | 'query' or 'passage' |
| `--output FILE` | embeddings.json | Output JSON file path |

### Performance Tuning

**For Speed:**
```bash
--batch-size 64 --max-text-length 512
```

**For Quality:**
```bash
--batch-size 32 --max-text-length 4096 --input-type passage
```

**For Complete Extraction:**
```bash
--num-samples -1
```

---

## 📦 Output Format

The JSON output includes rich metadata for UMAP visualization:

```json
{
  "metadata": {
    "extraction_time": "2025-12-04 10:30:00",
    "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "embedding_dimension": 1024,
    "total_samples": 1500
  },
  "embeddings": [
    {
      "dataset": "llama-nemotron-sft",
      "split": "math",
      "text": "Sample text...",
      "embedding": [0.123, -0.456, ...],
      "original_prompt": "...",
      "original_response": "..."
    }
  ],
  "umap_ready": true
}
```

## 🛠️ Complete Workflow (End-to-End)

### Step-by-Step Production Workflow

```bash
# ═══════════════════════════════════════════════════════════
# STEP 1: Setup Environment
# ═══════════════════════════════════════════════════════════
conda activate nemotron
pip install -r requirements.txt

# ═══════════════════════════════════════════════════════════
# STEP 2: Download Datasets (One-Time Setup)
# ═══════════════════════════════════════════════════════════
conda run -n nemotron python download_nemotron_datasets.py --llama-sft

# ═══════════════════════════════════════════════════════════
# STEP 3: Configure NGC API Key
# ═══════════════════════════════════════════════════════════
export NGC_API_KEY=nvapi-xxxxx
# Add to ~/.bashrc for persistence:
echo 'export NGC_API_KEY=nvapi-xxxxx' >> ~/.bashrc

# ═══════════════════════════════════════════════════════════
# STEP 4: Test Cloud API
# ═══════════════════════════════════════════════════════════
conda run -n nemotron python extract_emb.py --cloud --test

# ═══════════════════════════════════════════════════════════
# STEP 5: Extract Embeddings (Multi-Domain)
# ═══════════════════════════════════════════════════════════
conda run -n nemotron python extract_emb.py --cloud \
  --llama-sft-math \
  --llama-sft-code \
  --llama-sft-science \
  --num-samples 300 \
  --output stem_embeddings.json

# ═══════════════════════════════════════════════════════════
# STEP 6: Install Visualization Tools
# ═══════════════════════════════════════════════════════════
conda activate nemotron
pip install umap-learn matplotlib plotly pandas

# ═══════════════════════════════════════════════════════════
# STEP 7: Create Static Visualization
# ═══════════════════════════════════════════════════════════
conda run -n nemotron python visualize_embeddings.py \
  stem_embeddings.json \
  --output stem_umap.png

# ═══════════════════════════════════════════════════════════
# STEP 8: Create Interactive HTML
# ═══════════════════════════════════════════════════════════
conda run -n nemotron python visualize_embeddings.py \
  stem_embeddings.json \
  --interactive stem_umap.html

# ═══════════════════════════════════════════════════════════
# STEP 9: Explore Results
# ═══════════════════════════════════════════════════════════
firefox stem_umap.html  # or: open stem_umap.html (macOS)
```

---

## 📚 Files

| File | Description |
|------|-------------|
| `extract_emb.py` | Main extraction script |
| `visualize_embeddings.py` | UMAP visualization |
| `deploy_nim.py` | Local Docker management |
| `download_nemotron_datasets.py` | Download datasets |
| `requirements.txt` | Python dependencies |
| `COMPLETE_GUIDE.md` | Full documentation |

## 🌟 Features

✅ **Dual Mode**: Cloud API or local Docker  
✅ **Nemotron-style flags**: Similar to download script  
✅ **Multi-dataset**: Process multiple splits in one command  
✅ **UMAP-ready**: Rich metadata for visualization  
✅ **Batch processing**: Efficient with progress tracking  
✅ **Interactive viz**: Hover to see original text  

## 💡 Pro Tips

### Persistent API Key Setup

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export NGC_API_KEY=nvapi-xxxxx
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Conda Alias for Quick Access

Add to `~/.bashrc`:
```bash
alias extract-emb='conda run -n nemotron python ~/llm-analysis/extract_emb.py'
alias viz-emb='conda run -n nemotron python ~/llm-analysis/visualize_embeddings.py'
```

Usage:
```bash
extract-emb --cloud --llama-sft-math --num-samples 100
viz-emb embeddings.json --interactive umap.html
```

### Batch Processing Script

Create `extract_all.sh`:
```bash
#!/bin/bash
conda activate nemotron

# Extract v1
python extract_emb.py --cloud --v1-all --num-samples -1 \
  --output v1_complete.json

# Extract v2  
python extract_emb.py --cloud --v2-all --num-samples -1 \
  --output v2_complete.json

# Extract SFT
python extract_emb.py --cloud --llama-sft-all --num-samples -1 \
  --output sft_complete.json

echo "✅ All extractions complete!"
```

### UMAP Parameter Optimization

**For Dense Clusters:**
```bash
--n-neighbors 5 --min-dist 0.01
```

**For Separated Clusters:**
```bash
--n-neighbors 50 --min-dist 0.5
```

**For Balanced View (Default):**
```bash
--n-neighbors 15 --min-dist 0.1
```

### Complete Dataset Extraction

```bash
# Run in screen/tmux for large extractions
screen -S embedding-extraction

conda run -n nemotron python extract_emb.py --cloud \
  --all \
  --num-samples -1 \
  --output complete_nemotron.json

# Detach: Ctrl+A, D
# Reattach: screen -r embedding-extraction
```

---

## 🐛 Troubleshooting

### Cloud API Issues

#### API Key Not Found
```bash
# Check if API key is set
echo $NGC_API_KEY

# If empty, set it
export NGC_API_KEY=nvapi-xxxxx

# Make persistent (add to ~/.bashrc)
echo 'export NGC_API_KEY=nvapi-xxxxx' >> ~/.bashrc
source ~/.bashrc
```

#### Authentication Failed
```bash
# Get a new API key at: https://org.ngc.nvidia.com/setup/api-key
# Test with new key
conda run -n nemotron python extract_emb.py --cloud --test
```

### Dataset Issues

#### Datasets Not Found
```bash
# Download datasets first
conda run -n nemotron python download_nemotron_datasets.py --v1 --v2 --llama-sft

# List available datasets
conda run -n nemotron python extract_emb.py --list
```

#### Import Errors
```bash
# Ensure all dependencies are installed
conda activate nemotron
pip install -r requirements.txt
```

### Conda Environment Issues

#### Wrong Environment
```bash
# Check current environment
conda env list

# Activate correct environment
conda activate nemotron

# Verify Python version
python --version  # Should be 3.10+
```

#### Missing Packages
```bash
# Update conda
conda update conda

# Reinstall requirements
conda activate nemotron
pip install --upgrade -r requirements.txt
```

### Performance Issues

#### Slow Extraction
```bash
# Increase batch size
--batch-size 64

# Reduce text length
--max-text-length 1024

# Use fewer samples for testing
--num-samples 100
```

### Get Help

```bash
# Show all options
conda run -n nemotron python extract_emb.py --help

# Show visualization options
conda run -n nemotron python visualize_embeddings.py --help

# Check version compatibility
conda list | grep -E "datasets|requests|numpy"
```

---

## 📖 Documentation

- **COMPLETE_GUIDE.md** - Full guide with all examples
- **USAGE_GUIDE.md** - Quick reference
- **NIM_TROUBLESHOOTING.md** - Troubleshooting tips

## 🎯 One-Command Quick Start

Get started with embeddings in one command:

```bash
# Setup and run (paste all at once)
conda activate nemotron && \
export NGC_API_KEY=nvapi-xxxxx && \
conda run -n nemotron python extract_emb.py --cloud --llama-sft-math --num-samples 100 && \
conda run -n nemotron python visualize_embeddings.py embeddings.json --interactive umap.html && \
echo "✅ Done! Open umap.html in your browser"
```

---

**Ready to extract embeddings? Happy analyzing! 🚀**
