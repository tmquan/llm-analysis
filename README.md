# Datasets
Nemotron v1:

  ✅ Downloaded successfully
  
    Splits: chat, code, math, stem, tool_calling
  
    Total samples: 25,659,642
  
      - chat: 746,622 samples
    
        - code: 1,896,395 samples
    
        - math: 2,044,407 samples
    
        - stem: 20,662,167 samples
    
        - tool_calling: 310,051 samples
    
    

Nemotron v2:

  ✅ Downloaded successfully
  
    Splits: stem, chat, math, code, multilingual_ja, multilingual_de, multilingual_it, multilingual_es, multilingual_fr
  
    Total samples: 6,341,414
  
      - stem: 355,000 samples
    
        - chat: 627,720 samples
    
        - math: 239,467 samples
    
        - code: 175,000 samples
    
        - multilingual_ja: 975,202 samples
    
        - multilingual_de: 1,015,314 samples
    
        - multilingual_it: 1,016,503 samples
    
        - multilingual_es: 935,704 samples
    
        - multilingual_fr: 1,001,504 samples
    
    

Llama-Nemotron SFT:

  ✅ Downloaded successfully
  
    Splits: code, math, science, chat, safety
  
    Total samples: 32,955,418
  
      - code: 10,108,883 samples
    
        - math: 22,066,397 samples
    
        - science: 708,920 samples
    
        - chat: 39,792 samples
    
        - safety: 31,426 samples
    
    

Llama-Nemotron RL:

  ❌ Download failed or skipped
  
  
# NVIDIA NIM Embedding Extraction Toolkit

Extract high-quality embeddings from NVIDIA's Nemotron datasets using the **Llama-3.2-NemoRetriever-300M-Embed-v2** model (2048-dimensional embeddings), with support for both cloud and local deployment.

---

## 🚀 Quick Start

### 1. Setup Conda Environment

```bash
# Create conda environment with Python 3.12
conda create -n nemotron python=3.12

# Activate your conda environment
conda activate nemotron

# Install PyTorch with CUDA 13.0 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install additional dependencies
pip install bs4 rich huggingface_hub

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Get Your NGC API Key

Get a free API key from NVIDIA: https://org.ngc.nvidia.com/setup/api-key

```bash
# Set API key (add to ~/.bashrc for persistence)
export NGC_API_KEY=nvapi-xxxxx
```

### 2b. Hugging Face Authentication (Optional)

If you need to access gated datasets, login to Hugging Face:

```bash
# Activate nemotron environment first
conda activate nemotron

# Login to Hugging Face (use 'huggingface-cli' not 'hf')
huggingface-cli login

# Or set token directly
export HF_TOKEN=hf_xxxxx
```

### 3. Test the Cloud API

```bash
conda run -n nemotron python extract_emb.py --cloud --test
```

### 4. Extract Your First Embeddings

```bash
conda run -n nemotron python extract_emb.py --cloud \
  --llama-sft-math \
  --num-samples 200 \
  --output math_embeddings.jsonl
```

---

## 📋 Common Use Cases

### 🎯 Extract All Splits to Separate Files (Recommended)

Use the batch extraction script to process all dataset splits efficiently:

```bash
# Extract all Nemotron v1, v2, and Llama-SFT splits
# Each split saved to: Nemotron-Post-Training-Dataset-v{1|2|llama-sft}-{split}.json
bash extract_emb_all_splits.sh --cloud

# Or use local NIM (requires GPU + Docker)
bash extract_emb_all_splits.sh --local
```

**This will create 15 separate JSONL files:**
- `Nemotron-Post-Training-Dataset-v1-{chat,code,math,stem,tool}.jsonl` (5 files)
- `Nemotron-Post-Training-Dataset-v2-{chat,code,math,stem,tool}.jsonl` (5 files)
- `Nemotron-Post-Training-Dataset-llama-sft-{math,code,science,chat,safety}.jsonl` (5 files)

**Configuration in `extract_emb_all_splits.sh`:**
- `NUM_SAMPLES=-1` (extracts ALL samples from each split)
- `BATCH_SIZE=64` (processes 64 samples per API request)
- `INPUT_TYPE=passage` (for indexing/search corpus)
- `MAX_TEXT_LENGTH=8192` (8K character limit per sample)

---

### Extract All Samples from All v1 Splits (Single File)
```bash
python extract_emb.py --cloud \
  --v1-all \
  --num-samples -1 \
  --output Nemotron-Post-Training-Dataset-v1.jsonl
```
This extracts **all samples** from all Nemotron v1 splits (chat, code, math, stem, tool_calling).

### Extract All Samples from All v2 Splits
```bash
python extract_emb.py --cloud \
  --v2-all \
  --num-samples -1 \
  --output Nemotron-Post-Training-Dataset-v2.jsonl
```

### Extract All Llama-Nemotron SFT Data
```bash
python extract_emb.py --cloud \
  --llama-sft-all \
  --num-samples -1 \
  --output Llama-Nemotron-SFT-Complete.jsonl
```

### Compare Math vs Code vs Science (200 samples each)
```bash
python extract_emb.py --cloud \
  --llama-sft-math \
  --llama-sft-code \
  --llama-sft-science \
  --num-samples 200 \
  --output stem_comparison.jsonl
```

### Extract Specific Split with Custom Settings
```bash
python extract_emb.py --cloud \
  --v1-chat \
  --num-samples 500 \
  --batch-size 64 \
  --max-text-length 8192 \
  --output chat_detailed.jsonl
```

### Extract for Quick Exploration (50 samples per split)
```bash
python extract_emb.py --cloud \
  --v1-all \
  --v2-all \
  --num-samples 50 \
  --output exploration.jsonl
```

### Extract Safety Data for Analysis
```bash
python extract_emb.py --cloud \
  --llama-sft-safety \
  --num-samples -1 \
  --output safety_complete.jsonl
```

## 🎨 Visualize with UMAP

After extracting embeddings, create beautiful visualizations:

```bash
# Install visualization dependencies
pip install umap-learn matplotlib plotly

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
| `--batch-size N` | 64 | Batch size for API requests |
| `--max-text-length N` | 8192 | Maximum text length in chars |
| `--input-type TYPE` | passage | 'query' or 'passage' (see below) |
| `--output FILE` | embeddings.jsonl | Output JSONL file path |

### Input Type: Query vs Passage

The `--input-type` parameter optimizes embeddings for different use cases:

**`passage`** (Default - for indexing):
- Use when embedding **documents/content** to be searched
- For building a searchable corpus
- Longer text (articles, dataset samples)
- **Use this for extracting Nemotron datasets**

**`query`** (for searching):
- Use when embedding **search queries/questions**
- For finding relevant documents
- Shorter text ("What is machine learning?")
- Use when searching through pre-built embeddings

**Why it matters:** Modern embedding models use different encoding strategies for queries vs passages to optimize retrieval quality. Always use `passage` when building your index, and `query` when searching it.

### Performance Tuning

**For Speed:**
```bash
--batch-size 64 --max-text-length 1024
```

**For Quality (Recommended):**
```bash
--batch-size 64 --max-text-length 8192 --input-type passage
```

**For Complete Extraction:**
```bash
--num-samples -1
```

---

## 📦 Output Format

The JSONL (JSON Lines) output format is optimized for streaming and memory-efficient processing:

**Format:**
- First line: Metadata object
- Subsequent lines: One embedding object per line

**Example structure:**

```jsonl
{"type":"metadata","extraction_time":"2025-12-04 10:30:00","model":"nvidia/llama-3.2-nemoretriever-300m-embed-v2","embedding_dimension":2048,"total_samples":1500,...}
{"type":"embedding","dataset":"llama-nemotron-sft","split":"math","text":"Sample text...","embedding":[0.123,-0.456,...],"original_prompt":"..."}
{"type":"embedding","dataset":"llama-nemotron-sft","split":"math","text":"Another sample...","embedding":[0.789,-0.012,...],"original_prompt":"..."}
...
```

### Loading JSONL Files

**Python (basic):**
```python
import json

# Load metadata
with open('embeddings.jsonl') as f:
    metadata = json.loads(f.readline())
    embeddings = [json.loads(line) for line in f]

print(f"Total samples: {metadata['total_samples']}")
print(f"Embedding dim: {metadata['embedding_dimension']}")
```

**Python (with pandas):**
```python
import pandas as pd

# Load all at once
df = pd.read_json('embeddings.jsonl', lines=True)

# Separate metadata and embeddings
metadata = df[df['type'] == 'metadata'].iloc[0].to_dict()
embeddings_df = df[df['type'] == 'embedding']

# Access embeddings
import numpy as np
embedding_matrix = np.array(embeddings_df['embedding'].tolist())
labels = embeddings_df['split'].values
texts = embeddings_df['text'].values
```

**Stream processing (memory-efficient):**
```python
import json

def process_embeddings(filepath):
    with open(filepath) as f:
        metadata = json.loads(f.readline())
        
        # Process one embedding at a time
        for line in f:
            emb_data = json.loads(line)
            # Process emb_data without loading all into memory
            yield emb_data

# Usage
for embedding in process_embeddings('large_file.jsonl'):
    # Process each embedding individually
    pass
```

### Why JSONL?

✅ **Memory Efficient**: Stream large files without loading everything  
✅ **Fast Processing**: Read/write line by line  
✅ **Append-Friendly**: Easy to add new embeddings  
✅ **Standard Format**: Compatible with many tools  
✅ **Smaller Files**: No formatting overhead  

## 🛠️ Complete Workflow (End-to-End)

### Step-by-Step Production Workflow

```bash
# STEP 1: Setup Environment
conda activate nemotron
pip install -r requirements.txt

# STEP 2: Download Datasets (One-Time Setup)
conda run -n nemotron python download_nemotron_datasets.py --llama-sft

# STEP 3: Configure NGC API Key
export NGC_API_KEY=nvapi-xxxxx
# Add to ~/.bashrc for persistence:
echo 'export NGC_API_KEY=nvapi-xxxxx' >> ~/.bashrc

# STEP 4: Test Cloud API
conda run -n nemotron python extract_emb.py --cloud --test

# STEP 5: Extract Embeddings (Multi-Domain)
conda run -n nemotron python extract_emb.py --cloud \
  --llama-sft-math \
  --llama-sft-code \
  --llama-sft-science \
  --num-samples 300 \
  --output stem_embeddings.jsonl

# STEP 6: Install Visualization Tools
conda activate nemotron
pip install umap-learn matplotlib plotly pandas

# STEP 7: Create Static Visualization
conda run -n nemotron python visualize_embeddings.py \
  stem_embeddings.json \
  --output stem_umap.png

# STEP 8: Create Interactive HTML
conda run -n nemotron python visualize_embeddings.py \
  stem_embeddings.json \
  --interactive stem_umap.html

# STEP 9: Explore Results
firefox stem_umap.html  # or: open stem_umap.html (macOS)
```

---

## 📚 Files

| File | Description |
|------|-------------|
| `extract_emb.py` | Main extraction script |
| `extract_emb_all_splits.sh` | Batch script to extract all splits |
| `visualize_embeddings.py` | UMAP visualization |
| `docker_nim.py` | Local Docker NIM management |
| `download_nemotron_datasets.py` | Download datasets |
| `debug_nim_api.py` | API debugging utility |
| `requirements.txt` | Python dependencies |

## 🌟 Features

✅ **Dual Mode**: Cloud API or local Docker  
✅ **Batch extraction script**: Process all splits automatically  
✅ **Nemotron-style flags**: Similar to download script  
✅ **Multi-dataset**: Process multiple splits in one command  
✅ **UMAP-ready**: Rich metadata for visualization  
✅ **Batch processing**: Efficient with progress tracking  
✅ **Interactive viz**: Hover to see original text  
✅ **2048-dim embeddings**: High-quality Llama-3.2-NemoRetriever model  
✅ **Optimized for passage indexing**: Perfect for search/retrieval tasks  

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

Use the provided script to extract all splits automatically:

```bash
#!/bin/bash
# Extract all datasets to separate files
bash extract_emb_all_splits.sh --cloud

# Or with local NIM
bash extract_emb_all_splits.sh --local
```

**What it does:**
- Extracts all 15 dataset splits (v1, v2, llama-sft)
- Saves each split to a separate JSON file
- Uses optimal settings: batch_size=64, max_text_length=8192
- Processes ALL samples (`num_samples=-1`)
- Shows progress and summary at the end

**Output files:**
```
Nemotron-Post-Training-Dataset-v1-chat.jsonl
Nemotron-Post-Training-Dataset-v1-code.jsonl
Nemotron-Post-Training-Dataset-v1-math.jsonl
Nemotron-Post-Training-Dataset-v1-stem.jsonl
Nemotron-Post-Training-Dataset-v1-tool.jsonl
Nemotron-Post-Training-Dataset-v2-chat.jsonl
Nemotron-Post-Training-Dataset-v2-code.jsonl
Nemotron-Post-Training-Dataset-v2-math.jsonl
Nemotron-Post-Training-Dataset-v2-stem.jsonl
Nemotron-Post-Training-Dataset-v2-tool.jsonl
Nemotron-Post-Training-Dataset-llama-sft-math.jsonl
Nemotron-Post-Training-Dataset-llama-sft-code.jsonl
Nemotron-Post-Training-Dataset-llama-sft-science.jsonl
Nemotron-Post-Training-Dataset-llama-sft-chat.jsonl
Nemotron-Post-Training-Dataset-llama-sft-safety.jsonl
```

### Manual Batch Processing

Create your own extraction script:
```bash
#!/bin/bash
conda activate nemotron

# Extract v1
python extract_emb.py --cloud --v1-all --num-samples -1 \
  --output v1_complete.jsonl

# Extract v2  
python extract_emb.py --cloud --v2-all --num-samples -1 \
  --output v2_complete.jsonl

# Extract SFT
python extract_emb.py --cloud --llama-sft-all --num-samples -1 \
  --output sft_complete.jsonl

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
  --output complete_nemotron.jsonl

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
# Increase batch size (faster processing)
--batch-size 64

# Reduce text length (faster but less context)
--max-text-length 2048

# Use fewer samples for testing
--num-samples 100
```

#### Large Dataset Extraction
```bash
# Run in screen/tmux for long-running extractions
screen -S embedding-extraction

# Use the batch script for all splits
bash extract_emb_all_splits.sh --cloud

# Detach: Ctrl+A, D
# Reattach: screen -r embedding-extraction
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

- **README.md** (this file) - Complete guide with examples
- **Model**: `nvidia/llama-3.2-nemoretriever-300m-embed-v2` (2048 dimensions)
- **API Documentation**: https://docs.nvidia.com/nim/

## 🚀 Quick Reference

### Start Extraction Immediately
```bash
# Extract all datasets (recommended)
bash extract_emb_all_splits.sh --cloud

# Extract single split for testing
python extract_emb.py --cloud --v1-chat --num-samples 100

# Check API status
python extract_emb.py --test --cloud
```

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
