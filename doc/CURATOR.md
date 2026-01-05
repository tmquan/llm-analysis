# NeMo Curator Dataset Explorer & Processor

This document explains the `process_data_nemo_curator.py` script, which provides dataset discovery, processing, and **embedding extraction** capabilities using NVIDIA's NeMo Curator framework.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Dataset Discovery](#dataset-discovery)
- [NeMo Curator Integration](#nemo-curator-integration)
- [Embedding Extraction](#embedding-extraction)
- [Command Line Interface](#command-line-interface)
- [Discovered Datasets Summary](#discovered-datasets-summary)
- [Code Walkthrough](#code-walkthrough)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is NeMo Curator?

[NeMo Curator](https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html) is NVIDIA's data curation toolkit designed for preparing high-quality datasets for large language model (LLM) training. It provides:

- **Scalable Data Loading**: Read JSONL, Parquet, and other formats efficiently
- **Distributed Processing**: Built on Ray for parallel execution
- **Quality Filtering**: Word count filters, classifiers, deduplication
- **Pipeline Architecture**: Composable stages for flexible workflows

### What Does This Script Do?

The `process_data_nemo_curator.py` script serves three main purposes:

1. **Dataset Exploration** (works without NeMo Curator installed)
   - Discovers all JSONL, Parquet, and Arrow datasets in `/raid/datasets`
   - Extracts metadata: columns, data types, splits, sample counts
   - Identifies columns suitable for embedding extraction
   - Exports information to JSON for programmatic use

2. **Dataset Processing** (requires NeMo Curator)
   - Creates processing pipelines using NeMo Curator readers
   - Applies filters (e.g., word count filtering)
   - Executes pipelines on Ray distributed backend

3. **Embedding Extraction** (requires NeMo Curator)
   - Extracts embeddings using `nvidia/llama-embed-nemotron-8b` or other models
   - Automatically concatenates meaningful text columns
   - Saves embeddings to `/raid/embeddings` in Parquet format
   - Supports batch processing of all discovered datasets

---

## Quick Start

### Explore Datasets (No Dependencies Required)

```bash
# Basic exploration
python process_data_nemo_curator.py --explore

# Verbose mode with nested field details
python process_data_nemo_curator.py --explore --verbose

# Export to JSON file
python process_data_nemo_curator.py --explore --output-json datasets_info.json

# Explore a custom directory
python process_data_nemo_curator.py --explore --datasets-dir /path/to/your/data
```

### Process Datasets (Requires NeMo Curator)

```bash
# Process JSONL files
python process_data_nemo_curator.py --process-jsonl /raid/datasets/jsonl/llama-nemotron

# Process Parquet files
python process_data_nemo_curator.py --process-parquet /raid/datasets/llama-nemotron

# Process with specific fields and word count filter
python process_data_nemo_curator.py \
    --process-jsonl /raid/datasets/jsonl/llama-nemotron \
    --fields input reasoning system_prompt \
    --min-words 50 \
    --max-words 1000
```

### Extract Embeddings (Requires NeMo Curator)

```bash
# Extract embeddings from a single dataset (uses nvidia/llama-embed-nemotron-8b by default)
python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron

# Extract with custom model and batch size
python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron \
    --embedding-model nvidia/llama-embed-nemotron-8b \
    --batch-size 256 \
    --max-seq-length 32768

# Extract embeddings from ALL discovered datasets
python process_data_nemo_curator.py --extract-all-embeddings

# Use HuggingFace token for gated models
python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron \
    --embedding-model nvidia/llama-embed-nemotron-8b \
    --hf-token YOUR_HF_TOKEN

# Specify which text columns to use
python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron \
    --text-columns input reasoning output
```

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      process_data_nemo_curator.py                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATASET DISCOVERY MODULE                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │ discover_jsonl_ │  │ discover_arrow_ │  │ discover_parquet_   │  │   │
│  │  │ datasets()      │  │ datasets()      │  │ datasets()          │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │   │
│  │           │                    │                       │             │   │
│  │           └────────────────────┼───────────────────────┘             │   │
│  │                                ▼                                     │   │
│  │                    ┌───────────────────────┐                         │   │
│  │                    │ discover_all_datasets │                         │   │
│  │                    └───────────────────────┘                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA CLASSES                                    │   │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────────────────────┐   │   │
│  │  │ ColumnInfo │   │ SplitInfo  │   │ DatasetInfo                │   │   │
│  │  │ • name     │   │ • name     │   │ • name, path, format       │   │   │
│  │  │ • dtype    │   │ • num_files│   │ • columns: List[ColumnInfo]│   │   │
│  │  │ • is_nested│   │ • samples  │   │ • splits: Dict[SplitInfo]  │   │   │
│  │  │ • candidate│   │ • paths    │   │ • embedding_columns        │   │   │
│  │  └────────────┘   └────────────┘   └────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NEMO CURATOR INTEGRATION                          │   │
│  │  ┌──────────────────────┐     ┌──────────────────────┐              │   │
│  │  │ create_jsonl_pipeline│     │create_parquet_pipeline│              │   │
│  │  │     JsonlReader      │     │    ParquetReader     │              │   │
│  │  └──────────┬───────────┘     └──────────┬───────────┘              │   │
│  │             │                            │                          │   │
│  │             └────────────┬───────────────┘                          │   │
│  │                          ▼                                          │   │
│  │              ┌─────────────────────────┐                            │   │
│  │              │   run_pipeline_with_ray │                            │   │
│  │              │   • RayClient.start()   │                            │   │
│  │              │   • pipeline.run()      │                            │   │
│  │              │   • RayClient.stop()    │                            │   │
│  │              └─────────────────────────┘                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### File Format Support

| Format | Extension | Discovery Method | Reader Class |
|--------|-----------|------------------|--------------|
| JSONL | `.jsonl` | First-line JSON parsing | `JsonlReader` |
| Parquet | `.parquet` | PyArrow schema inspection | `ParquetReader` |
| Arrow (HF Cache) | `.arrow` + `dataset_info.json` | HuggingFace metadata parsing | `ParquetReader` |

---

## Dataset Discovery

### How Discovery Works

The script searches `/raid/datasets` (or a custom directory) for three types of data:

#### 1. JSONL Datasets

```python
def discover_jsonl_datasets(base_dir: Path) -> List[DatasetInfo]:
    """
    Recursively finds all .jsonl files and extracts schema by reading
    the first line of each file.
    """
```

**Process:**
1. Find all `*.jsonl` files recursively
2. Group files by parent directory
3. Read first line to infer schema (column names and types)
4. Identify embedding candidate columns

#### 2. Arrow/HuggingFace Cached Datasets

```python
def discover_arrow_datasets(base_dir: Path) -> List[DatasetInfo]:
    """
    Finds HuggingFace cached datasets by locating dataset_info.json files
    which contain complete metadata about the dataset.
    """
```

**Process:**
1. Find all `dataset_info.json` files (HuggingFace cache format)
2. Parse features schema from JSON
3. Extract split information (train, test, validation, etc.)
4. Match with known NVIDIA Nemotron dataset configurations

#### 3. Standalone Parquet Datasets

```python
def discover_parquet_datasets(base_dir: Path) -> List[DatasetInfo]:
    """
    Finds Parquet files that are NOT part of HuggingFace cache
    (no accompanying dataset_info.json).
    """
```

**Process:**
1. Find all `*.parquet` files
2. Filter out files with nearby `dataset_info.json`
3. Read schema using PyArrow
4. Identify nested types and embedding candidates

### Embedding Candidate Detection

The script automatically identifies columns suitable for embedding extraction using pattern matching:

```python
EMBEDDING_CANDIDATE_PATTERNS = [
    # Direct text content
    'text', 'content', 'output', 'response', 'answer', 'completion',
    # Input/query columns
    'input', 'query', 'question', 'prompt', 'instruction',
    # Conversation columns
    'messages', 'conversation', 'turns', 'dialogue',
    # Reasoning columns
    'reasoning', 'explanation', 'rationale', 'thinking',
    # Code columns
    'code', 'solution', 'program',
    # System prompts
    'system_prompt', 'system',
]
```

A column is marked as an embedding candidate if its name contains any of these patterns (case-insensitive).

---

## NeMo Curator Integration

### Pipeline Architecture

NeMo Curator uses a pipeline-based architecture following the official documentation:

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter
```

### JSONL Pipeline Example

Based on [NeMo Curator Documentation](https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html):

```python
# Initialize Ray client
ray_client = RayClient()
ray_client.start()

# Create pipeline
pipeline = Pipeline(name="jsonl_data_processing")

# Add JSONL reader stage
reader = JsonlReader(
    file_paths="/path/to/data",
    files_per_partition=4,
    fields=["text", "url"]  # Optional: only read specific columns
)
pipeline.add_stage(reader)

# Add optional filter stage
word_filter = ScoreFilter(
    filter_obj=WordCountFilter(min_words=50, max_words=1000),
    text_field="text"
)
pipeline.add_stage(word_filter)

# Execute pipeline
results = pipeline.run()

# Cleanup
ray_client.stop()
```

### Parquet Pipeline Example

```python
# Initialize Ray client
ray_client = RayClient()
ray_client.start()

# Create pipeline
pipeline = Pipeline(name="parquet_data_processing")

# Add Parquet reader (uses PyArrow engine for better performance)
reader = ParquetReader(
    file_paths="/path/to/data",
    files_per_partition=4,
    fields=["text", "metadata"]
)
pipeline.add_stage(reader)

# Execute
results = pipeline.run()

# Cleanup
ray_client.stop()
```

### Reader Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `file_paths` | `str \| list[str]` | File paths or glob patterns | Required |
| `files_per_partition` | `int \| None` | Files per Dask partition | `None` |
| `blocksize` | `int \| str` | Target partition size (e.g., "128MB") | `None` |
| `fields` | `list[str] \| None` | Columns to read (column selection) | `None` (all) |
| `read_kwargs` | `dict` | Extra arguments for underlying reader | `None` |

---

## Embedding Extraction

### Overview

The script includes comprehensive embedding extraction capabilities using NeMo Curator's `EmbeddingCreatorStage`. It can:

1. Automatically detect meaningful text columns
2. Concatenate multiple columns into a single text field
3. Extract embeddings using state-of-the-art models
4. Save results in efficient Parquet format

### Default Model: nvidia/llama-embed-nemotron-8b

The script defaults to using `nvidia/llama-embed-nemotron-8b`, a powerful embedding model from NVIDIA. Key characteristics:

- **Architecture**: Based on Llama architecture optimized for embeddings
- **Embedding Dimension**: 4096
- **Max Sequence Length**: Up to 32K tokens (32768)
- **Best For**: Semantic search, retrieval, clustering

### EmbeddingCreatorStage API

Based on [NeMo Curator Embedders Documentation](https://docs.nvidia.com/nemo/curator/25.09/apidocs/stages/stages.text.embedders.base.html):

```python
from nemo_curator.stages.text.embedders.base import EmbeddingCreatorStage

embedding_stage = EmbeddingCreatorStage(
    model_identifier="nvidia/llama-embed-nemotron-8b",  # HuggingFace model
    text_field="text",                                   # Input column
    embedding_field="embeddings",                        # Output column
    embedding_pooling="mean_pooling",                    # or 'last_token'
    model_inference_batch_size=512,                      # Batch size
    max_seq_length=32768,                                # Max tokens (32K)
    sort_by_length=True,                                 # Efficiency optimization
    padding_side="right",                                # Padding direction
    autocast=True,                                       # Mixed precision
    hf_token=None                                        # For gated models
)
```

### EmbeddingCreatorStage Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_identifier` | `str` | HuggingFace model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `text_field` | `str` | Column containing text to embed | `text` |
| `embedding_field` | `str` | Output column for embeddings | `embeddings` |
| `embedding_pooling` | `str` | `mean_pooling` or `last_token` | `mean_pooling` |
| `model_inference_batch_size` | `int` | Inference batch size | `1024` |
| `max_seq_length` | `int \| None` | Maximum sequence length | `None` |
| `max_chars` | `int \| None` | Maximum characters per text | `None` |
| `sort_by_length` | `bool` | Sort for efficiency | `True` |
| `padding_side` | `str` | `left` or `right` | `right` |
| `autocast` | `bool` | Use automatic mixed precision | `True` |
| `hf_token` | `str \| None` | HuggingFace token for gated models | `None` |

### Text Column Concatenation

The script automatically concatenates meaningful text columns into a single field for embedding. The detection uses these patterns:

```python
EMBEDDING_CANDIDATE_PATTERNS = [
    'text', 'content', 'output', 'response', 'answer', 'completion',
    'input', 'query', 'question', 'prompt', 'instruction',
    'messages', 'conversation', 'turns', 'dialogue',
    'reasoning', 'explanation', 'rationale', 'thinking',
    'code', 'solution', 'program',
    'system_prompt', 'system',
]
```

Example concatenation output:

```
### INPUT ###
[user]: What is 2+2?

### REASONING ###
This is a simple arithmetic problem...

### OUTPUT ###
4
```

### Complete Embedding Pipeline Example

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.embedders.base import EmbeddingCreatorStage

# Initialize Ray
ray_client = RayClient()
ray_client.start()

# Create pipeline
pipeline = Pipeline(name="embedding_extraction")

# Stage 1: Read data
reader = JsonlReader(
    file_paths="/raid/datasets/jsonl/llama-nemotron/*.jsonl",
    files_per_partition=4
)
pipeline.add_stage(reader)

# Stage 2: Extract embeddings
embedding_stage = EmbeddingCreatorStage(
    model_identifier="nvidia/llama-embed-nemotron-8b",
    text_field="input",  # Or concatenated field
    embedding_field="embeddings",
    embedding_pooling="mean_pooling",
    model_inference_batch_size=512,
    max_seq_length=32768,  # 32K tokens
    hf_token="YOUR_HF_TOKEN"  # Required for gated models
)
pipeline.add_stage(embedding_stage)

# Stage 3: Write results
writer = ParquetWriter(
    output_path="/raid/embeddings/llama-nemotron",
    output_type="parquet"
)
pipeline.add_stage(writer)

# Execute
results = pipeline.run()

# Cleanup
ray_client.stop()
```

### Output Format

Embeddings are saved as Parquet files in `/raid/embeddings/<dataset_name>/`:

```
/raid/embeddings/
├── llama-nemotron/
│   ├── part-00000.parquet
│   ├── part-00001.parquet
│   └── ...
├── v3-agentic/
│   └── ...
└── v3-science/
    └── ...
```

Each Parquet file contains the original columns plus an `embeddings` column with the extracted vectors.

---

## Command Line Interface

### Full Options Reference

```
usage: process_data_nemo_curator.py [-h] [--explore] [--datasets-dir DIR]
                                    [--verbose] [--output-json FILE]
                                    [--process-jsonl PATH] [--process-parquet PATH]
                                    [--fields FIELD [FIELD ...]]
                                    [--min-words N] [--max-words N]
                                    [--extract-embeddings PATH]
                                    [--extract-all-embeddings]
                                    [--embedding-model MODEL]
                                    [--embedding-output-dir DIR]
                                    [--batch-size N] [--max-seq-length N]
                                    [--pooling {mean_pooling,last_token}]
                                    [--hf-token TOKEN]
                                    [--text-columns COL [COL ...]]

General Options:
  --explore             Discover and display information about all datasets
  --datasets-dir DIR    Base directory to search (default: /raid/datasets)
  --verbose, -v         Show detailed column info including nested fields
  --output-json FILE    Export dataset information to JSON file

Processing Options:
  --process-jsonl PATH  Process JSONL files using NeMo Curator
  --process-parquet PATH Process Parquet files using NeMo Curator
  --fields FIELD [...]  Specific columns to read
  --min-words N         Minimum word count for filtering
  --max-words N         Maximum word count for filtering

Embedding Extraction Options:
  --extract-embeddings PATH     Extract embeddings from dataset at PATH
  --extract-all-embeddings      Extract embeddings from ALL discovered datasets
  --embedding-model MODEL       HuggingFace model (default: nvidia/llama-embed-nemotron-8b)
  --embedding-output-dir DIR    Output directory (default: /raid/embeddings)
  --batch-size N                Inference batch size (default: 512)
  --max-seq-length N            Maximum sequence length (default: 32768)
  --pooling {mean_pooling,last_token}  Pooling strategy (default: mean_pooling)
  --hf-token TOKEN              HuggingFace token for gated models
  --text-columns COL [...]      Specific text columns to use (auto-detected if not specified)
```

### Usage Examples

```bash
# Example 1: Basic exploration
python process_data_nemo_curator.py --explore

# Example 2: Verbose exploration with JSON export
python process_data_nemo_curator.py --explore -v --output-json /tmp/datasets.json

# Example 3: Explore custom directory
python process_data_nemo_curator.py --explore --datasets-dir ~/my_datasets

# Example 4: Process JSONL with NeMo Curator
python process_data_nemo_curator.py \
    --process-jsonl /raid/datasets/jsonl/llama-nemotron \
    --fields input reasoning

# Example 5: Process with word count filtering
python process_data_nemo_curator.py \
    --process-parquet /raid/datasets/llama-nemotron \
    --fields text \
    --min-words 100 \
    --max-words 5000

# Example 6: Extract embeddings from a single dataset
python process_data_nemo_curator.py \
    --extract-embeddings /raid/datasets/jsonl/llama-nemotron

# Example 7: Extract embeddings with custom parameters
python process_data_nemo_curator.py \
    --extract-embeddings /raid/datasets/jsonl/llama-nemotron \
    --embedding-model nvidia/llama-embed-nemotron-8b \
    --batch-size 256 \
    --max-seq-length 32768 \
    --hf-token YOUR_HF_TOKEN

# Example 8: Batch extract embeddings from ALL datasets
python process_data_nemo_curator.py --extract-all-embeddings

# Example 9: Extract with specific text columns
python process_data_nemo_curator.py \
    --extract-embeddings /raid/datasets/jsonl/llama-nemotron \
    --text-columns input reasoning output \
    --embedding-output-dir /raid/embeddings/custom
```

---

## Discovered Datasets Summary

Running the script on `/raid/datasets` discovers the following datasets:

### JSONL Datasets (5 found)

| Dataset | Path | Splits | Embedding Columns |
|---------|------|--------|-------------------|
| `jsonl-llama-nemotron` | `/raid/datasets/jsonl/llama-nemotron` | instruction_following | input, reasoning, system_prompt |
| `jsonl-llama-rl` | `/raid/datasets/jsonl/llama-rl` | instruction_following | input, reasoning, system_prompt |
| `jsonl-v3-agentic` | `/raid/datasets/jsonl/v3-agentic` | interactive_agent | messages, reasoning |
| `jsonl-v3-rl-blend` | `/raid/datasets/jsonl/v3-rl-blend` | train | prompt, responses_create_params |
| `nemotron-v3-jsonl-rl-blend` | `/raid/datasets/nemotron-v3/jsonl/rl-blend` | train | prompt, responses_create_params |

### Arrow/HuggingFace Datasets (6 found)

| Dataset | HuggingFace Name | Splits | Files | Samples | Embedding Columns |
|---------|------------------|--------|-------|---------|-------------------|
| `llama-nemotron-post-training-dataset:SFT` | `nvidia/Llama-Nemotron-Post-Training-Dataset` | code, math, science, chat, safety | 246 | 32.9M+ | input, output, reasoning, system_prompt |
| `nemotron-post-training-dataset-v1` | `nvidia/Nemotron-Post-Training-Dataset-v1` | chat, code, math, stem, tools | 997 | 25.6M+ | reasoning, messages |
| `nemotron-post-training-dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v2` | stem, chat, math, code, multiturn_* | 196 | 6.3M+ | reasoning, messages |
| `nemotron-science-v1` | `nvidia/Nemotron-Science-v1` | MCQ, RQA | 6 | 226K+ | messages |
| `nemotron-instruction-following-chat-v1` | `nvidia/Nemotron-Instruction-Following-Chat-v1` | chat_if, structured_outputs | 15 | 430K+ | messages, reasoning |
| `nemotron-math-proofs-v1` | `nvidia/Nemotron-Math-Proofs-v1` | lean | 57 | 1.37M+ | messages |

### Parquet Datasets (1 found)

| Dataset | Path | Files | Embedding Columns |
|---------|------|-------|-------------------|
| `llama-nemotron` | `/raid/datasets/llama-nemotron` | 1 | input, reasoning, system_prompt |

### Known Dataset Configurations

The script recognizes these NVIDIA Nemotron datasets:

```python
DATASET_CONFIGS = {
    'v1': 'nvidia/Nemotron-Post-Training-Dataset-v1',
    'v2': 'nvidia/Nemotron-Post-Training-Dataset-v2',
    'llama-sft': 'nvidia/Llama-Nemotron-Post-Training-Dataset (SFT)',
    'llama-rl': 'nvidia/Llama-Nemotron-Post-Training-Dataset (RL)',
    'v3-science': 'nvidia/Nemotron-Science-v1',
    'v3-instruction-chat': 'nvidia/Nemotron-Instruction-Following-Chat-v1',
    'v3-math-proofs': 'nvidia/Nemotron-Math-Proofs-v1',
    'v3-rl-blend': 'nvidia/Nemotron-3-Nano-RL-Training-Blend',
    'v3-agentic': 'nvidia/Nemotron-Agentic-v1',
    'v3-competitive-programming': 'nvidia/Nemotron-Competitive-Programming-v1',
    'v3-math': 'nvidia/Nemotron-Math-v2',
}
```

---

## Code Walkthrough

### Data Classes

#### `ColumnInfo`
Stores metadata about a single column:
```python
@dataclass
class ColumnInfo:
    name: str                          # Column name
    dtype: str                         # Data type (string, List[Dict], etc.)
    is_nested: bool = False            # Contains nested structures?
    nested_fields: List[str] = []      # Field names if nested
    is_embedding_candidate: bool = False  # Suitable for embeddings?
```

#### `SplitInfo`
Stores metadata about a dataset split:
```python
@dataclass
class SplitInfo:
    name: str                          # Split name (train, test, etc.)
    num_files: int = 0                 # Number of data files
    num_samples: Optional[int] = None  # Sample count (if known)
    size_bytes: Optional[int] = None   # Size in bytes
    file_paths: List[str] = []         # Paths to files
```

#### `DatasetInfo`
Complete dataset metadata:
```python
@dataclass
class DatasetInfo:
    name: str                          # Dataset identifier
    path: str                          # Root path
    format: str                        # 'jsonl', 'parquet', 'arrow'
    hf_name: Optional[str] = None      # HuggingFace dataset name
    config: Optional[str] = None       # HuggingFace config
    columns: List[ColumnInfo] = []     # Column metadata
    splits: Dict[str, SplitInfo] = {}  # Split information
    total_files: int = 0               # Total data files
    total_samples: Optional[int] = None  # Total samples
    
    @property
    def embedding_columns(self) -> List[ColumnInfo]:
        """Returns columns marked as embedding candidates."""
        return [col for col in self.columns if col.is_embedding_candidate]
```

### Key Functions

#### Dataset Discovery

```python
def discover_all_datasets(base_dir: Path) -> List[DatasetInfo]:
    """Main entry point for dataset discovery."""
    all_datasets = []
    all_datasets.extend(discover_jsonl_datasets(base_dir))
    all_datasets.extend(discover_arrow_datasets(base_dir))
    all_datasets.extend(discover_parquet_datasets(base_dir))
    return all_datasets
```

#### NeMo Curator Pipeline Creation

```python
def create_jsonl_pipeline(
    file_paths: Union[str, List[str]],
    pipeline_name: str = "jsonl_processing",
    fields: Optional[List[str]] = None,
    files_per_partition: int = 4
) -> Optional[Pipeline]:
    """Create a NeMo Curator pipeline for JSONL files."""
    
    pipeline = Pipeline(name=pipeline_name)
    
    reader = JsonlReader(
        file_paths=file_paths,
        files_per_partition=files_per_partition,
        fields=fields
    )
    pipeline.add_stage(reader)
    
    return pipeline
```

#### Pipeline Execution

```python
def run_pipeline_with_ray(pipeline: Pipeline) -> Any:
    """Execute pipeline using Ray distributed backend."""
    
    ray_client = RayClient()
    ray_client.start()
    
    try:
        results = pipeline.run()
        return results
    finally:
        ray_client.stop()
```

---

## Best Practices

### For Dataset Exploration

1. **Start with `--explore`**: Always explore datasets first to understand structure
2. **Use `--verbose`**: Enable verbose mode for nested field details
3. **Export to JSON**: Use `--output-json` for programmatic access to metadata
4. **Check embedding candidates**: Review automatically detected embedding columns

### For NeMo Curator Processing

1. **Use `--fields`**: Specify only needed columns for better performance
2. **Set appropriate partition sizes**: Adjust `files_per_partition` based on file sizes
3. **Apply filters judiciously**: Word count filters can significantly reduce data volume
4. **Monitor Ray dashboard**: Track pipeline execution in Ray dashboard

### Performance Tips

From [NeMo Curator Documentation](https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html):

- Use `fields` parameter to read only required columns
- Set `files_per_partition` based on cluster size and memory
- Use `blocksize` for fine-grained partition control
- ParquetReader is optimized for columnar access

---

## Troubleshooting

### Common Issues

#### NeMo Curator Not Installed

```
⚠️  NeMo Curator not installed. Running in exploration-only mode.
   Install with: pip install nemo-curator
```

**Solution**: The exploration mode works without NeMo Curator. Install it only if you need processing:
```bash
pip install nemo-curator
```

#### PyArrow Not Installed

```
⚠️  PyArrow not installed. Parquet reading will be limited.
```

**Solution**: Install PyArrow for full Parquet support:
```bash
pip install pyarrow
```

#### Rich Not Installed

The script will fall back to plain-text output if Rich is not available. For better visualization:
```bash
pip install rich
```

#### Empty Dataset Discovery

If no datasets are found:
1. Verify the `--datasets-dir` path exists
2. Check file permissions
3. Ensure files have correct extensions (`.jsonl`, `.parquet`, `.arrow`)
4. For HuggingFace datasets, verify `dataset_info.json` exists

### Dependencies

| Package | Required For | Installation |
|---------|--------------|--------------|
| `nemo-curator` | Pipeline processing | `pip install nemo-curator` |
| `pyarrow` | Parquet schema reading | `pip install pyarrow` |
| `rich` | Beautiful terminal output | `pip install rich` |

---

## References

- [NeMo Curator Documentation - Read Existing Data](https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html)
- [NeMo Curator GitHub Repository](https://github.com/NVIDIA/NeMo-Curator)
- [NVIDIA Nemotron Datasets on HuggingFace](https://huggingface.co/nvidia)
- [Ray Documentation](https://docs.ray.io/)

---

## See Also

- [DATASETS.md](DATASETS.md) - Dataset structure and organization
- [EMBEDDINGS.md](EMBEDDINGS.md) - Embedding extraction guide
- [TROUBLESHOOT.md](TROUBLESHOOT.md) - General troubleshooting

