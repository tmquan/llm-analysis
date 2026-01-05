#!/usr/bin/env python3
"""
================================================================================
NeMo Curator Dataset Explorer & Processor with Embedding Extraction
================================================================================

This script explores and processes datasets in /raid/datasets using NeMo Curator.
It discovers JSONL and Parquet files, extracts metadata, identifies columns
suitable for embedding extraction, and can extract embeddings using the
nvidia/llama-embed-nemotron-8b model.

Reference:
    NeMo Curator Documentation (v25.09):
    - Read Existing Data: https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html
    - Embedders API: https://docs.nvidia.com/nemo/curator/25.09/apidocs/stages/stages.text.embedders.base.html

Usage Examples:
    # Explore all datasets and print summary
    python process_data_nemo_curator.py --explore
    
    # Explore with verbose column details
    python process_data_nemo_curator.py --explore --verbose
    
    # Process a specific JSONL dataset
    python process_data_nemo_curator.py --process-jsonl /raid/datasets/jsonl/llama-nemotron
    
    # Process a specific Parquet dataset
    python process_data_nemo_curator.py --process-parquet /raid/datasets/llama-nemotron
    
    # Extract embeddings from a dataset (saves to /raid/embeddings)
    python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron
    
    # Extract embeddings with custom model and batch size
    python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron \
        --embedding-model nvidia/llama-embed-nemotron-8b \
        --batch-size 512

Author: Auto-generated
Date: 2026-01-05
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# =============================================================================
# OPTIONAL IMPORTS WITH FALLBACKS
# =============================================================================
# We try to import NeMo Curator components, but also provide standalone 
# functionality for dataset exploration without requiring Ray cluster.

try:
    from nemo_curator.core.client import RayClient
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
    from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
    from nemo_curator.stages.text.modules import ScoreFilter
    from nemo_curator.stages.text.filters import WordCountFilter
    # Import embedding stages from NeMo Curator
    # Reference: https://docs.nvidia.com/nemo/curator/25.09/apidocs/stages/stages.text.embedders.base.html
    from nemo_curator.stages.text.embedders.base import EmbeddingCreatorStage
    from nemo_curator.stages.text.modifiers import DocumentModifier
    from nemo_curator.stages.base import ProcessingStage
    NEMO_CURATOR_AVAILABLE = True
    EMBEDDING_AVAILABLE = True
except ImportError as e:
    NEMO_CURATOR_AVAILABLE = False
    EMBEDDING_AVAILABLE = False
    print("⚠️  NeMo Curator not installed. Running in exploration-only mode.")
    print("   Install with: pip install nemo-curator")
    print(f"   Import error: {e}")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    print("⚠️  PyArrow not installed. Parquet reading will be limited.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================

# Default base directory for datasets
DEFAULT_DATASETS_DIR = Path("/raid/datasets")

# Default output directory for embeddings
DEFAULT_EMBEDDINGS_DIR = Path("/raid/embeddings")

# Default embedding model configuration
# Reference: https://docs.nvidia.com/nemo/curator/25.09/apidocs/stages/stages.text.embedders.base.html
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-embed-nemotron-8b"
DEFAULT_EMBEDDING_BATCH_SIZE = 512
DEFAULT_MAX_SEQ_LENGTH = 32768  # 32K tokens for long context embeddings
DEFAULT_EMBEDDING_POOLING = "mean_pooling"  # Options: 'mean_pooling', 'last_token'

# GPU configuration
DEFAULT_NUM_GPUS = 8  # Default to using all 8 GPUs

# Known dataset configurations (matching extract_embeddings_parallel_shards.py)
# These are NVIDIA Nemotron post-training datasets with their HuggingFace names
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'v1': {
        'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v1', 
        'subdir': 'nemotron-v1', 
        'config': None,
        'description': 'Nemotron Post-Training Dataset Version 1'
    },
    'v2': {
        'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v2', 
        'subdir': 'nemotron-v2', 
        'config': None,
        'description': 'Nemotron Post-Training Dataset Version 2'
    },
    'llama-sft': {
        'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset', 
        'subdir': 'llama-nemotron', 
        'config': 'SFT',
        'description': 'Llama-Nemotron Supervised Fine-Tuning Dataset'
    },
    'llama-rl': {
        'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset', 
        'subdir': 'llama-nemotron', 
        'config': 'RL',
        'description': 'Llama-Nemotron Reinforcement Learning Dataset'
    },
    'v3-science': {
        'hf_name': 'nvidia/Nemotron-Science-v1', 
        'subdir': 'nemotron-v3/science', 
        'config': None,
        'description': 'Nemotron Science Dataset (MCQ & RQA)'
    },
    'v3-instruction-chat': {
        'hf_name': 'nvidia/Nemotron-Instruction-Following-Chat-v1', 
        'subdir': 'nemotron-v3/instruction-chat', 
        'config': None,
        'description': 'Nemotron Instruction Following & Chat Dataset'
    },
    'v3-math-proofs': {
        'hf_name': 'nvidia/Nemotron-Math-Proofs-v1', 
        'subdir': 'nemotron-v3/math-proofs', 
        'config': None,
        'description': 'Nemotron Mathematical Proofs (Lean) Dataset'
    },
    'v3-rl-blend': {
        'hf_name': 'nvidia/Nemotron-3-Nano-RL-Training-Blend', 
        'subdir': 'nemotron-v3/rl-blend', 
        'config': None,
        'description': 'Nemotron RL Training Blend Dataset'
    },
    'v3-agentic': {
        'hf_name': 'nvidia/Nemotron-Agentic-v1', 
        'subdir': 'nemotron-v3/agentic', 
        'config': None,
        'description': 'Nemotron Agentic/Interactive Dataset'
    },
    'v3-competitive-programming': {
        'hf_name': 'nvidia/Nemotron-Competitive-Programming-v1', 
        'subdir': 'nemotron-v3/competitive-programming', 
        'config': None,
        'description': 'Nemotron Competitive Programming Dataset'
    },
    'v3-math': {
        'hf_name': 'nvidia/Nemotron-Math-v2', 
        'subdir': 'nemotron-v3/math-v2', 
        'config': None,
        'description': 'Nemotron Math Dataset Version 2'
    },
}

# Columns commonly used for text embedding extraction
# These are heuristic patterns that identify text-rich columns
EMBEDDING_CANDIDATE_PATTERNS = [
    # Direct text content columns
    'text', 'content', 'output', 'response', 'answer', 'completion',
    # Input/query columns
    'input', 'query', 'question', 'prompt', 'instruction',
    # Message-based columns (often contain conversation turns)
    'messages', 'conversation', 'turns', 'dialogue',
    # Reasoning/explanation columns
    'reasoning', 'explanation', 'rationale', 'thinking',
    # Code-related columns
    'code', 'solution', 'program',
    # System prompts
    'system_prompt', 'system',
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ColumnInfo:
    """
    Represents metadata about a single column in a dataset.
    
    Attributes:
        name: Column name
        dtype: Data type (e.g., 'string', 'int64', 'list')
        is_nested: Whether the column contains nested structures
        nested_fields: If nested, the field names within
        is_embedding_candidate: Whether this column is suitable for embeddings
    """
    name: str
    dtype: str
    is_nested: bool = False
    nested_fields: List[str] = field(default_factory=list)
    is_embedding_candidate: bool = False
    
    def __str__(self) -> str:
        """Human-readable representation of column info."""
        candidate_marker = " ✓ (embedding candidate)" if self.is_embedding_candidate else ""
        if self.is_nested:
            fields_str = ", ".join(self.nested_fields[:5])
            if len(self.nested_fields) > 5:
                fields_str += f", ... (+{len(self.nested_fields) - 5} more)"
            return f"{self.name}: {self.dtype} [{fields_str}]{candidate_marker}"
        return f"{self.name}: {self.dtype}{candidate_marker}"


@dataclass
class SplitInfo:
    """
    Represents metadata about a dataset split (e.g., 'train', 'chat', 'code').
    
    Attributes:
        name: Split name
        num_files: Number of files in this split
        num_samples: Total number of samples (if known)
        size_bytes: Total size in bytes (if known)
        file_paths: List of file paths belonging to this split
    """
    name: str
    num_files: int = 0
    num_samples: Optional[int] = None
    size_bytes: Optional[int] = None
    file_paths: List[str] = field(default_factory=list)
    
    @property
    def size_human(self) -> str:
        """Returns human-readable size string."""
        if self.size_bytes is None:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if self.size_bytes < 1024:
                return f"{self.size_bytes:.2f} {unit}"
            self.size_bytes /= 1024
        return f"{self.size_bytes:.2f} PB"


@dataclass
class DatasetInfo:
    """
    Complete metadata about a discovered dataset.
    
    Attributes:
        name: Dataset identifier/name
        path: Root path to the dataset
        format: File format ('jsonl', 'parquet', 'arrow')
        hf_name: HuggingFace dataset name (if applicable)
        config: HuggingFace config name (if applicable)
        description: Human-readable description
        columns: List of column metadata
        splits: Dictionary of split name -> SplitInfo
        total_files: Total number of data files
        total_samples: Total number of samples across all splits
    """
    name: str
    path: str
    format: str
    hf_name: Optional[str] = None
    config: Optional[str] = None
    description: Optional[str] = None
    columns: List[ColumnInfo] = field(default_factory=list)
    splits: Dict[str, SplitInfo] = field(default_factory=dict)
    total_files: int = 0
    total_samples: Optional[int] = None
    
    @property
    def embedding_columns(self) -> List[ColumnInfo]:
        """Returns columns suitable for embedding extraction."""
        return [col for col in self.columns if col.is_embedding_candidate]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'path': self.path,
            'format': self.format,
            'hf_name': self.hf_name,
            'config': self.config,
            'description': self.description,
            'total_files': self.total_files,
            'total_samples': self.total_samples,
            'columns': [
                {
                    'name': col.name,
                    'dtype': col.dtype,
                    'is_nested': col.is_nested,
                    'nested_fields': col.nested_fields,
                    'is_embedding_candidate': col.is_embedding_candidate
                }
                for col in self.columns
            ],
            'splits': {
                name: {
                    'name': split.name,
                    'num_files': split.num_files,
                    'num_samples': split.num_samples,
                    'size_bytes': split.size_bytes
                }
                for name, split in self.splits.items()
            },
            'embedding_columns': [col.name for col in self.embedding_columns]
        }


# =============================================================================
# DATASET DISCOVERY FUNCTIONS
# =============================================================================

def is_embedding_candidate(column_name: str) -> bool:
    """
    Determines if a column is a candidate for embedding extraction.
    
    The function checks if the column name matches known patterns for
    text-rich content that would be suitable for generating embeddings.
    
    Args:
        column_name: Name of the column to check
        
    Returns:
        True if the column is likely suitable for embeddings
        
    Example:
        >>> is_embedding_candidate('text')
        True
        >>> is_embedding_candidate('messages')
        True
        >>> is_embedding_candidate('id')
        False
    """
    column_lower = column_name.lower()
    
    # Check exact matches and substring matches
    for pattern in EMBEDDING_CANDIDATE_PATTERNS:
        if pattern in column_lower:
            return True
    
    return False


def parse_hf_features(features: Dict[str, Any]) -> List[ColumnInfo]:
    """
    Parse HuggingFace dataset_info.json features into ColumnInfo objects.
    
    The HuggingFace datasets library stores schema information in a specific
    format within dataset_info.json files. This function extracts that
    information into our ColumnInfo dataclass.
    
    Args:
        features: Dictionary of feature definitions from dataset_info.json
        
    Returns:
        List of ColumnInfo objects describing each column
        
    Example features structure:
        {
            "text": {"dtype": "string", "_type": "Value"},
            "messages": {
                "feature": {"role": {...}, "content": {...}},
                "_type": "List"
            }
        }
    """
    columns = []
    
    for col_name, col_def in features.items():
        # Determine the data type
        if isinstance(col_def, dict):
            col_type = col_def.get('_type', 'Unknown')
            
            # Handle nested structures (List type)
            if col_type == 'List':
                nested_feature = col_def.get('feature', {})
                if isinstance(nested_feature, dict):
                    # Check if it's a list of dictionaries
                    if '_type' not in nested_feature:
                        # It's a dictionary with multiple fields
                        nested_fields = list(nested_feature.keys())
                        dtype = f"List[Dict]"
                        is_nested = True
                    else:
                        # It's a list of simple values
                        dtype = f"List[{nested_feature.get('dtype', 'unknown')}]"
                        nested_fields = []
                        is_nested = False
                else:
                    dtype = "List[unknown]"
                    nested_fields = []
                    is_nested = False
            # Handle simple Value type
            elif col_type == 'Value':
                dtype = col_def.get('dtype', 'unknown')
                nested_fields = []
                is_nested = False
            else:
                dtype = col_type
                nested_fields = []
                is_nested = False
        else:
            dtype = str(type(col_def).__name__)
            nested_fields = []
            is_nested = False
        
        # Create ColumnInfo with embedding candidacy check
        col_info = ColumnInfo(
            name=col_name,
            dtype=dtype,
            is_nested=is_nested,
            nested_fields=nested_fields,
            is_embedding_candidate=is_embedding_candidate(col_name)
        )
        columns.append(col_info)
    
    return columns


def find_dataset_info_json(directory: Path) -> Optional[Path]:
    """
    Recursively search for dataset_info.json in a directory.
    
    HuggingFace datasets cache their metadata in dataset_info.json files.
    This function searches through the directory structure to find them.
    
    Args:
        directory: Root directory to search
        
    Returns:
        Path to dataset_info.json if found, None otherwise
    """
    # Search pattern for HuggingFace cached datasets
    # They typically have a hash-named subdirectory containing dataset_info.json
    for json_file in directory.rglob('dataset_info.json'):
        return json_file
    return None


def discover_jsonl_datasets(base_dir: Path) -> List[DatasetInfo]:
    """
    Discover all JSONL datasets in the given directory.
    
    This function recursively searches for .jsonl files and extracts
    metadata by reading the first line of each file to determine
    the schema (column names and types).
    
    Args:
        base_dir: Base directory to search for JSONL files
        
    Returns:
        List of DatasetInfo objects for discovered JSONL datasets
    """
    datasets = []
    
    # Find all JSONL files
    jsonl_files = list(base_dir.rglob('*.jsonl'))
    
    if not jsonl_files:
        return datasets
    
    # Group files by their parent directory (each directory is a "dataset")
    dir_to_files: Dict[Path, List[Path]] = {}
    for jsonl_file in jsonl_files:
        parent = jsonl_file.parent
        if parent not in dir_to_files:
            dir_to_files[parent] = []
        dir_to_files[parent].append(jsonl_file)
    
    # Process each directory as a dataset
    for dir_path, files in dir_to_files.items():
        # Create dataset name from path
        rel_path = dir_path.relative_to(base_dir)
        dataset_name = str(rel_path).replace('/', '-')
        
        # Read schema from first file's first line
        columns = []
        sample_file = files[0]
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    sample_data = json.loads(first_line)
                    for key, value in sample_data.items():
                        dtype = type(value).__name__
                        is_nested = isinstance(value, (list, dict))
                        nested_fields = []
                        
                        if isinstance(value, dict):
                            nested_fields = list(value.keys())
                            dtype = 'Dict'
                        elif isinstance(value, list) and value:
                            if isinstance(value[0], dict):
                                nested_fields = list(value[0].keys())
                                dtype = 'List[Dict]'
                            else:
                                dtype = f'List[{type(value[0]).__name__}]'
                        
                        col_info = ColumnInfo(
                            name=key,
                            dtype=dtype,
                            is_nested=is_nested,
                            nested_fields=nested_fields,
                            is_embedding_candidate=is_embedding_candidate(key)
                        )
                        columns.append(col_info)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not read schema from {sample_file}: {e}")
        
        # Count total samples (approximate from file sizes if needed)
        # For accurate count, we'd need to count lines which can be slow
        total_samples = None
        
        # Create splits based on file names
        splits: Dict[str, SplitInfo] = {}
        for file_path in files:
            split_name = file_path.stem  # Filename without extension
            splits[split_name] = SplitInfo(
                name=split_name,
                num_files=1,
                file_paths=[str(file_path)],
                size_bytes=file_path.stat().st_size if file_path.exists() else None
            )
        
        dataset_info = DatasetInfo(
            name=dataset_name,
            path=str(dir_path),
            format='jsonl',
            columns=columns,
            splits=splits,
            total_files=len(files)
        )
        datasets.append(dataset_info)
    
    return datasets


def discover_arrow_datasets(base_dir: Path) -> List[DatasetInfo]:
    """
    Discover Arrow/Parquet datasets (HuggingFace cached format).
    
    HuggingFace datasets are cached as Arrow files with metadata in
    dataset_info.json. This function finds and processes these datasets.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of DatasetInfo objects for discovered Arrow datasets
    """
    datasets = []
    
    # Find all dataset_info.json files (HuggingFace cache format)
    info_files = list(base_dir.rglob('dataset_info.json'))
    
    for info_file in info_files:
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
            
            # Extract dataset metadata
            dataset_name = info_data.get('dataset_name', info_file.parent.name)
            config_name = info_data.get('config_name', 'default')
            
            # Parse features into column info
            features = info_data.get('features', {})
            columns = parse_hf_features(features)
            
            # Extract split information
            splits_data = info_data.get('splits', {})
            splits: Dict[str, SplitInfo] = {}
            total_samples = 0
            
            for split_name, split_info in splits_data.items():
                num_examples = split_info.get('num_examples', 0)
                num_bytes = split_info.get('num_bytes', 0)
                shard_lengths = split_info.get('shard_lengths', [])
                
                # Find Arrow files for this split
                arrow_dir = info_file.parent
                split_pattern = f"*-{split_name}-*.arrow"
                split_files = list(arrow_dir.glob(split_pattern))
                
                # Also try without split name prefix
                if not split_files:
                    split_files = list(arrow_dir.glob(f"*{split_name}*.arrow"))
                
                splits[split_name] = SplitInfo(
                    name=split_name,
                    num_files=len(split_files) if split_files else len(shard_lengths),
                    num_samples=num_examples,
                    size_bytes=num_bytes,
                    file_paths=[str(f) for f in split_files]
                )
                total_samples += num_examples
            
            # Match with known dataset configs
            hf_name = None
            description = None
            for config_key, config_info in DATASET_CONFIGS.items():
                if config_info['hf_name'].split('/')[-1].lower() in dataset_name.lower():
                    hf_name = config_info['hf_name']
                    description = config_info.get('description')
                    break
            
            dataset_info = DatasetInfo(
                name=f"{dataset_name}:{config_name}" if config_name != 'default' else dataset_name,
                path=str(info_file.parent),
                format='arrow',
                hf_name=hf_name,
                config=config_name,
                description=description,
                columns=columns,
                splits=splits,
                total_files=sum(s.num_files for s in splits.values()),
                total_samples=total_samples if total_samples > 0 else None
            )
            datasets.append(dataset_info)
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not parse {info_file}: {e}")
    
    return datasets


def discover_parquet_datasets(base_dir: Path) -> List[DatasetInfo]:
    """
    Discover standalone Parquet files (not in HuggingFace cache format).
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of DatasetInfo objects for discovered Parquet datasets
    """
    datasets = []
    
    # Find all .parquet files that are NOT in HuggingFace cache directories
    parquet_files = list(base_dir.rglob('*.parquet'))
    
    # Filter out files that have dataset_info.json nearby (those are HF cached)
    standalone_files = []
    for pq_file in parquet_files:
        if not (pq_file.parent / 'dataset_info.json').exists():
            standalone_files.append(pq_file)
    
    if not standalone_files:
        return datasets
    
    # Group by directory
    dir_to_files: Dict[Path, List[Path]] = {}
    for pq_file in standalone_files:
        parent = pq_file.parent
        if parent not in dir_to_files:
            dir_to_files[parent] = []
        dir_to_files[parent].append(pq_file)
    
    for dir_path, files in dir_to_files.items():
        rel_path = dir_path.relative_to(base_dir)
        dataset_name = str(rel_path).replace('/', '-')
        
        # Read schema from first file using PyArrow
        columns = []
        if PYARROW_AVAILABLE:
            try:
                parquet_meta = pq.read_metadata(files[0])
                schema = pq.read_schema(files[0])
                
                for field_idx in range(len(schema)):
                    field = schema.field(field_idx)
                    dtype_str = str(field.type)
                    
                    # Check for nested types
                    is_nested = pa.types.is_list(field.type) or pa.types.is_struct(field.type)
                    nested_fields = []
                    
                    if pa.types.is_struct(field.type):
                        nested_fields = [f.name for f in field.type]
                    elif pa.types.is_list(field.type) and pa.types.is_struct(field.type.value_type):
                        nested_fields = [f.name for f in field.type.value_type]
                    
                    col_info = ColumnInfo(
                        name=field.name,
                        dtype=dtype_str,
                        is_nested=is_nested,
                        nested_fields=nested_fields,
                        is_embedding_candidate=is_embedding_candidate(field.name)
                    )
                    columns.append(col_info)
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not read Parquet schema from {files[0]}: {e}")
        
        # Create dataset info
        splits = {
            'default': SplitInfo(
                name='default',
                num_files=len(files),
                file_paths=[str(f) for f in files]
            )
        }
        
        dataset_info = DatasetInfo(
            name=dataset_name,
            path=str(dir_path),
            format='parquet',
            columns=columns,
            splits=splits,
            total_files=len(files)
        )
        datasets.append(dataset_info)
    
    return datasets


def discover_all_datasets(base_dir: Path) -> List[DatasetInfo]:
    """
    Discover all datasets in the base directory.
    
    This is the main entry point for dataset discovery. It combines results
    from JSONL, Parquet, and Arrow dataset discovery functions.
    
    Args:
        base_dir: Base directory to search for datasets
        
    Returns:
        List of all discovered DatasetInfo objects
    """
    print(f"\n🔍 Discovering datasets in: {base_dir}\n")
    
    all_datasets = []
    
    # Discover JSONL datasets
    print("  📄 Searching for JSONL files...")
    jsonl_datasets = discover_jsonl_datasets(base_dir)
    print(f"     Found {len(jsonl_datasets)} JSONL dataset(s)")
    all_datasets.extend(jsonl_datasets)
    
    # Discover Arrow/HuggingFace cached datasets
    print("  📊 Searching for Arrow/HuggingFace cached datasets...")
    arrow_datasets = discover_arrow_datasets(base_dir)
    print(f"     Found {len(arrow_datasets)} Arrow dataset(s)")
    all_datasets.extend(arrow_datasets)
    
    # Discover standalone Parquet datasets
    print("  📁 Searching for standalone Parquet files...")
    parquet_datasets = discover_parquet_datasets(base_dir)
    print(f"     Found {len(parquet_datasets)} Parquet dataset(s)")
    all_datasets.extend(parquet_datasets)
    
    print(f"\n✅ Total datasets discovered: {len(all_datasets)}\n")
    
    return all_datasets


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_dataset_summary_rich(datasets: List[DatasetInfo], verbose: bool = False) -> None:
    """
    Print a beautiful summary of discovered datasets using Rich library.
    
    Args:
        datasets: List of DatasetInfo objects to display
        verbose: If True, show detailed column information
    """
    console = Console()
    
    # Create main summary table
    summary_table = Table(
        title="📚 Dataset Discovery Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    summary_table.add_column("Dataset Name", style="green", no_wrap=True)
    summary_table.add_column("Format", style="yellow")
    summary_table.add_column("Splits", style="blue")
    summary_table.add_column("Files", style="magenta", justify="right")
    summary_table.add_column("Samples", style="cyan", justify="right")
    summary_table.add_column("Embedding Columns", style="bright_green")
    
    for ds in datasets:
        splits_str = ", ".join(ds.splits.keys())
        samples_str = f"{ds.total_samples:,}" if ds.total_samples else "N/A"
        embed_cols = ", ".join([c.name for c in ds.embedding_columns]) or "None detected"
        
        summary_table.add_row(
            ds.name,
            ds.format.upper(),
            splits_str,
            str(ds.total_files),
            samples_str,
            embed_cols
        )
    
    console.print(summary_table)
    console.print()
    
    # Detailed view for each dataset
    for ds in datasets:
        # Create a panel for each dataset
        tree = Tree(f"[bold green]{ds.name}[/bold green]")
        
        # Add basic info
        info_branch = tree.add("📋 [bold]Basic Info[/bold]")
        info_branch.add(f"Path: {ds.path}")
        info_branch.add(f"Format: {ds.format.upper()}")
        if ds.hf_name:
            info_branch.add(f"HuggingFace: {ds.hf_name}")
        if ds.description:
            info_branch.add(f"Description: {ds.description}")
        
        # Add splits info
        splits_branch = tree.add("📊 [bold]Splits[/bold]")
        for split_name, split_info in ds.splits.items():
            split_str = f"[cyan]{split_name}[/cyan]: {split_info.num_files} file(s)"
            if split_info.num_samples:
                split_str += f", {split_info.num_samples:,} samples"
            splits_branch.add(split_str)
        
        # Add columns info
        cols_branch = tree.add("📝 [bold]Columns[/bold]")
        for col in ds.columns:
            col_style = "bright_green" if col.is_embedding_candidate else "white"
            col_str = f"[{col_style}]{col.name}[/{col_style}]: {col.dtype}"
            if col.is_embedding_candidate:
                col_str += " ✓ [dim](embedding candidate)[/dim]"
            if verbose and col.is_nested and col.nested_fields:
                col_str += f"\n    Fields: {', '.join(col.nested_fields[:5])}"
                if len(col.nested_fields) > 5:
                    col_str += f" ... (+{len(col.nested_fields) - 5} more)"
            cols_branch.add(col_str)
        
        # Add embedding candidates summary
        if ds.embedding_columns:
            embed_branch = tree.add("🎯 [bold]Recommended for Embedding Extraction[/bold]")
            for col in ds.embedding_columns:
                embed_branch.add(f"[bright_green]{col.name}[/bright_green] ({col.dtype})")
        
        console.print(Panel(tree, border_style="blue"))
        console.print()


def print_dataset_summary_plain(datasets: List[DatasetInfo], verbose: bool = False) -> None:
    """
    Print a plain-text summary of discovered datasets (fallback when Rich unavailable).
    
    Args:
        datasets: List of DatasetInfo objects to display
        verbose: If True, show detailed column information
    """
    print("=" * 80)
    print("📚 DATASET DISCOVERY SUMMARY")
    print("=" * 80)
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{'─' * 80}")
        print(f"[{i}] {ds.name}")
        print(f"{'─' * 80}")
        print(f"  📁 Path:   {ds.path}")
        print(f"  📄 Format: {ds.format.upper()}")
        
        if ds.hf_name:
            print(f"  🤗 HuggingFace: {ds.hf_name}")
        if ds.description:
            print(f"  📝 Description: {ds.description}")
        
        print(f"  📊 Total Files: {ds.total_files}")
        if ds.total_samples:
            print(f"  📈 Total Samples: {ds.total_samples:,}")
        
        # Splits
        print(f"\n  📂 Splits:")
        for split_name, split_info in ds.splits.items():
            split_line = f"     • {split_name}: {split_info.num_files} file(s)"
            if split_info.num_samples:
                split_line += f", {split_info.num_samples:,} samples"
            print(split_line)
        
        # Columns
        print(f"\n  📝 Columns:")
        for col in ds.columns:
            marker = " ✓ (embedding candidate)" if col.is_embedding_candidate else ""
            print(f"     • {col.name}: {col.dtype}{marker}")
            if verbose and col.is_nested and col.nested_fields:
                fields_preview = ", ".join(col.nested_fields[:5])
                if len(col.nested_fields) > 5:
                    fields_preview += f" ... (+{len(col.nested_fields) - 5} more)"
                print(f"       └─ Fields: {fields_preview}")
        
        # Embedding candidates
        if ds.embedding_columns:
            print(f"\n  🎯 Recommended for Embedding Extraction:")
            for col in ds.embedding_columns:
                print(f"     ★ {col.name} ({col.dtype})")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Total datasets discovered: {len(datasets)}")
    print("=" * 80)


def print_dataset_summary(datasets: List[DatasetInfo], verbose: bool = False) -> None:
    """
    Print dataset summary using Rich if available, otherwise plain text.
    
    Args:
        datasets: List of DatasetInfo objects to display
        verbose: If True, show detailed column information
    """
    if RICH_AVAILABLE:
        print_dataset_summary_rich(datasets, verbose)
    else:
        print_dataset_summary_plain(datasets, verbose)


# =============================================================================
# NEMO CURATOR PROCESSING FUNCTIONS
# =============================================================================

def create_jsonl_pipeline(
    file_paths: Union[str, List[str]],
    pipeline_name: str = "jsonl_processing",
    fields: Optional[List[str]] = None,
    files_per_partition: int = 4
) -> Optional['Pipeline']:
    """
    Create a NeMo Curator pipeline for processing JSONL files.
    
    This function demonstrates how to use NeMo Curator's JsonlReader
    to load and process JSONL datasets. Following the official documentation:
    https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html
    
    Args:
        file_paths: Path(s) to JSONL files or directories containing them
        pipeline_name: Name for the pipeline (for logging/tracking)
        fields: Specific columns to read (None = all columns)
        files_per_partition: Number of files per Dask partition
        
    Returns:
        Configured Pipeline object, or None if NeMo Curator unavailable
        
    Example:
        >>> pipeline = create_jsonl_pipeline(
        ...     file_paths="/raid/datasets/jsonl/llama-nemotron",
        ...     fields=["input", "output", "reasoning"]
        ... )
    """
    if not NEMO_CURATOR_AVAILABLE:
        print("❌ NeMo Curator is not installed. Cannot create pipeline.")
        return None
    
    # Create the pipeline
    pipeline = Pipeline(name=pipeline_name)
    
    # Configure the JSONL reader
    # According to NeMo Curator docs, JsonlReader supports:
    #   - file_paths: str | list[str] - File paths or glob patterns
    #   - files_per_partition: int - Number of files per partition
    #   - fields: list[str] | None - Column selection (None = all)
    reader = JsonlReader(
        file_paths=file_paths,
        files_per_partition=files_per_partition,
        fields=fields  # Only read specific columns for better performance
    )
    pipeline.add_stage(reader)
    
    print(f"✅ Created JSONL pipeline: {pipeline_name}")
    print(f"   Source: {file_paths}")
    if fields:
        print(f"   Fields: {', '.join(fields)}")
    
    return pipeline


def create_parquet_pipeline(
    file_paths: Union[str, List[str]],
    pipeline_name: str = "parquet_processing",
    fields: Optional[List[str]] = None,
    files_per_partition: int = 4
) -> Optional['Pipeline']:
    """
    Create a NeMo Curator pipeline for processing Parquet files.
    
    This function demonstrates how to use NeMo Curator's ParquetReader
    to load and process Parquet datasets. Following the official documentation:
    https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html
    
    ParquetReader provides these optimizations:
        - PyArrow Engine: Uses pyarrow for better performance
        - Storage Options: Supports cloud storage via read_kwargs
        - Schema Handling: Automatic schema inference and validation
        - Columnar Efficiency: Optimized for reading specific columns
    
    Args:
        file_paths: Path(s) to Parquet files or directories
        pipeline_name: Name for the pipeline
        fields: Specific columns to read (None = all columns)
        files_per_partition: Number of files per partition
        
    Returns:
        Configured Pipeline object, or None if NeMo Curator unavailable
        
    Example:
        >>> pipeline = create_parquet_pipeline(
        ...     file_paths="/raid/datasets/llama-nemotron/*.parquet",
        ...     fields=["text", "metadata"]
        ... )
    """
    if not NEMO_CURATOR_AVAILABLE:
        print("❌ NeMo Curator is not installed. Cannot create pipeline.")
        return None
    
    # Create the pipeline
    pipeline = Pipeline(name=pipeline_name)
    
    # Configure the Parquet reader
    # ParquetReader uses PyArrow engine by default for better performance
    reader = ParquetReader(
        file_paths=file_paths,
        files_per_partition=files_per_partition,
        fields=fields  # Column selection for efficiency
    )
    pipeline.add_stage(reader)
    
    print(f"✅ Created Parquet pipeline: {pipeline_name}")
    print(f"   Source: {file_paths}")
    if fields:
        print(f"   Fields: {', '.join(fields)}")
    
    return pipeline


def run_pipeline_with_ray(
    pipeline: 'Pipeline',
    num_gpus: Optional[int] = None
) -> Any:
    """
    Execute a NeMo Curator pipeline using Ray backend with multi-GPU support.
    
    This function demonstrates the standard pattern for running NeMo Curator
    pipelines with Ray distributed computing. It handles:
        1. Initializing the Ray client with GPU resources
        2. Running the pipeline across available GPUs
        3. Cleaning up resources
    
    Args:
        pipeline: Configured NeMo Curator Pipeline object
        num_gpus: Number of GPUs to use (None = auto-detect all available)
        
    Returns:
        Pipeline execution results (typically a Dask DataFrame)
        
    Example:
        >>> pipeline = create_jsonl_pipeline("/data/*.jsonl")
        >>> results = run_pipeline_with_ray(pipeline, num_gpus=8)
        >>> print(results.compute())  # Materialize results
    """
    if not NEMO_CURATOR_AVAILABLE:
        print("❌ NeMo Curator is not installed. Cannot run pipeline.")
        return None
    
    print(f"\n🚀 Starting pipeline: {pipeline.name}")
    print("=" * 50)
    
    # Detect available GPUs if not specified
    try:
        import torch
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            num_gpus = available_gpus
        else:
            num_gpus = min(num_gpus, available_gpus)
        print(f"🖥️  Available GPUs: {available_gpus}, Using: {num_gpus}")
    except ImportError:
        print("⚠️  PyTorch not available for GPU detection")
        num_gpus = num_gpus or DEFAULT_NUM_GPUS
    
    # Initialize Ray client with GPU configuration
    # RayClient handles connection to Ray cluster (local or distributed)
    # The cluster will distribute work across available GPU workers
    ray_client = RayClient()
    ray_client.start()
    print(f"✅ Ray client started (targeting {num_gpus} GPU(s))")
    
    try:
        # Execute the pipeline
        # This returns lazy Dask objects - call .compute() to materialize
        # Ray will automatically distribute the work across GPU workers
        print("⏳ Executing pipeline...")
        results = pipeline.run()
        print("✅ Pipeline execution complete")
        
        return results
        
    finally:
        # Always stop the Ray client to clean up resources
        ray_client.stop()
        print("✅ Ray client stopped")


def process_dataset_with_nemo_curator(
    dataset_info: DatasetInfo,
    output_dir: Optional[str] = None,
    filter_words: Optional[Tuple[int, int]] = None
) -> None:
    """
    Process a discovered dataset using NeMo Curator pipeline.
    
    This function creates and runs an appropriate pipeline based on the
    dataset format (JSONL or Parquet/Arrow), optionally applying filters.
    
    Args:
        dataset_info: DatasetInfo object describing the dataset
        output_dir: Directory to save processed results (optional)
        filter_words: Tuple of (min_words, max_words) for word count filter
        
    Example:
        >>> ds_info = discover_all_datasets(Path("/raid/datasets"))[0]
        >>> process_dataset_with_nemo_curator(
        ...     ds_info,
        ...     filter_words=(50, 1000)
        ... )
    """
    if not NEMO_CURATOR_AVAILABLE:
        print("❌ NeMo Curator is not installed. Cannot process dataset.")
        return
    
    print(f"\n🔄 Processing dataset: {dataset_info.name}")
    print(f"   Format: {dataset_info.format}")
    print(f"   Path: {dataset_info.path}")
    
    # Determine which columns to use for embedding
    embedding_cols = [c.name for c in dataset_info.embedding_columns]
    print(f"   Embedding candidate columns: {embedding_cols}")
    
    # Collect all file paths
    all_files = []
    for split_info in dataset_info.splits.values():
        all_files.extend(split_info.file_paths)
    
    if not all_files:
        # Construct file pattern
        if dataset_info.format == 'jsonl':
            file_pattern = f"{dataset_info.path}/*.jsonl"
        elif dataset_info.format == 'parquet':
            file_pattern = f"{dataset_info.path}/*.parquet"
        else:  # arrow
            file_pattern = f"{dataset_info.path}/*.arrow"
        all_files = file_pattern
    
    # Create appropriate pipeline
    if dataset_info.format == 'jsonl':
        pipeline = create_jsonl_pipeline(
            file_paths=all_files,
            pipeline_name=f"process_{dataset_info.name}",
            fields=embedding_cols if embedding_cols else None
        )
    else:
        pipeline = create_parquet_pipeline(
            file_paths=all_files,
            pipeline_name=f"process_{dataset_info.name}",
            fields=embedding_cols if embedding_cols else None
        )
    
    if pipeline is None:
        return
    
    # Add optional word count filter
    if filter_words and embedding_cols:
        min_words, max_words = filter_words
        text_field = embedding_cols[0]  # Use first embedding column
        
        word_filter = ScoreFilter(
            filter_obj=WordCountFilter(min_words=min_words, max_words=max_words),
            text_field=text_field
        )
        pipeline.add_stage(word_filter)
        print(f"   Added word count filter: {min_words}-{max_words} words on '{text_field}'")
    
    # Run the pipeline
    results = run_pipeline_with_ray(pipeline)
    
    if results is not None:
        print(f"\n✅ Processing complete for {dataset_info.name}")
        # Note: results is a lazy Dask DataFrame
        # Call results.compute() to materialize, or add writer stages


# =============================================================================
# EMBEDDING EXTRACTION FUNCTIONS
# =============================================================================

def concatenate_text_columns(
    columns: List[str],
    separator: str = "\n\n---\n\n"
) -> str:
    """
    Create a text concatenation expression for combining multiple columns.
    
    This is used to combine all meaningful text columns into a single text
    field suitable for embedding extraction.
    
    Args:
        columns: List of column names to concatenate
        separator: String to use between columns
        
    Returns:
        A format string for concatenation
        
    Example:
        >>> cols = ['input', 'reasoning', 'output']
        >>> concatenate_text_columns(cols)
        '{input}\n\n---\n\n{reasoning}\n\n---\n\n{output}'
    """
    return separator.join([f"{{{col}}}" for col in columns])


def format_nested_content(value: Any, depth: int = 0) -> str:
    """
    Recursively format nested content (lists, dicts) into readable text.
    
    This handles the common case in LLM datasets where content is stored
    as lists of message dictionaries with 'role' and 'content' fields.
    
    Args:
        value: The value to format (can be str, list, dict, or nested)
        depth: Current recursion depth (for indentation)
        
    Returns:
        Formatted string representation
        
    Example:
        >>> messages = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi!'}]
        >>> print(format_nested_content(messages))
        [user]: Hello
        [assistant]: Hi!
    """
    if value is None:
        return ""
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, (int, float, bool)):
        return str(value)
    
    if isinstance(value, dict):
        # Special handling for message-style dicts with 'role' and 'content'
        if 'role' in value and 'content' in value:
            role = value.get('role', 'unknown')
            content = value.get('content', '')
            reasoning = value.get('reasoning_content', '') or value.get('reasoning', '')
            result = f"[{role}]: {content}"
            if reasoning:
                result += f"\n[reasoning]: {reasoning}"
            return result
        
        # General dict formatting
        parts = []
        for k, v in value.items():
            formatted_v = format_nested_content(v, depth + 1)
            if formatted_v:
                parts.append(f"{k}: {formatted_v}")
        return "\n".join(parts)
    
    if isinstance(value, list):
        # Format list items
        formatted_items = []
        for item in value:
            formatted_item = format_nested_content(item, depth + 1)
            if formatted_item:
                formatted_items.append(formatted_item)
        return "\n".join(formatted_items)
    
    return str(value)


def create_combined_text_field(row: Dict[str, Any], columns: List[str]) -> str:
    """
    Combine multiple columns from a data row into a single text string.
    
    This function handles various data types including nested structures
    commonly found in LLM training datasets (messages, conversations, etc.).
    
    Args:
        row: Dictionary containing the data row
        columns: List of column names to combine
        
    Returns:
        Combined text string suitable for embedding
        
    Example:
        >>> row = {
        ...     'input': [{'role': 'user', 'content': 'What is 2+2?'}],
        ...     'reasoning': 'Simple arithmetic',
        ...     'output': '4'
        ... }
        >>> text = create_combined_text_field(row, ['input', 'reasoning', 'output'])
    """
    parts = []
    
    for col in columns:
        if col not in row:
            continue
            
        value = row[col]
        if value is None:
            continue
        
        # Format the value
        formatted = format_nested_content(value)
        
        if formatted and formatted.strip():
            # Add column header for clarity
            parts.append(f"### {col.upper()} ###\n{formatted}")
    
    return "\n\n".join(parts)


def create_embedding_pipeline(
    file_paths: Union[str, List[str]],
    file_format: str = "jsonl",
    output_dir: str = str(DEFAULT_EMBEDDINGS_DIR),
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    text_field: str = "combined_text",
    embedding_field: str = "embeddings",
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    pooling: str = DEFAULT_EMBEDDING_POOLING,
    hf_token: Optional[str] = None,
    pipeline_name: str = "embedding_extraction"
) -> Optional['Pipeline']:
    """
    Create a NeMo Curator pipeline for extracting embeddings.
    
    This function creates a complete pipeline that:
    1. Reads data from JSONL or Parquet files
    2. Extracts embeddings using the specified model
    3. Writes results to the output directory
    
    Based on NeMo Curator EmbeddingCreatorStage:
    https://docs.nvidia.com/nemo/curator/25.09/apidocs/stages/stages.text.embedders.base.html
    
    EmbeddingCreatorStage Parameters:
        - model_identifier: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)
        - text_field: Column containing text to embed (default: 'text')
        - embedding_field: Output column for embeddings (default: 'embeddings')
        - embedding_pooling: 'mean_pooling' or 'last_token' (default: 'mean_pooling')
        - model_inference_batch_size: Batch size for inference (default: 1024)
        - max_seq_length: Maximum sequence length (default: None)
        - max_chars: Maximum characters per text (default: None)
        - sort_by_length: Sort by length for efficiency (default: True)
        - padding_side: 'left' or 'right' (default: 'right')
        - autocast: Use automatic mixed precision (default: True)
        - hf_token: HuggingFace token for gated models
    
    Args:
        file_paths: Path(s) to input files or glob patterns
        file_format: 'jsonl' or 'parquet'
        output_dir: Directory to save embeddings
        embedding_model: HuggingFace model identifier
        text_field: Column name containing text to embed
        embedding_field: Column name for output embeddings
        batch_size: Inference batch size
        max_seq_length: Maximum sequence length for tokenization
        pooling: Pooling strategy ('mean_pooling' or 'last_token')
        hf_token: HuggingFace token for accessing gated models
        pipeline_name: Name for the pipeline
        
    Returns:
        Configured Pipeline object, or None if NeMo Curator unavailable
        
    Example:
        >>> pipeline = create_embedding_pipeline(
        ...     file_paths="/raid/datasets/jsonl/llama-nemotron/*.jsonl",
        ...     embedding_model="nvidia/llama-embed-nemotron-8b",
        ...     output_dir="/raid/embeddings/llama-nemotron"
        ... )
        >>> results = run_pipeline_with_ray(pipeline)
    """
    if not NEMO_CURATOR_AVAILABLE:
        print("❌ NeMo Curator is not installed. Cannot create embedding pipeline.")
        return None
    
    print(f"\n🔮 Creating embedding pipeline: {pipeline_name}")
    print(f"   Model: {embedding_model}")
    print(f"   Input format: {file_format.upper()}")
    print(f"   Text field: {text_field}")
    print(f"   Embedding field: {embedding_field}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Pooling: {pooling}")
    print(f"   Output directory: {output_dir}")
    
    # Create the pipeline
    pipeline = Pipeline(name=pipeline_name)
    
    # Stage 1: Read input data
    if file_format.lower() == 'jsonl':
        reader = JsonlReader(
            file_paths=file_paths,
            files_per_partition=4
        )
    else:
        reader = ParquetReader(
            file_paths=file_paths,
            files_per_partition=4
        )
    pipeline.add_stage(reader)
    print(f"   ✓ Added {file_format.upper()} reader stage")
    
    # Stage 2: Add embedding extraction stage
    # Using EmbeddingCreatorStage from NeMo Curator
    embedding_stage = EmbeddingCreatorStage(
        model_identifier=embedding_model,
        text_field=text_field,
        embedding_field=embedding_field,
        embedding_pooling=pooling,
        model_inference_batch_size=batch_size,
        max_seq_length=max_seq_length,
        sort_by_length=True,
        padding_side='right',
        autocast=True,
        hf_token=hf_token
    )
    pipeline.add_stage(embedding_stage)
    print(f"   ✓ Added embedding extraction stage")
    
    # Stage 3: Write output
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write as Parquet for efficient embedding storage
    writer = ParquetWriter(
        output_path=str(output_path),
        output_type="parquet"
    )
    pipeline.add_stage(writer)
    print(f"   ✓ Added Parquet writer stage")
    
    print(f"\n✅ Embedding pipeline created successfully!")
    return pipeline


def extract_embeddings_from_dataset(
    input_path: str,
    output_dir: str = str(DEFAULT_EMBEDDINGS_DIR),
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    pooling: str = DEFAULT_EMBEDDING_POOLING,
    hf_token: Optional[str] = None,
    text_columns: Optional[List[str]] = None,
    num_gpus: Optional[int] = None
) -> None:
    """
    Extract embeddings from a dataset, combining meaningful text columns.
    
    This is the main entry point for embedding extraction. It:
    1. Discovers the dataset structure and columns
    2. Identifies meaningful text columns for embedding
    3. Creates a combined text field from all text columns
    4. Extracts embeddings using the specified model (distributed across GPUs)
    5. Saves results to the output directory
    
    Args:
        input_path: Path to the input dataset (directory or file pattern)
        output_dir: Directory to save embeddings (default: /raid/embeddings)
        embedding_model: HuggingFace model identifier
        batch_size: Inference batch size
        max_seq_length: Maximum sequence length
        pooling: Pooling strategy
        hf_token: HuggingFace token for gated models
        text_columns: Specific columns to use (auto-detected if None)
        num_gpus: Number of GPUs to use (None = auto-detect all available)
        
    Example:
        >>> extract_embeddings_from_dataset(
        ...     input_path="/raid/datasets/jsonl/llama-nemotron",
        ...     embedding_model="nvidia/llama-embed-nemotron-8b",
        ...     batch_size=256,
        ...     num_gpus=8
        ... )
    """
    if not NEMO_CURATOR_AVAILABLE:
        print("❌ NeMo Curator is not installed. Cannot extract embeddings.")
        return
    
    input_path_obj = Path(input_path)
    
    print(f"\n{'='*70}")
    print(f"🔮 EMBEDDING EXTRACTION")
    print(f"{'='*70}")
    print(f"   Input: {input_path}")
    print(f"   Model: {embedding_model}")
    print(f"   Output: {output_dir}")
    
    # Determine file format
    if input_path_obj.is_dir():
        jsonl_files = list(input_path_obj.glob("*.jsonl"))
        parquet_files = list(input_path_obj.glob("*.parquet"))
        
        if jsonl_files:
            file_format = "jsonl"
            file_paths = str(input_path_obj / "*.jsonl")
            sample_file = jsonl_files[0]
        elif parquet_files:
            file_format = "parquet"
            file_paths = str(input_path_obj / "*.parquet")
            sample_file = parquet_files[0]
        else:
            print(f"❌ No JSONL or Parquet files found in {input_path}")
            return
    else:
        if str(input_path).endswith('.jsonl'):
            file_format = "jsonl"
        else:
            file_format = "parquet"
        file_paths = input_path
        sample_file = input_path_obj
    
    print(f"   Format: {file_format.upper()}")
    
    # Discover columns from sample file
    if text_columns is None:
        print(f"\n📝 Discovering text columns from: {sample_file}")
        
        if file_format == "jsonl":
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_data = json.loads(f.readline().strip())
                    all_columns = list(sample_data.keys())
            except Exception as e:
                print(f"❌ Error reading sample file: {e}")
                return
        else:
            if PYARROW_AVAILABLE:
                try:
                    schema = pq.read_schema(sample_file)
                    all_columns = [field.name for field in schema]
                except Exception as e:
                    print(f"❌ Error reading Parquet schema: {e}")
                    return
            else:
                print("❌ PyArrow required for reading Parquet schemas")
                return
        
        # Filter to embedding candidate columns
        text_columns = [col for col in all_columns if is_embedding_candidate(col)]
        
        if not text_columns:
            print(f"⚠️  No embedding candidate columns found. Using all string columns.")
            text_columns = all_columns[:5]  # Limit to first 5 columns
    
    print(f"   Text columns to combine: {text_columns}")
    
    # Determine output path (preserve input structure)
    input_name = input_path_obj.name if input_path_obj.is_dir() else input_path_obj.stem
    output_path = Path(output_dir) / input_name
    
    print(f"   Output path: {output_path}")
    
    # Create and run pipeline
    pipeline = create_embedding_pipeline(
        file_paths=file_paths,
        file_format=file_format,
        output_dir=str(output_path),
        embedding_model=embedding_model,
        text_field=text_columns[0] if len(text_columns) == 1 else "text",
        embedding_field="embeddings",
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        pooling=pooling,
        hf_token=hf_token,
        pipeline_name=f"embed_{input_name}"
    )
    
    if pipeline is None:
        return
    
    # Run the pipeline with multi-GPU support
    results = run_pipeline_with_ray(pipeline, num_gpus=num_gpus)
    
    if results is not None:
        print(f"\n{'='*70}")
        print(f"✅ EMBEDDING EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"   Output saved to: {output_path}")
        print(f"   Model used: {embedding_model}")


def process_all_datasets_embeddings(
    datasets_dir: str = str(DEFAULT_DATASETS_DIR),
    output_dir: str = str(DEFAULT_EMBEDDINGS_DIR),
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    hf_token: Optional[str] = None,
    num_gpus: Optional[int] = None
) -> None:
    """
    Process all discovered datasets and extract embeddings using multiple GPUs.
    
    This function discovers all datasets in the specified directory
    and extracts embeddings from each, saving to the output directory
    with the same structure. Uses all available GPUs by default.
    
    Args:
        datasets_dir: Directory containing datasets
        output_dir: Directory to save embeddings
        embedding_model: HuggingFace model identifier
        batch_size: Inference batch size
        hf_token: HuggingFace token for gated models
        num_gpus: Number of GPUs to use (None = auto-detect all available)
        
    Example:
        >>> process_all_datasets_embeddings(
        ...     embedding_model="nvidia/llama-embed-nemotron-8b",
        ...     num_gpus=8
        ... )
    """
    print(f"\n{'='*70}")
    print(f"🔮 BATCH EMBEDDING EXTRACTION")
    print(f"{'='*70}")
    print(f"   Datasets directory: {datasets_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Model: {embedding_model}")
    
    # Discover all datasets
    datasets = discover_all_datasets(Path(datasets_dir))
    
    if not datasets:
        print("⚠️  No datasets found.")
        return
    
    print(f"\n📊 Found {len(datasets)} datasets to process")
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'─'*70}")
        print(f"[{i}/{len(datasets)}] Processing: {dataset.name}")
        print(f"{'─'*70}")
        
        # Determine output path
        output_path = Path(output_dir) / dataset.name.replace(':', '_')
        
        # Get embedding columns
        embedding_cols = [col.name for col in dataset.embedding_columns]
        
        if not embedding_cols:
            print(f"   ⚠️  No embedding columns found, skipping...")
            continue
        
        print(f"   Embedding columns: {embedding_cols}")
        
        # Extract embeddings using all GPUs
        extract_embeddings_from_dataset(
            input_path=dataset.path,
            output_dir=str(output_path),
            embedding_model=embedding_model,
            batch_size=batch_size,
            hf_token=hf_token,
            text_columns=embedding_cols,
            num_gpus=num_gpus
        )
    
    print(f"\n{'='*70}")
    print(f"✅ BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the NeMo Curator dataset explorer.
    
    This script provides two main functionalities:
    1. Explore mode: Discover and display information about datasets
    2. Process mode: Use NeMo Curator pipelines to process datasets
    """
    parser = argparse.ArgumentParser(
        description="Explore, process, and extract embeddings from datasets using NeMo Curator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore all datasets in /raid/datasets
  python process_data_nemo_curator.py --explore
  
  # Explore with verbose output
  python process_data_nemo_curator.py --explore --verbose
  
  # Explore a specific directory
  python process_data_nemo_curator.py --explore --datasets-dir /path/to/data
  
  # Export dataset info to JSON
  python process_data_nemo_curator.py --explore --output-json datasets_info.json
  
  # Process a JSONL dataset (requires NeMo Curator)
  python process_data_nemo_curator.py --process-jsonl /raid/datasets/jsonl/llama-nemotron
  
  # Process with word count filtering
  python process_data_nemo_curator.py --process-jsonl /data --min-words 50 --max-words 1000
  
  # Extract embeddings from a single dataset
  python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron
  
  # Extract embeddings with custom model and parameters
  python process_data_nemo_curator.py --extract-embeddings /raid/datasets/jsonl/llama-nemotron \\
      --embedding-model nvidia/llama-embed-nemotron-8b \\
      --batch-size 256 \\
      --max-seq-length 32768
  
  # Extract embeddings from all discovered datasets
  python process_data_nemo_curator.py --extract-all-embeddings
  
  # Extract embeddings with HuggingFace token for gated models
  python process_data_nemo_curator.py --extract-embeddings /data \\
      --embedding-model nvidia/llama-embed-nemotron-8b \\
      --hf-token YOUR_TOKEN

Reference:
  NeMo Curator Documentation:
  - Read Data: https://docs.nvidia.com/nemo/curator/25.09/curate-text/load-data/read-existing.html
  - Embedders: https://docs.nvidia.com/nemo/curator/25.09/apidocs/stages/stages.text.embedders.base.html
        """
    )
    
    # Exploration options
    parser.add_argument(
        '--explore',
        action='store_true',
        help='Explore and display information about all datasets'
    )
    parser.add_argument(
        '--datasets-dir',
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help=f'Base directory to search for datasets (default: {DEFAULT_DATASETS_DIR})'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed column information including nested fields'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Export dataset information to JSON file'
    )
    
    # Processing options
    parser.add_argument(
        '--process-jsonl',
        type=str,
        metavar='PATH',
        help='Process JSONL files at the specified path using NeMo Curator'
    )
    parser.add_argument(
        '--process-parquet',
        type=str,
        metavar='PATH',
        help='Process Parquet files at the specified path using NeMo Curator'
    )
    parser.add_argument(
        '--fields',
        type=str,
        nargs='+',
        help='Specific fields/columns to read (optional)'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=None,
        help='Minimum word count for filtering'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=None,
        help='Maximum word count for filtering'
    )
    
    # -------------------------------------------------------------------------
    # Embedding extraction options
    # -------------------------------------------------------------------------
    embedding_group = parser.add_argument_group('Embedding Extraction Options')
    
    embedding_group.add_argument(
        '--extract-embeddings',
        type=str,
        metavar='PATH',
        help='Extract embeddings from dataset at PATH (saves to /raid/embeddings)'
    )
    embedding_group.add_argument(
        '--extract-all-embeddings',
        action='store_true',
        help='Extract embeddings from all discovered datasets'
    )
    embedding_group.add_argument(
        '--embedding-model',
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f'HuggingFace model for embeddings (default: {DEFAULT_EMBEDDING_MODEL})'
    )
    embedding_group.add_argument(
        '--embedding-output-dir',
        type=str,
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help=f'Output directory for embeddings (default: {DEFAULT_EMBEDDINGS_DIR})'
    )
    embedding_group.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=f'Batch size for embedding inference (default: {DEFAULT_EMBEDDING_BATCH_SIZE})'
    )
    embedding_group.add_argument(
        '--max-seq-length',
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f'Maximum sequence length for tokenization (default: {DEFAULT_MAX_SEQ_LENGTH})'
    )
    embedding_group.add_argument(
        '--pooling',
        type=str,
        choices=['mean_pooling', 'last_token'],
        default=DEFAULT_EMBEDDING_POOLING,
        help=f'Pooling strategy for embeddings (default: {DEFAULT_EMBEDDING_POOLING})'
    )
    embedding_group.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace token for accessing gated models (e.g., nvidia/llama-embed-nemotron-8b)'
    )
    embedding_group.add_argument(
        '--text-columns',
        type=str,
        nargs='+',
        help='Specific text columns to use for embedding (auto-detected if not specified)'
    )
    embedding_group.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help=f'Number of GPUs to use for embedding extraction (default: auto-detect all available, typically {DEFAULT_NUM_GPUS})'
    )
    
    args = parser.parse_args()
    
    # Default to explore mode if no specific action is specified
    if not any([args.explore, args.process_jsonl, args.process_parquet, 
                args.extract_embeddings, args.extract_all_embeddings]):
        args.explore = True
    
    # =========================================================================
    # EXPLORE MODE
    # =========================================================================
    if args.explore:
        datasets_dir = Path(args.datasets_dir)
        
        if not datasets_dir.exists():
            print(f"❌ Error: Datasets directory does not exist: {datasets_dir}")
            sys.exit(1)
        
        # Discover all datasets
        datasets = discover_all_datasets(datasets_dir)
        
        if not datasets:
            print("⚠️  No datasets found in the specified directory.")
            sys.exit(0)
        
        # Print summary
        print_dataset_summary(datasets, verbose=args.verbose)
        
        # Export to JSON if requested
        if args.output_json:
            output_data = {
                'base_directory': str(datasets_dir),
                'num_datasets': len(datasets),
                'datasets': [ds.to_dict() for ds in datasets]
            }
            
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n📁 Dataset information exported to: {args.output_json}")
    
    # =========================================================================
    # PROCESS JSONL MODE
    # =========================================================================
    if args.process_jsonl:
        if not NEMO_CURATOR_AVAILABLE:
            print("❌ Error: NeMo Curator is required for processing.")
            print("   Install with: pip install nemo-curator")
            sys.exit(1)
        
        # Prepare word filter if specified
        filter_words = None
        if args.min_words is not None or args.max_words is not None:
            min_w = args.min_words or 0
            max_w = args.max_words or float('inf')
            filter_words = (min_w, max_w)
        
        # Create and run pipeline
        pipeline = create_jsonl_pipeline(
            file_paths=args.process_jsonl,
            fields=args.fields
        )
        
        if pipeline:
            if filter_words and args.fields:
                word_filter = ScoreFilter(
                    filter_obj=WordCountFilter(
                        min_words=filter_words[0],
                        max_words=filter_words[1]
                    ),
                    text_field=args.fields[0]
                )
                pipeline.add_stage(word_filter)
            
            results = run_pipeline_with_ray(pipeline)
            print("\n✅ JSONL processing complete!")
    
    # =========================================================================
    # PROCESS PARQUET MODE
    # =========================================================================
    if args.process_parquet:
        if not NEMO_CURATOR_AVAILABLE:
            print("❌ Error: NeMo Curator is required for processing.")
            print("   Install with: pip install nemo-curator")
            sys.exit(1)
        
        # Prepare word filter if specified
        filter_words = None
        if args.min_words is not None or args.max_words is not None:
            min_w = args.min_words or 0
            max_w = args.max_words or float('inf')
            filter_words = (min_w, max_w)
        
        # Create and run pipeline
        pipeline = create_parquet_pipeline(
            file_paths=args.process_parquet,
            fields=args.fields
        )
        
        if pipeline:
            if filter_words and args.fields:
                word_filter = ScoreFilter(
                    filter_obj=WordCountFilter(
                        min_words=filter_words[0],
                        max_words=filter_words[1]
                    ),
                    text_field=args.fields[0]
                )
                pipeline.add_stage(word_filter)
            
            results = run_pipeline_with_ray(pipeline)
            print("\n✅ Parquet processing complete!")
    
    # =========================================================================
    # EXTRACT EMBEDDINGS MODE (Single Dataset)
    # =========================================================================
    if args.extract_embeddings:
        if not NEMO_CURATOR_AVAILABLE:
            print("❌ Error: NeMo Curator is required for embedding extraction.")
            print("   Install with: pip install nemo-curator")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("🔮 EMBEDDING EXTRACTION MODE (Multi-GPU)")
        print(f"{'='*70}")
        print(f"   Model: {args.embedding_model}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Max sequence length: {args.max_seq_length}")
        print(f"   Pooling: {args.pooling}")
        print(f"   Output directory: {args.embedding_output_dir}")
        print(f"   GPUs: {args.num_gpus if args.num_gpus else 'auto-detect (all available)'}")
        
        # Extract embeddings using multiple GPUs
        extract_embeddings_from_dataset(
            input_path=args.extract_embeddings,
            output_dir=args.embedding_output_dir,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            pooling=args.pooling,
            hf_token=args.hf_token,
            text_columns=args.text_columns,
            num_gpus=args.num_gpus
        )
    
    # =========================================================================
    # EXTRACT ALL EMBEDDINGS MODE (Batch Processing)
    # =========================================================================
    if args.extract_all_embeddings:
        if not NEMO_CURATOR_AVAILABLE:
            print("❌ Error: NeMo Curator is required for embedding extraction.")
            print("   Install with: pip install nemo-curator")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("🔮 BATCH EMBEDDING EXTRACTION MODE (Multi-GPU)")
        print(f"{'='*70}")
        print(f"   Datasets directory: {args.datasets_dir}")
        print(f"   Model: {args.embedding_model}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Output directory: {args.embedding_output_dir}")
        print(f"   GPUs: {args.num_gpus if args.num_gpus else 'auto-detect (all available)'}")
        
        # Process all datasets using multiple GPUs
        process_all_datasets_embeddings(
            datasets_dir=args.datasets_dir,
            output_dir=args.embedding_output_dir,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
            num_gpus=args.num_gpus
        )


if __name__ == "__main__":
    main()

