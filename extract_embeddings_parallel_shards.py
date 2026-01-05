#!/usr/bin/env python3
"""
Multi-GPU Parallel Embedding Extraction with Shard-Based Work Distribution

Optimized for maximum throughput with:
    - Persistent model/tokenizer per GPU
    - Pre-allocated CUDA tensors for zero-copy operations
    - cuDF for GPU-accelerated data loading (optional)
    - Pinned memory for fast CPU-GPU transfers
    - CUDA streams for async operations

Usage Examples:
    python extract_embeddings_parallel_shards.py --all --num-gpus 8 --batch-size 64
    python extract_embeddings_parallel_shards.py --splits v1:chat --dry-run
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Event, Manager, Process, Queue
from pathlib import Path
from queue import Empty
from threading import Event as ThreadEvent
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# =============================================================================
# CONSTANTS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()

DEFAULT_DATASETS_DIR = SCRIPT_DIR / "datasets"
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings_output"

SHARD_FILENAME_DIGITS = 5
PROGRESS_UPDATE_INTERVAL = 500
WORKER_QUEUE_TIMEOUT = 1
WORKER_JOIN_TIMEOUT = 10

STAGE_EMOJIS = {
    'loading': '📂',
    'extracting': '📝',
    'embedding': '🔮',
    'saving': '💾',
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'v1': {'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v1', 'subdir': 'nemotron-v1', 'config': None},
    'v2': {'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v2', 'subdir': 'nemotron-v2', 'config': None},
    'llama-sft': {'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset', 'subdir': 'llama-nemotron', 'config': 'SFT'},
    'llama-rl': {'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset', 'subdir': 'llama-nemotron', 'config': 'RL'},
    'v3-science': {'hf_name': 'nvidia/Nemotron-Science-v1', 'subdir': 'nemotron-v3/science', 'config': None},
    'v3-instruction-chat': {'hf_name': 'nvidia/Nemotron-Instruction-Following-Chat-v1', 'subdir': 'nemotron-v3/instruction-chat', 'config': None},
    'v3-math-proofs': {'hf_name': 'nvidia/Nemotron-Math-Proofs-v1', 'subdir': 'nemotron-v3/math-proofs', 'config': None},
    'v3-rl-blend': {'hf_name': 'nvidia/Nemotron-3-Nano-RL-Training-Blend', 'subdir': 'nemotron-v3/rl-blend', 'config': None},
    'v3-agentic': {'hf_name': 'nvidia/Nemotron-Agentic-v1', 'subdir': 'nemotron-v3/agentic', 'config': None},
    'v3-competitive-programming': {'hf_name': 'nvidia/Nemotron-Competitive-Programming-v1', 'subdir': 'nemotron-v3/competitive-programming', 'config': None},
    'v3-math': {'hf_name': 'nvidia/Nemotron-Math-v2', 'subdir': 'nemotron-v3/math-v2', 'config': None},
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ShardInfo:
    """Information about a dataset shard."""
    dataset_name: str
    split_name: str
    shard_idx: int
    total_shards: int
    hf_name: str
    hf_config: Optional[str]
    cache_dir: str

    @property
    def spec(self) -> str:
        return f"{self.dataset_name}:{self.split_name}:shard{self.shard_idx}"

    @property
    def parquet_filename(self) -> str:
        return (
            f"{self.dataset_name}-{self.split_name}-"
            f"{str(self.shard_idx).zfill(SHARD_FILENAME_DIGITS)}-of-"
            f"{str(self.total_shards).zfill(SHARD_FILENAME_DIGITS)}.parquet"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ['dataset_name', 'split_name', 'shard_idx', 
                'total_shards', 'hf_name', 'hf_config', 'cache_dir']}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShardInfo':
        return cls(**data)


@dataclass
class ProcessingResult:
    """Result of processing a shard."""
    status: str
    samples: int
    shard_spec: str
    time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ValidationSummary:
    """Pre-extraction validation summary."""
    existing_samples: int = 0
    existing_size_mb: float = 0.0
    embedding_dims: set = field(default_factory=set)
    invalid_files: List[Tuple[Path, List[str]]] = field(default_factory=list)
    pending_shards: List[ShardInfo] = field(default_factory=list)
    skipped_shards: List[ShardInfo] = field(default_factory=list)


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

class Config:
    """Global configuration singleton."""
    datasets_dir: Path = None
    checkpoints_dir: Path = None
    embeddings_dir: Path = None

    @classmethod
    def setup(cls, datasets_dir: Path, checkpoints_dir: Path, embeddings_dir: Path):
        cls.datasets_dir = datasets_dir
        cls.checkpoints_dir = checkpoints_dir
        cls.embeddings_dir = embeddings_dir

        for path in [datasets_dir, checkpoints_dir, embeddings_dir]:
            path.mkdir(parents=True, exist_ok=True)

        os.environ['HF_HOME'] = str(checkpoints_dir)
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(checkpoints_dir)
        os.environ['HF_MODULES_CACHE'] = str(checkpoints_dir / "modules")
        os.environ['HF_DATASETS_CACHE'] = str(datasets_dir)

    @classmethod
    def get_dataset_cache_dir(cls, dataset_name: str) -> Path:
        config = DATASET_CONFIGS.get(dataset_name)
        if not config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return cls.datasets_dir / config['subdir']


# Early initialization
def _parse_path_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datasets-dir', type=str, default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument('--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR))
    parser.add_argument('--embeddings-dir', type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    args, _ = parser.parse_known_args()
    return args

_path_args = _parse_path_args()
Config.setup(
    datasets_dir=Path(_path_args.datasets_dir),
    checkpoints_dir=Path(_path_args.checkpoints_dir),
    embeddings_dir=Path(_path_args.embeddings_dir),
)


# =============================================================================
# OPTIMIZED EMBEDDING ENGINE
# =============================================================================

class EmbeddingEngine:
    """
    High-performance embedding engine with pre-allocated buffers.
    
    Features:
        - Persistent model and tokenizer
        - Pre-allocated CUDA tensors for input/output
        - Pinned memory for fast CPU-GPU transfers
        - Optional cuDF acceleration
        - Reusable buffers to minimize allocations
    """
    
    def __init__(
        self,
        model_name: str,
        device: int,
        batch_size: int,
        max_length: int,
        input_type: str = 'document',
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.input_type = input_type
        self.console = Console()
        
        # Set CUDA device
        torch.cuda.set_device(device)
        self.cuda_device = f'cuda:{device}'
        
        # Load model and tokenizer
        self._load_model(model_name)
        
        # Pre-allocate buffers
        self._allocate_buffers()
        
        # Try to import cuDF for GPU-accelerated data loading
        self.cudf_available = self._init_cudf()
        
        # Create CUDA stream for async operations
        self.stream = torch.cuda.Stream(device=device)
        
    def _load_model(self, model_name: str):
        """Load model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer
        
        self.console.print(f"[cyan]🔄 GPU {self.device}:[/cyan] Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(Config.checkpoints_dir),
            trust_remote_code=True,
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(Config.checkpoints_dir),
            dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.cuda_device)
        
        self.model.eval()
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        
        self.console.print(f"[green]✅ GPU {self.device}:[/green] Model loaded (dim={self.embedding_dim})")
    
    def _allocate_buffers(self):
        """Pre-allocate all CUDA buffers for zero-allocation inference."""
        self.console.print(
            f"[cyan]🔥 GPU {self.device}:[/cyan] Pre-allocating buffers "
            f"(batch={self.batch_size}, len={self.max_length})..."
        )
        
        # Pre-allocated input tensors on GPU
        self.input_ids_buffer = torch.zeros(
            (self.batch_size, self.max_length),
            dtype=torch.long,
            device=self.cuda_device,
        )
        self.attention_mask_buffer = torch.zeros(
            (self.batch_size, self.max_length),
            dtype=torch.long,
            device=self.cuda_device,
        )
        
        # Pre-allocated output buffer on GPU
        self.embeddings_buffer = torch.zeros(
            (self.batch_size, self.embedding_dim),
            dtype=torch.float16,
            device=self.cuda_device,
        )
        
        # Pinned memory for fast CPU->GPU transfers
        self.input_ids_pinned = torch.zeros(
            (self.batch_size, self.max_length),
            dtype=torch.long,
            pin_memory=True,
        )
        self.attention_mask_pinned = torch.zeros(
            (self.batch_size, self.max_length),
            dtype=torch.long,
            pin_memory=True,
        )
        
        # Pinned memory for fast GPU->CPU transfers (embeddings output)
        self.embeddings_pinned = torch.zeros(
            (self.batch_size, self.embedding_dim),
            dtype=torch.float32,  # float32 for numpy compatibility
            pin_memory=True,
        )
        
        # Warmup forward pass to allocate all intermediate buffers
        self._warmup()
        
        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        
        self.console.print(
            f"[green]✅ GPU {self.device}:[/green] Buffers allocated: "
            f"{allocated_mb:.0f}MB used, {reserved_mb:.0f}MB reserved"
        )
    
    def _warmup(self):
        """Warmup forward pass to allocate all CUDA memory."""
        # Fill with dummy data
        self.input_ids_buffer.fill_(1)
        self.attention_mask_buffer.fill_(1)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=self.input_ids_buffer,
                attention_mask=self.attention_mask_buffer,
            )
            # Compute embeddings
            hidden = outputs.last_hidden_state
            embeddings = hidden.mean(dim=1)
            torch.nn.functional.normalize(embeddings, p=2, dim=1, out=self.embeddings_buffer)
        
        # Clear dummy data
        self.input_ids_buffer.zero_()
        self.attention_mask_buffer.zero_()
        self.embeddings_buffer.zero_()
        
        torch.cuda.synchronize(self.device)
    
    def _init_cudf(self) -> bool:
        """Try to initialize cuDF for GPU-accelerated data loading."""
        try:
            import cudf
            self.cudf = cudf
            self.console.print(f"[green]✅ GPU {self.device}:[/green] cuDF available for fast data loading")
            return True
        except ImportError:
            self.console.print(f"[dim]ℹ️  GPU {self.device}: cuDF not available, using pandas[/dim]")
            return False
    
    def tokenize_batch(self, texts: List[str], actual_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize texts using pre-allocated buffers.
        
        Uses pinned memory for fast async CPU->GPU transfer.
        """
        # Add prefix for Nemotron models
        if 'nemotron' in self.tokenizer.name_or_path.lower():
            prefix = "query: " if self.input_type == 'query' else "passage: "
            texts = [f"{prefix}{t}" for t in texts]
        
        # Tokenize to CPU (will go to pinned memory)
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Copy to pinned memory (zero-copy if already there)
        batch_size = actual_batch_size
        self.input_ids_pinned[:batch_size].copy_(encoded['input_ids'])
        self.attention_mask_pinned[:batch_size].copy_(encoded['attention_mask'])
        
        # Async copy to GPU using stream
        with torch.cuda.stream(self.stream):
            self.input_ids_buffer[:batch_size].copy_(
                self.input_ids_pinned[:batch_size], non_blocking=True
            )
            self.attention_mask_buffer[:batch_size].copy_(
                self.attention_mask_pinned[:batch_size], non_blocking=True
            )
        
        return self.input_ids_buffer[:batch_size], self.attention_mask_buffer[:batch_size]
    
    @torch.inference_mode()
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a batch using pre-allocated buffers.
        
        This is the hot path - optimized for minimal allocations.
        """
        actual_batch_size = len(texts)
        
        # Tokenize with async GPU transfer
        input_ids, attention_mask = self.tokenize_batch(texts, actual_batch_size)
        
        # Wait for transfer to complete
        self.stream.synchronize()
        
        return self._forward_pass(input_ids, attention_mask, actual_batch_size)
    
    @torch.inference_mode()
    def compute_embeddings_pretokenized(self, input_ids: torch.Tensor, 
                                         attention_mask: torch.Tensor) -> np.ndarray:
        """
        Compute embeddings from pre-tokenized tensors.
        
        Used with DataPrefetcher for maximum throughput.
        """
        actual_batch_size = input_ids.shape[0]
        
        # Copy to pinned memory
        self.input_ids_pinned[:actual_batch_size].copy_(input_ids)
        self.attention_mask_pinned[:actual_batch_size].copy_(attention_mask)
        
        # Async copy to GPU
        with torch.cuda.stream(self.stream):
            self.input_ids_buffer[:actual_batch_size].copy_(
                self.input_ids_pinned[:actual_batch_size], non_blocking=True
            )
            self.attention_mask_buffer[:actual_batch_size].copy_(
                self.attention_mask_pinned[:actual_batch_size], non_blocking=True
            )
        
        self.stream.synchronize()
        
        return self._forward_pass(
            self.input_ids_buffer[:actual_batch_size],
            self.attention_mask_buffer[:actual_batch_size],
            actual_batch_size
        )
    
    def _forward_pass(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      actual_batch_size: int) -> np.ndarray:
        """Core forward pass logic."""
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Mean pooling over sequence dimension
            hidden_states = outputs.last_hidden_state
            
            # Compute mean only over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).half()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Async copy to pinned memory for fast GPU->CPU transfer
        self.embeddings_pinned[:actual_batch_size].copy_(
            embeddings.float(), non_blocking=True
        )
        
        # Synchronize and return numpy array
        torch.cuda.synchronize(self.device)
        
        return self.embeddings_pinned[:actual_batch_size].numpy().copy()
    
    def load_data_fast(self, path: str) -> Any:
        """
        Load parquet data using cuDF if available, else pandas.
        
        cuDF loads data directly to GPU memory for faster processing.
        """
        if self.cudf_available:
            try:
                return self.cudf.read_parquet(path)
            except Exception:
                pass  # Fallback to pandas
        
        import pandas as pd
        return pd.read_parquet(path)


# =============================================================================
# TEXT EXTRACTION (MULTI-THREADED)
# =============================================================================

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Number of CPU workers for data loading
NUM_DATA_WORKERS = min(8, mp.cpu_count())


def extract_text_from_sample(sample: Dict[str, Any]) -> str:
    """Extract text from dataset sample."""
    # Messages format
    if 'messages' in sample:
        messages = sample['messages']
        if isinstance(messages, list):
            texts = [m['content'] for m in messages if isinstance(m, dict) and 'content' in m]
            if texts:
                return "\n\n".join(texts)
        elif isinstance(messages, str):
            return messages

    # Conversation format
    if 'conversation' in sample:
        conv = sample['conversation']
        if isinstance(conv, list):
            texts = [t['content'] for t in conv if isinstance(t, dict) and 'content' in t]
            if texts:
                return "\n\n".join(texts)

    # Direct text fields
    for field in ['text', 'content', 'instruction', 'prompt', 'question']:
        if field in sample and sample[field]:
            return str(sample[field])

    # Fallback
    texts = [str(v) for v in sample.values() if isinstance(v, str) and len(v) > 10]
    return "\n\n".join(texts) if texts else ""


def _extract_single(args: Tuple[int, Dict]) -> Tuple[int, str]:
    """Helper for parallel extraction."""
    idx, sample = args
    text = extract_text_from_sample(sample)
    return (idx, text) if text and text.strip() else (idx, "")


def extract_texts_parallel(samples: List[Dict], num_workers: int = NUM_DATA_WORKERS) -> Tuple[List[str], List[int]]:
    """
    Extract texts using multiple CPU threads.
    
    Much faster than sequential for large batches.
    """
    if len(samples) < 100:
        # Small batch - sequential is faster due to overhead
        texts, indices = [], []
        for idx, sample in enumerate(samples):
            text = extract_text_from_sample(sample)
            if text and text.strip():
                texts.append(text)
                indices.append(idx)
        return texts, indices
    
    # Large batch - use thread pool
    texts, indices = [], []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_extract_single, enumerate(samples)))
    
    for idx, text in results:
        if text:
            texts.append(text)
            indices.append(idx)
    
    return texts, indices


class TextDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for text tokenization.
    
    Used with DataLoader for multi-process prefetching.
    """
    
    def __init__(self, texts: List[str], indices: List[int], tokenizer, 
                 max_length: int, input_type: str = 'document'):
        self.texts = texts
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_type = input_type
        
        # Determine prefix once
        self.prefix = ""
        if 'nemotron' in tokenizer.name_or_path.lower():
            self.prefix = "query: " if input_type == 'query' else "passage: "
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize single text (called by DataLoader workers)."""
        text = self.prefix + self.texts[idx]
        
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'original_idx': self.indices[idx],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'original_indices': [b['original_idx'] for b in batch],
        'batch_size': len(batch),
    }


def create_dataloader(
    texts: List[str],
    indices: List[int],
    tokenizer,
    batch_size: int,
    max_length: int,
    input_type: str = 'document',
    num_workers: int = 8,
    prefetch_factor: int = 4,
) -> torch.utils.data.DataLoader:
    """
    Create optimized DataLoader with multi-process prefetching.
    
    Args:
        texts: List of text strings
        indices: Original indices in dataset
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        input_type: 'document' or 'query'
        num_workers: Number of worker processes for data loading
        prefetch_factor: Batches to prefetch per worker
    
    Returns:
        DataLoader with prefetching enabled
    """
    dataset = TextDataset(
        texts=texts,
        indices=indices,
        tokenizer=tokenizer,
        max_length=max_length,
        input_type=input_type,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # Fast CPU->GPU transfer
        prefetch_factor=prefetch_factor,  # Prefetch batches per worker
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        collate_fn=collate_fn,
        drop_last=False,
    )


# =============================================================================
# SHARD PROCESSING (OPTIMIZED)
# =============================================================================

def process_shard_optimized(
    shard_info: ShardInfo,
    engine: EmbeddingEngine,
    output_dir: Path,
    progress_dict: Optional[Dict] = None,
) -> ProcessingResult:
    """
    Process a shard using the optimized EmbeddingEngine.
    
    Key optimizations:
        - Multi-threaded text extraction (CPU)
        - Prefetched tokenization in background threads
        - Pre-allocated GPU buffers (zero allocation in hot path)
        - Async CUDA operations with pinned memory
    """
    from datasets import Dataset, load_dataset
    import pyarrow.parquet as pq

    console = engine.console
    device = engine.device
    parquet_filename = shard_info.parquet_filename

    output_subdir = output_dir / shard_info.dataset_name / shard_info.split_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_subdir / parquet_filename

    # Skip if exists
    if parquet_path.exists():
        console.print(f"   [dim]⏭️  GPU {device}: Skipping {parquet_filename}[/dim]")
        try:
            existing = pq.read_table(str(parquet_path))
            return ProcessingResult('skipped', len(existing), shard_info.spec)
        except Exception:
            pass

    start_time = time.time()

    def update_progress(current: int, total: int, stage: str):
        if progress_dict is not None:
            progress_dict[device] = {
                'current': current, 'total': total,
                'filename': parquet_filename, 'stage': stage,
            }

    def clear_progress():
        if progress_dict is not None and device in progress_dict:
            del progress_dict[device]

    try:
        # Load dataset
        load_args = {
            'path': shard_info.hf_name,
            'cache_dir': shard_info.cache_dir,
            'split': shard_info.split_name,
        }
        if shard_info.hf_config:
            load_args['name'] = shard_info.hf_config

        dataset = load_dataset(**load_args)
        shard_data = dataset.shard(num_shards=shard_info.total_shards, index=shard_info.shard_idx)
        num_samples = len(shard_data)

        console.print(f"[cyan]🔄 GPU {device}:[/cyan] {parquet_filename} ({num_samples:,} samples)")
        update_progress(0, num_samples, 'loading')

        # ===== OPTIMIZATION: Parallel text extraction =====
        update_progress(0, num_samples, 'extracting')
        texts, indices = extract_texts_parallel(list(shard_data), num_workers=NUM_DATA_WORKERS)

        if not texts:
            console.print(f"[yellow]⚠️  GPU {device}:[/yellow] No text in {parquet_filename}")
            clear_progress()
            return ProcessingResult('no_text', 0, shard_info.spec)

        total_texts = len(texts)
        batch_size = engine.batch_size

        # ===== OPTIMIZATION: PyTorch DataLoader with multi-process prefetching =====
        dataloader = create_dataloader(
            texts=texts,
            indices=indices,
            tokenizer=engine.tokenizer,
            batch_size=batch_size,
            max_length=engine.max_length,
            input_type=engine.input_type,
            num_workers=NUM_DATA_WORKERS,  # Worker processes for tokenization
            prefetch_factor=4,  # Prefetch 4 batches per worker
        )

        # Process batches - GPU never waits thanks to prefetching
        all_embeddings = []
        all_indices = []
        processed = 0

        for batch_data in dataloader:
            # Batch is pre-tokenized and in pinned memory, fast GPU transfer
            batch_embeddings = engine.compute_embeddings_pretokenized(
                batch_data['input_ids'],
                batch_data['attention_mask'],
            )
            all_embeddings.append(batch_embeddings)
            all_indices.extend(batch_data['original_indices'])
            
            processed += batch_data['batch_size']
            update_progress(processed, total_texts, 'embedding')

        # Cleanup dataloader workers
        del dataloader

        # Concatenate and save
        update_progress(total_texts, total_texts, 'saving')
        embeddings_array = np.vstack(all_embeddings)

        embedding_dataset = Dataset.from_dict({
            'embeddings': embeddings_array.tolist(),
            'original_index': all_indices,
        })
        embedding_dataset.to_parquet(str(parquet_path))

        clear_progress()

        # Cleanup (preserve engine buffers)
        num_texts = len(texts)
        del texts, indices, all_indices, all_embeddings, embeddings_array, embedding_dataset
        del shard_data, dataset
        gc.collect()
        # NOTE: Don't empty_cache - preserve pre-allocated pools

        elapsed = time.time() - start_time
        speed = num_texts / elapsed if elapsed > 0 else 0

        console.print(
            f"[green]✅ GPU {device}:[/green] {parquet_filename} - "
            f"[yellow]{num_texts:,}[/yellow] in [cyan]{elapsed:.1f}s[/cyan] "
            f"([magenta]{speed:.1f}[/magenta]/s)"
        )

        return ProcessingResult('success', num_texts, shard_info.spec, elapsed)

    except Exception as e:
        console.print(f"[red]❌ GPU {device}:[/red] {parquet_filename}: {e}")
        traceback.print_exc()
        clear_progress()
        gc.collect()
        torch.cuda.empty_cache()
        return ProcessingResult('error', 0, shard_info.spec, error=str(e))


# =============================================================================
# GPU WORKER
# =============================================================================

def gpu_worker(
    gpu_id: int,
    work_queue: Queue,
    results_queue: Queue,
    model_name: str,
    batch_size: int,
    max_length: int,
    input_type: str,
    output_dir: Path,
    progress_dict: Dict,
    shutdown_event: Event,
) -> None:
    """GPU worker with persistent EmbeddingEngine."""
    console = Console()

    try:
        # Create optimized embedding engine (persistent for this worker)
        engine = EmbeddingEngine(
            model_name=model_name,
            device=gpu_id,
            batch_size=batch_size,
            max_length=max_length,
            input_type=input_type,
        )

        # Process shards
        while not shutdown_event.is_set():
            try:
                shard_dict = work_queue.get(timeout=WORKER_QUEUE_TIMEOUT)

                if shard_dict is None:
                    console.print(f"[dim]🛑 GPU {gpu_id}: Shutdown[/dim]")
                    break

                shard_info = ShardInfo.from_dict(shard_dict)
                result = process_shard_optimized(shard_info, engine, output_dir, progress_dict)

                results_queue.put({
                    'gpu_id': gpu_id,
                    'result': {
                        'status': result.status,
                        'samples': result.samples,
                        'shard': result.shard_spec,
                        'time': result.time_seconds,
                        'error': result.error,
                    },
                })

            except Empty:
                continue
            except Exception as e:
                console.print(f"[red]❌ GPU {gpu_id}:[/red] {e}")
                traceback.print_exc()

        console.print(f"[green]✅ GPU {gpu_id}: Done[/green]")

    except Exception as e:
        console.print(f"[red]❌ GPU {gpu_id}:[/red] Fatal: {e}")
        traceback.print_exc()


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def discover_all_splits(console: Console) -> List[str]:
    """Discover all available splits."""
    from datasets import load_dataset

    all_splits = []
    console.print("[cyan]🔍 Scanning datasets...[/cyan]")

    for name, config in DATASET_CONFIGS.items():
        cache_dir = Config.get_dataset_cache_dir(name)
        if not cache_dir.exists():
            console.print(f"   [dim]⏭️  {name}: Not downloaded[/dim]")
            continue

        try:
            load_args = {'path': config['hf_name'], 'cache_dir': str(cache_dir)}
            if config['config']:
                load_args['name'] = config['config']
            ds = load_dataset(**load_args)
            splits = [s for s in ds.keys() if 'multilingual' not in s.lower()]
            if splits:
                console.print(f"   [green]✅ {name}:[/green] {', '.join(splits)}")
                all_splits.extend(f"{name}:{s}" for s in splits)
        except Exception as e:
            console.print(f"   [yellow]⚠️  {name}:[/yellow] {e}")

    return all_splits


def discover_dataset_shards(splits: List[str], console: Console) -> List[ShardInfo]:
    """Discover shards for specified splits."""
    from datasets import load_dataset

    all_shards = []

    for spec in splits:
        if ':' not in spec:
            continue
        dataset_name, split_name = spec.split(':', 1)
        if dataset_name not in DATASET_CONFIGS:
            continue

        config = DATASET_CONFIGS[dataset_name]
        cache_dir = Config.get_dataset_cache_dir(dataset_name)
        if not cache_dir.exists():
            continue

        try:
            console.print(f"[cyan]🔍 {spec}...[/cyan]")
            load_args = {'path': config['hf_name'], 'cache_dir': str(cache_dir)}
            if config['config']:
                load_args['name'] = config['config']
            ds = load_dataset(**load_args)

            if split_name not in ds:
                continue

            split_ds = ds[split_name]
            num_shards = len(split_ds.cache_files) if hasattr(split_ds, 'cache_files') and split_ds.cache_files else 1

            console.print(f"[green]✅ {spec}:[/green] {len(split_ds):,} samples, {num_shards} shard(s)")

            for idx in range(num_shards):
                all_shards.append(ShardInfo(
                    dataset_name=dataset_name,
                    split_name=split_name,
                    shard_idx=idx,
                    total_shards=num_shards,
                    hf_name=config['hf_name'],
                    hf_config=config['config'],
                    cache_dir=str(cache_dir),
                ))
        except Exception as e:
            console.print(f"[yellow]⚠️  {spec}:[/yellow] {e}")

    all_shards.sort(key=lambda s: (s.dataset_name, s.split_name, s.shard_idx))
    return all_shards


# =============================================================================
# VALIDATION
# =============================================================================

def validate_existing_embeddings(shards: List[ShardInfo], console: Console) -> ValidationSummary:
    """Check existing embeddings."""
    import pyarrow.parquet as pq

    console.print(Panel.fit(
        "[bold cyan]📊 Pre-Extraction Validation[/bold cyan]",
        border_style="cyan",
    ))

    summary = ValidationSummary()
    embeddings_dir = Config.embeddings_dir

    if embeddings_dir.exists():
        for dataset_dir in embeddings_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for split_dir in dataset_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                for f in split_dir.glob("*.parquet"):
                    try:
                        size = f.stat().st_size / (1024 * 1024)
                        table = pq.read_table(str(f))
                        summary.existing_samples += len(table)
                        summary.existing_size_mb += size
                    except Exception:
                        pass

    for shard in shards:
        path = embeddings_dir / shard.dataset_name / shard.split_name / shard.parquet_filename
        if path.exists():
            summary.skipped_shards.append(shard)
        else:
            summary.pending_shards.append(shard)

    console.print(f"   Existing: [cyan]{summary.existing_samples:,}[/cyan] samples")
    console.print(f"   Complete: [green]{len(summary.skipped_shards)}[/green] shards")
    console.print(f"   Pending:  [yellow]{len(summary.pending_shards)}[/yellow] shards\n")

    return summary


# =============================================================================
# PROGRESS DISPLAY
# =============================================================================

def create_progress_thread(progress_dict: Dict, console: Console, stop: ThreadEvent) -> Thread:
    """Create progress display thread."""
    def loop():
        with Live(console=console, refresh_per_second=2) as live:
            while stop.is_set():
                table = Table(title="GPU Progress", header_style="bold cyan")
                table.add_column("GPU", width=6)
                table.add_column("Stage", width=15)
                table.add_column("File", width=45)
                table.add_column("Progress", width=25)

                if progress_dict:
                    for gid in sorted(progress_dict.keys()):
                        try:
                            info = progress_dict[gid]
                            fname = info.get('filename', '?')
                            if len(fname) > 40:
                                fname = "..." + fname[-37:]
                            cur, tot = info.get('current', 0), info.get('total', 1)
                            pct = cur / tot * 100 if tot > 0 else 0
                            stage = info.get('stage', '?')
                            emoji = STAGE_EMOJIS.get(stage, '⚙️')
                            table.add_row(f"GPU {gid}", f"{emoji} {stage}", fname, f"{cur:,}/{tot:,} ({pct:.0f}%)")
                        except KeyError:
                            continue
                else:
                    table.add_row("—", "—", "Idle", "—")

                live.update(table)
                time.sleep(0.5)

    return Thread(target=loop, daemon=True)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU embedding extraction")
    parser.add_argument('--datasets-dir', type=str, default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument('--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR))
    parser.add_argument('--embeddings-dir', type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-text-length', type=int, default=8192)
    parser.add_argument('--model', default='nvidia/llama-embed-nemotron-8b')
    parser.add_argument('--input-type', default='document', choices=['document', 'query'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--output', default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    if not args.all and not args.splits:
        console.print("[red]❌ Specify --splits or --all[/red]")
        sys.exit(1)

    if args.output:
        Config.embeddings_dir = Path(args.output)

    output_dir = Config.embeddings_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]Multi-GPU Embedding Extraction[/bold cyan]\n"
        "[dim]Optimized with pre-allocated buffers & pinned memory[/dim]",
        border_style="cyan",
    ))

    # Discover splits
    if args.all:
        splits = sorted(discover_all_splits(console))
        if not splits:
            console.print("[red]❌ No datasets found[/red]")
            sys.exit(1)
    else:
        splits = args.splits

    # Print config
    console.print(f"\n[bold]Config:[/bold] {args.num_gpus} GPUs, batch={args.batch_size}, len={args.max_text_length}")
    console.print(f"[bold]Model:[/bold] {args.model}\n")

    # Discover shards
    shards = discover_dataset_shards(splits, console)
    if not shards:
        console.print("[red]❌ No shards found[/red]")
        sys.exit(1)

    console.print(f"[green]✅ Found {len(shards)} shard(s)[/green]\n")

    # Validation
    if not args.skip_validation:
        summary = validate_existing_embeddings(shards, console)

        if args.dry_run:
            console.print(Panel.fit(
                f"[yellow]DRY RUN[/yellow]: Would process {len(summary.pending_shards)} shards",
                border_style="yellow",
            ))
            sys.exit(0)

        if not summary.pending_shards and not args.force:
            console.print(Panel.fit("[green]✅ All complete[/green]", border_style="green"))
            sys.exit(0)

        if not args.force:
            shards = summary.pending_shards

    # Setup queues
    manager = Manager()
    work_queue = manager.Queue()
    results_queue = manager.Queue()
    progress_dict = manager.dict()
    shutdown = manager.Event()

    for shard in shards:
        work_queue.put(shard.to_dict())
    for _ in range(args.num_gpus):
        work_queue.put(None)

    console.print(f"[cyan]📋 Queue: {len(shards)} shards[/cyan]\n")

    # Start workers
    console.print(f"[bold cyan]🚀 Starting {args.num_gpus} workers...[/bold cyan]")
    workers = []
    for gid in range(args.num_gpus):
        w = Process(target=gpu_worker, args=(
            gid, work_queue, results_queue, args.model,
            args.batch_size, args.max_text_length, args.input_type,
            output_dir, progress_dict, shutdown,
        ))
        w.start()
        workers.append(w)

    console.print("[green]✅ Workers started[/green]\n")

    # Progress display
    progress_active = ThreadEvent()
    progress_active.set()
    progress_thread = create_progress_thread(progress_dict, console, progress_active)
    progress_thread.start()

    console.print("=" * 70)

    # Collect results
    completed, total_samples = 0, 0
    start = time.time()

    try:
        while completed < len(shards):
            try:
                info = results_queue.get(timeout=1)
                completed += 1
                r = info['result']
                gid = info['gpu_id']

                if r['status'] == 'success':
                    total_samples += r['samples']
                    elapsed = time.time() - start
                    speed = total_samples / elapsed if elapsed > 0 else 0
                    console.print(f"[green]✅ [{completed}/{len(shards)}][/green] GPU {gid}: {r['shard']} - {r['samples']:,} | {speed:.0f}/s")
                elif r['status'] == 'skipped':
                    total_samples += r['samples']
                    console.print(f"[dim]⏭️  [{completed}/{len(shards)}] GPU {gid}: {r['shard']}[/dim]")
                else:
                    console.print(f"[yellow]⚠️  [{completed}/{len(shards)}][/yellow] GPU {gid}: {r['shard']} - {r['status']}")

            except Empty:
                continue

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        shutdown.set()

    progress_active.clear()
    if progress_thread.is_alive():
        progress_thread.join(timeout=2)

    console.print("\n[cyan]Waiting for workers...[/cyan]")
    for w in workers:
        w.join(timeout=WORKER_JOIN_TIMEOUT)
        if w.is_alive():
            w.terminate()

    # Summary
    elapsed = time.time() - start
    console.print()
    console.print(Panel.fit(
        f"[bold green]✅ COMPLETE[/bold green]\n\n"
        f"Shards: {completed}\n"
        f"Samples: {total_samples:,}\n"
        f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)\n"
        f"Speed: {total_samples/elapsed:.0f}/s" if elapsed > 0 else "",
        border_style="green",
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ {e}")
        traceback.print_exc()
        sys.exit(1)
