#!/usr/bin/env python3
"""
Multi-GPU Parallel Embedding Extraction with Shard-Based Work Distribution

This script efficiently extracts embeddings from dataset shards using multiple GPUs.
It uses a work queue system where each GPU processes individual shards for maximum utilization.

Features:
- Shard-based work distribution for optimal load balancing
- Multiple GPUs can work on the same split simultaneously
- Automatic discovery of dataset shards
- Progress tracking and error handling
- Checkpointing to avoid reprocessing
- Memory-efficient batch processing

Usage:
    python extract_embeddings_parallel_shards.py \
        --splits v1:chat v2:math llama-sft:safety \
        --num-gpus 8 \
        --batch-size 32 \
        --max-text-length 8192
"""

import os
import sys
from pathlib import Path

# Set HuggingFace cache directories BEFORE importing any HF libraries
SCRIPT_DIR = Path(__file__).parent.absolute()
DATASETS_DIR = SCRIPT_DIR / "datasets"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

# Ensure directories exist
DATASETS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Set environment variables for HuggingFace cache
# Models/hub go to checkpoints
os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)
os.environ['HF_MODULES_CACHE'] = str(CHECKPOINTS_DIR / "modules")

# Datasets go to datasets folder
os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)

# Now safe to import other libraries
import argparse
import time
from multiprocessing import Process, Queue, Manager, Event
from queue import Empty
from typing import List, Dict, Optional
import json
from datetime import datetime

import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint


def discover_all_splits(datasets_dir: Path) -> List[str]:
    """
    Discover all available dataset splits across all datasets.
    
    Args:
        datasets_dir: Base datasets directory
    
    Returns:
        List of split specifications (e.g., ['v1:chat', 'v1:math', ...])
    """
    from datasets import load_dataset
    
    console = Console()
    all_splits = []
    
    # Define all possible datasets and their configurations
    datasets_to_check = {
        'v1': {
            'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v1',
            'cache_dir': str(datasets_dir / 'nemotron-v1'),
            'config': None
        },
        'v2': {
            'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v2',
            'cache_dir': str(datasets_dir / 'nemotron-v2'),
            'config': None
        },
        'llama-sft': {
            'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset',
            'cache_dir': str(datasets_dir / 'llama-nemotron'),
            'config': 'SFT'
        },
        'llama-rl': {
            'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset',
            'cache_dir': str(datasets_dir / 'llama-nemotron'),
            'config': 'RL'
        },
    }
    
    console.print("[cyan]🔍 Scanning for available datasets...[/cyan]")
    console.print("   [dim](Excluding multilingual splits)[/dim]")
    
    for dataset_name, config in datasets_to_check.items():
        cache_dir = Path(config['cache_dir'])
        
        # Skip if dataset directory doesn't exist
        if not cache_dir.exists():
            console.print(f"   [dim]⏭️  {dataset_name}: Not downloaded[/dim]")
            continue
        
        try:
            # Load the dataset
            load_args = {
                'path': config['hf_name'],
                'cache_dir': config['cache_dir']
            }
            
            if config['config']:
                load_args['name'] = config['config']
            
            dataset = load_dataset(**load_args)
            
            # Get all splits and filter out multilingual
            all_dataset_splits = list(dataset.keys())
            splits = [s for s in all_dataset_splits if 'multilingual' not in s.lower()]
            multilingual_splits = [s for s in all_dataset_splits if 'multilingual' in s.lower()]
            
            if splits:
                console.print(f"   [green]✅ {dataset_name}:[/green] Found {len(splits)} split(s) - {', '.join(splits)}")
                
                # Show excluded multilingual splits
                if multilingual_splits:
                    console.print(f"      [dim]Excluded multilingual: {', '.join(multilingual_splits)}[/dim]")
                
                # Add all splits to the list
                for split in splits:
                    all_splits.append(f"{dataset_name}:{split}")
            else:
                console.print(f"   [yellow]⚠️  {dataset_name}:[/yellow] No splits found")
                
        except Exception as e:
            console.print(f"   [yellow]⚠️  {dataset_name}:[/yellow] Could not load - {e}")
            console.print(f"   [dim]Skipping {dataset_name}[/dim]")
            continue
    
    return all_splits


def discover_dataset_shards(datasets_dir: Path, splits: List[str]) -> List[Dict[str, any]]:
    """
    Discover dataset shards for the specified splits.
    
    Args:
        datasets_dir: Base datasets directory
        splits: List of split specifications (e.g., "v1:chat", "llama-sft:safety")
    
    Returns:
        List of dictionaries with shard information
    """
    from datasets import load_dataset
    
    console = Console()
    all_shards = []
    
    for split_spec in splits:
        # Parse split specification
        if ':' in split_spec:
            dataset_name, split_name = split_spec.split(':', 1)
        else:
            console.print(f"[yellow]⚠️  Warning:[/yellow] Invalid split format '{split_spec}', expected 'dataset:split'")
            continue
        
        # Map dataset names to HuggingFace dataset names and cache dirs
        dataset_configs = {
            'v1': {
                'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v1',
                'cache_dir': str(datasets_dir / 'nemotron-v1'),
                'config': None
            },
            'v2': {
                'hf_name': 'nvidia/Nemotron-Post-Training-Dataset-v2',
                'cache_dir': str(datasets_dir / 'nemotron-v2'),
                'config': None
            },
            'llama-sft': {
                'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset',
                'cache_dir': str(datasets_dir / 'llama-nemotron'),
                'config': 'SFT'
            },
            'llama-rl': {
                'hf_name': 'nvidia/Llama-Nemotron-Post-Training-Dataset',
                'cache_dir': str(datasets_dir / 'llama-nemotron'),
                'config': 'RL'
            },
        }
        
        if dataset_name not in dataset_configs:
            console.print(f"[yellow]⚠️  Warning:[/yellow] Unknown dataset '{dataset_name}'")
            continue
        
        config = dataset_configs[dataset_name]
        cache_dir = Path(config['cache_dir'])
        
        if not cache_dir.exists():
            console.print(f"[yellow]⚠️  Warning:[/yellow] Dataset directory not found: {cache_dir}")
            console.print(f"   [dim]Run: python download_nemotron_datasets.py --{dataset_name.replace('-', '')}[/dim]")
            continue
        
        try:
            console.print(f"[cyan]🔍 Loading {split_spec}...[/cyan]")
            
            # Load the dataset
            load_args = {
                'path': config['hf_name'],
                'cache_dir': config['cache_dir']
            }
            
            if config['config']:
                load_args['name'] = config['config']
            
            dataset = load_dataset(**load_args)
            
            # Check if the split exists
            if split_name not in dataset:
                console.print(f"[yellow]⚠️  Warning:[/yellow] Split '{split_name}' not found in {dataset_name}")
                console.print(f"   [dim]Available splits: {list(dataset.keys())}[/dim]")
                continue
            
            split_dataset = dataset[split_name]
            num_samples = len(split_dataset)
            
            # Get the actual number of shards by checking the dataset's cache files
            num_shards = 1
            
            try:
                # Method 1: Use dataset's internal cache_files property
                if hasattr(split_dataset, 'cache_files') and split_dataset.cache_files:
                    num_shards = len(split_dataset.cache_files)
                    console.print(f"   [dim]Detected {num_shards} cache files[/dim]")
                
                # Method 2: Check _data.tables if Method 1 didn't work
                elif hasattr(split_dataset, '_data') and hasattr(split_dataset._data, 'tables'):
                    tables = split_dataset._data.tables
                    if tables:
                        num_shards = len(tables)
                        console.print(f"   [dim]Detected {num_shards} data tables[/dim]")
                
                # Method 3: Manually count arrow files in the cache directory
                else:
                    from pathlib import Path as P
                    cache_path = P(cache_dir)
                    
                    # Find the specific dataset cache directory
                    # Pattern: cache_dir / dataset_hash / split_name / *.arrow
                    arrow_files = []
                    
                    # Search for arrow files that match this split
                    for arrow_file in cache_path.rglob("*.arrow"):
                        # Check if file is in a directory named after the split
                        # or if the filename contains the split name
                        path_str = str(arrow_file)
                        if f"/{split_name}/" in path_str or f"-{split_name}-" in arrow_file.name:
                            arrow_files.append(arrow_file)
                    
                    if arrow_files:
                        num_shards = len(arrow_files)
                        console.print(f"   [dim]Found {num_shards} arrow file(s) in cache[/dim]")
                
            except Exception as e:
                console.print(f"   [yellow]⚠️  Could not detect shards: {e}, using 1[/yellow]")
                num_shards = 1
            
            console.print(
                f"[green]✅ Found {split_spec}:[/green] "
                f"{num_samples:,} samples in {num_shards} shard(s)"
            )
            
            # Create shard entries
            for shard_idx in range(num_shards):
                all_shards.append({
                    'dataset_name': dataset_name,
                    'split_name': split_name,
                    'shard_idx': shard_idx,
                    'total_shards': num_shards,
                    'hf_name': config['hf_name'],
                    'hf_config': config['config'],
                    'cache_dir': config['cache_dir'],
                    'spec': f"{split_spec}:shard{shard_idx}"
                })
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning:[/yellow] Could not load {split_spec}: {e}")
            continue
    
    # Sort by dataset and split for organized processing
    all_shards.sort(key=lambda x: (x['dataset_name'], x['split_name'], x['shard_idx']))
    
    return all_shards


def load_model(model_name: str, device: int):
    """
    Load the embedding model on the specified GPU.
    
    Args:
        model_name: HuggingFace model name
        device: GPU device ID
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModel
    
    console = Console()
    console.print(f"[cyan]🔄 GPU {device}:[/cyan] Loading model {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='./checkpoints',
        trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir='./checkpoints',
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(f'cuda:{device}')
    
    model.eval()
    
    console.print(f"[green]✅ GPU {device}:[/green] Model loaded successfully")
    
    return model, tokenizer


def extract_text_from_sample(sample: dict) -> str:
    """
    Extract text from a dataset sample.
    Handles different dataset formats (chat, instruction, text).
    
    Args:
        sample: Dictionary containing the dataset sample
    
    Returns:
        Extracted text string
    """
    # Common text fields to check
    text_fields = ['text', 'content', 'instruction', 'prompt', 'question']
    
    # Check for conversation/messages format
    if 'messages' in sample:
        messages = sample['messages']
        if isinstance(messages, list):
            # Concatenate all message contents
            texts = []
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    texts.append(msg['content'])
            return "\n\n".join(texts)
        elif isinstance(messages, str):
            return messages
    
    # Check for conversation field
    if 'conversation' in sample:
        conv = sample['conversation']
        if isinstance(conv, list):
            texts = []
            for turn in conv:
                if isinstance(turn, dict) and 'content' in turn:
                    texts.append(turn['content'])
            return "\n\n".join(texts)
    
    # Check standard text fields
    for field in text_fields:
        if field in sample:
            return str(sample[field])
    
    # If nothing found, concatenate all string values
    texts = []
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 10:
            texts.append(value)
    
    if texts:
        return "\n\n".join(texts)
    
    return ""


def compute_embeddings_batch(
    texts: List[str],
    model,
    tokenizer,
    device: int,
    max_length: int,
    input_type: str = 'document'
) -> np.ndarray:
    """
    Compute embeddings for a batch of texts.
    
    Args:
        texts: List of text strings
        model: The embedding model
        tokenizer: The tokenizer
        device: GPU device ID
        max_length: Maximum text length
        input_type: Input type ('document' or 'query')
    
    Returns:
        Numpy array of embeddings
    """
    # Add instruction prefix if using Nemotron model
    if 'nemotron' in tokenizer.name_or_path.lower():
        if input_type == 'query':
            texts = [f"query: {text}" for text in texts]
        else:
            texts = [f"passage: {text}" for text in texts]
    
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(f'cuda:{device}')
    
    # Compute embeddings
    with torch.no_grad():
        outputs = model(**encoded)
        # Use mean pooling on the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()


def process_shard(
    shard_info: Dict[str, any],
    model,
    tokenizer,
    device: int,
    batch_size: int,
    max_length: int,
    input_type: str,
    output_dir: Path,
    progress_dict: Optional[dict] = None
) -> Dict[str, any]:
    """
    Process a single dataset shard and save embeddings as parquet.
    
    Args:
        shard_info: Dictionary with shard information
        model: The embedding model
        tokenizer: The tokenizer
        device: GPU device ID
        batch_size: Batch size for inference
        max_length: Maximum text length
        input_type: Input type ('document' or 'query')
        output_dir: Output directory for embeddings
        progress_dict: Shared dictionary for progress tracking
    
    Returns:
        Dictionary with processing statistics
    """
    from datasets import load_dataset, Dataset
    import pyarrow.parquet as pq
    
    console = Console()
    
    dataset_name = shard_info['dataset_name']
    split_name = shard_info['split_name']
    shard_idx = shard_info['shard_idx']
    total_shards = shard_info['total_shards']
    hf_name = shard_info['hf_name']
    hf_config = shard_info['hf_config']
    cache_dir = shard_info['cache_dir']
    
    # Create output path
    output_subdir = output_dir / dataset_name / split_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename matching HuggingFace format
    # Format: {dataset}-{split}-{shard:05d}-of-{total:05d}.parquet
    num_digits = 5
    parquet_filename = (
        f"{dataset_name}-{split_name}-"
        f"{str(shard_idx).zfill(num_digits)}-of-"
        f"{str(total_shards).zfill(num_digits)}.parquet"
    )
    parquet_path = output_subdir / parquet_filename
    
    # Skip if already exists
    if parquet_path.exists():
        console.print(f"   [dim]⏭️  GPU {device}: Skipping existing {parquet_filename}[/dim]")
        try:
            existing_table = pq.read_table(str(parquet_path))
            return {
                'status': 'skipped',
                'samples': len(existing_table),
                'shard': shard_info['spec']
            }
        except:
            # If can't read, reprocess
            pass
    
    start_time = time.time()
    
    try:
        # Load the dataset
        load_args = {
            'path': hf_name,
            'cache_dir': cache_dir,
            'split': split_name
        }
        
        if hf_config:
            load_args['name'] = hf_config
        
        # Load specific shard
        dataset = load_dataset(**load_args)
        shard_data = dataset.shard(num_shards=total_shards, index=shard_idx)
        
        num_samples = len(shard_data)
        
        console.print(
            f"[cyan]🔄 GPU {device}:[/cyan] Processing {parquet_filename} "
            f"({num_samples:,} samples)"
        )
        
        # Initialize progress tracking for this GPU
        if progress_dict is not None:
            progress_dict[device] = {
                'current': 0,
                'total': num_samples,
                'filename': parquet_filename,
                'stage': 'loading'
            }
        
        # Extract texts and compute embeddings
        texts = []
        indices = []
        
        for idx, sample in enumerate(shard_data):
            text = extract_text_from_sample(sample)
            if text and len(text.strip()) > 0:
                texts.append(text)
                indices.append(idx)
            
            # Update progress every 500 samples
            if progress_dict is not None and (idx % 500 == 0 or idx == num_samples - 1):
                progress_dict[device] = {
                    'current': idx + 1,
                    'total': num_samples,
                    'filename': parquet_filename,
                    'stage': 'extracting'
                }
        
        if not texts:
            console.print(f"[yellow]⚠️  GPU {device}:[/yellow] No valid text in {parquet_filename}")
            if progress_dict is not None and device in progress_dict:
                del progress_dict[device]
            return {'status': 'no_text', 'samples': 0, 'shard': shard_info['spec']}
        
        # Process in batches
        all_embeddings = []
        total_texts = len(texts)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = compute_embeddings_batch(
                batch_texts, model, tokenizer, device, max_length, input_type
            )
            all_embeddings.append(batch_embeddings)
            
            # Update progress - show embedding generation progress
            if progress_dict is not None:
                texts_processed = min(i + batch_size, total_texts)
                progress_dict[device] = {
                    'current': texts_processed,
                    'total': total_texts,
                    'filename': parquet_filename,
                    'stage': 'embedding'
                }
        
        # Concatenate and save
        embeddings_array = np.vstack(all_embeddings)
        
        # Update progress - saving
        if progress_dict is not None:
            progress_dict[device] = {
                'current': total_texts,
                'total': total_texts,
                'filename': parquet_filename,
                'stage': 'saving'
            }
        
        # Create dataset with embeddings
        embedding_dataset = Dataset.from_dict({
            'embeddings': embeddings_array.tolist(),
            'original_index': indices
        })
        
        # Save as parquet
        embedding_dataset.to_parquet(str(parquet_path))
        
        # Clear progress for this GPU
        if progress_dict is not None and device in progress_dict:
            del progress_dict[device]
        
        elapsed = time.time() - start_time
        samples_per_sec = len(texts) / elapsed if elapsed > 0 else 0
        
        console.print(
            f"[green]✅ GPU {device}:[/green] Saved {parquet_filename} - "
            f"[yellow]{len(texts):,}[/yellow] samples in "
            f"[cyan]{elapsed:.1f}s[/cyan] ([magenta]{samples_per_sec:.1f}[/magenta] samples/s)"
        )
        
        return {
            'status': 'success',
            'samples': len(texts),
            'time': elapsed,
            'shard': shard_info['spec']
        }
        
    except Exception as e:
        console.print(f"[red]❌ GPU {device}:[/red] Error processing {parquet_filename}: {e}")
        import traceback
        traceback.print_exc()
        
        # Clear progress for this GPU
        if progress_dict is not None and device in progress_dict:
            del progress_dict[device]
        
        return {
            'status': 'error',
            'samples': 0,
            'error': str(e),
            'shard': shard_info['spec']
        }


def gpu_worker(
    gpu_id: int,
    work_queue: Queue,
    results_queue: Queue,
    model_name: str,
    batch_size: int,
    max_length: int,
    input_type: str,
    output_dir: Path,
    progress_dict: dict,
    shutdown_event: Event
):
    """
    GPU worker process that processes shards from the work queue.
    
    Args:
        gpu_id: GPU device ID
        work_queue: Queue containing work items (shard info dicts)
        results_queue: Queue for returning results
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        max_length: Maximum text length
        input_type: Input type ('document' or 'query')
        output_dir: Output directory
        progress_dict: Shared dictionary for progress tracking
        shutdown_event: Event to signal shutdown
    """
    try:
        # Load model on this GPU
        model, tokenizer = load_model(model_name, gpu_id)
        
        console = Console()
        
        # Process shards from the queue
        while not shutdown_event.is_set():
            try:
                # Get work item with timeout
                shard_info = work_queue.get(timeout=1)
                
                if shard_info is None:  # Poison pill
                    console.print(f"[dim]🛑 GPU {gpu_id}: Received shutdown signal[/dim]")
                    break
                
                # Process the shard
                result = process_shard(
                    shard_info, model, tokenizer, gpu_id,
                    batch_size, max_length, input_type, output_dir, progress_dict
                )
                
                # Put result in results queue
                results_queue.put({
                    'gpu_id': gpu_id,
                    'shard_info': shard_info,
                    'result': result
                })
                
            except Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as e:
                console.print(f"[red]❌ GPU {gpu_id}:[/red] Worker error: {e}")
                import traceback
                traceback.print_exc()
        
        console.print(f"[green]✅ GPU {gpu_id}: Worker finished[/green]")
        
    except Exception as e:
        console = Console()
        console.print(f"[red]❌ GPU {gpu_id}:[/red] Fatal worker error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to orchestrate multi-GPU parallel extraction."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel embedding extraction (shard-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings from v1 chat split using 4 GPUs
  python extract_embeddings_parallel_shards.py --splits v1:chat --num-gpus 4
  
  # Extract from multiple splits using 8 GPUs
  python extract_embeddings_parallel_shards.py --splits v1:chat v2:math llama-sft:safety --num-gpus 8
  
  # Extract from ALL available dataset splits
  python extract_embeddings_parallel_shards.py --all --num-gpus 8
  
  # Custom batch size and max length
  python extract_embeddings_parallel_shards.py --all --batch-size 64 --max-text-length 8192
        """
    )
    
    parser.add_argument(
        '--splits', nargs='+', required=False,
        help='Format: v1:chat v2:math llama-sft:safety'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Extract embeddings from all available dataset splits'
    )
    parser.add_argument(
        '--num-gpus', type=int, default=8,
        help='Number of GPUs to use (default: 8)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--output', default='embeddings_output',
        help='Output directory (default: embeddings_output)'
    )
    parser.add_argument(
        '--model', default='nvidia/llama-embed-nemotron-8b',
        help='Model to use (default: nvidia/llama-embed-nemotron-8b)'
    )
    parser.add_argument(
        '--max-text-length', type=int, default=8192,
        help='Max text length (default: 8192)'
    )
    parser.add_argument(
        '--input-type', default='document', choices=['document', 'query'],
        help='Input type (default: document)'
    )
    parser.add_argument(
        '--datasets-dir', default='./datasets',
        help='Base datasets directory (default: ./datasets)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.splits:
        console = Console()
        console.print("[red]❌ Error:[/red] Either --splits or --all must be specified")
        console.print("\nExamples:")
        console.print("  python extract_embeddings_parallel_shards.py --splits v1:chat v2:math")
        console.print("  python extract_embeddings_parallel_shards.py --all")
        sys.exit(1)
    
    if args.all and args.splits:
        console = Console()
        console.print("[yellow]⚠️  Warning:[/yellow] Both --all and --splits specified. --all takes precedence.")
    
    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.is_absolute():
        datasets_dir = script_dir / datasets_dir
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console = Console()
    
    # Print header
    console.print(Panel.fit(
        "[bold cyan]Multi-GPU Parallel Embedding Extraction[/bold cyan]\n"
        "[dim]Shard-Based Work Distribution for Maximum GPU Utilization[/dim]",
        border_style="cyan"
    ))
    
    # Determine which splits to process
    if args.all:
        console.print("\n[bold yellow]📦 Mode: Extract ALL available dataset splits[/bold yellow]")
        splits_to_process = discover_all_splits(datasets_dir)
        
        if not splits_to_process:
            console.print("[red]❌ No datasets found![/red]")
            console.print("\n[dim]Please download datasets first:[/dim]")
            console.print("  python download_nemotron_datasets.py --all")
            sys.exit(1)
        
        # Sort splits for organized processing
        splits_to_process = sorted(splits_to_process)
        console.print(f"[green]✅ Found {len(splits_to_process)} split(s) to process[/green]")
    else:
        splits_to_process = args.splits
    
    # Configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="yellow")
    
    if args.all:
        config_table.add_row("Mode", "Extract ALL splits (shard-based)")
        config_table.add_row("Splits found", str(len(splits_to_process)))
    else:
        config_table.add_row("Mode", "Extract specific splits (shard-based)")
        config_table.add_row("Splits", ', '.join(args.splits))
    
    config_table.add_row("GPUs", str(args.num_gpus))
    config_table.add_row("Batch size", str(args.batch_size))
    config_table.add_row("Max text length", str(args.max_text_length))
    config_table.add_row("Model", args.model)
    config_table.add_row("Input type", args.input_type)
    config_table.add_row("Output directory", str(output_dir))
    config_table.add_row("Datasets directory", str(datasets_dir))
    
    console.print("\n[bold]📊 Configuration:[/bold]")
    console.print(config_table)
    console.print()
    
    # Discover dataset shards
    console.print("[bold cyan]🔍 Discovering dataset shards...[/bold cyan]")
    dataset_shards = discover_dataset_shards(datasets_dir, splits_to_process)
    
    if not dataset_shards:
        console.print("[red]❌ No valid dataset shards found![/red]")
        sys.exit(1)
    
    console.print(f"[green]✅ Found {len(dataset_shards)} shard(s) to process[/green]")
    console.print()
    
    # Create work queue, results queue, and progress tracking
    manager = Manager()
    work_queue = manager.Queue()
    results_queue = manager.Queue()
    progress_dict = manager.dict()  # Shared dict for per-GPU progress
    shutdown_event = manager.Event()
    
    # Populate work queue
    for shard_info in dataset_shards:
        work_queue.put(shard_info)
    
    # Add poison pills (None) for each worker to signal completion
    for _ in range(args.num_gpus):
        work_queue.put(None)
    
    console.print(f"[cyan]📋 Work queue populated with {len(dataset_shards)} shard(s)[/cyan]")
    console.print()
    
    # Start GPU worker processes
    console.print(f"[bold cyan]🚀 Starting {args.num_gpus} GPU worker(s)...[/bold cyan]")
    workers = []
    for gpu_id in range(args.num_gpus):
        worker = Process(
            target=gpu_worker,
            args=(
                gpu_id, work_queue, results_queue, args.model,
                args.batch_size, args.max_text_length, args.input_type,
                output_dir, progress_dict, shutdown_event
            )
        )
        worker.start()
        workers.append(worker)
    
    console.print(f"[green]✅ All workers started[/green]")
    console.print()
    
    # Monitor progress
    console.print("[bold]📊 Processing progress:[/bold]")
    
    # Display per-GPU progress in a table
    from threading import Thread
    import threading
    
    progress_display_active = threading.Event()
    progress_display_active.set()
    
    def display_gpu_progress():
        """Background thread to display per-GPU progress."""
        from rich.live import Live
        from rich.table import Table as RichTable
        
        with Live(console=console, refresh_per_second=2) as live:
            while progress_display_active.is_set():
                table = RichTable(title="GPU Progress", show_header=True, header_style="bold cyan")
                table.add_column("GPU", style="cyan", width=6)
                table.add_column("Stage", style="magenta", width=20)
                table.add_column("File", style="yellow", width=45)
                table.add_column("Progress", style="green", width=30)
                
                if progress_dict:
                    for gpu_id in sorted(progress_dict.keys()):
                        info = progress_dict[gpu_id]
                        filename = info.get('filename', '?')
                        current = info.get('current', 0)
                        total = info.get('total', 1)
                        stage = info.get('stage', 'processing')
                        pct = (current / total * 100) if total > 0 else 0
                        
                        # Truncate filename if too long
                        if len(filename) > 40:
                            filename = "..." + filename[-37:]
                        
                        # Format stage
                        stage_emoji = {
                            'loading': '📂',
                            'extracting': '📝',
                            'embedding': '🔮',
                            'saving': '💾'
                        }
                        stage_display = stage_emoji.get(stage, '⚙️')
                        
                        table.add_row(
                            f"GPU {gpu_id}",
                            f"{stage_display} {stage}",
                            filename,
                            f"{current:,}/{total:,} ({pct:.1f}%)"
                        )
                else:
                    table.add_row("—", "—", "No active processing", "—")
                
                live.update(table)
                time.sleep(0.5)
    
    # Start progress display thread
    progress_thread = Thread(target=display_gpu_progress, daemon=True)
    progress_thread.start()
    
    console.print()
    console.print("=" * 80)
    
    completed = 0
    total_samples_processed = 0
    start_time = time.time()
    
    try:
        while completed < len(dataset_shards):
            try:
                result_info = results_queue.get(timeout=1)
                completed += 1
                
                result = result_info['result']
                gpu_id = result_info['gpu_id']
                shard_spec = result.get('shard', 'unknown')
                
                if result['status'] == 'success':
                    total_samples_processed += result['samples']
                    elapsed = time.time() - start_time
                    avg_speed = total_samples_processed / elapsed if elapsed > 0 else 0
                    console.print(
                        f"[green]✅ [{completed}/{len(dataset_shards)}][/green] "
                        f"[cyan]GPU {gpu_id}:[/cyan] {shard_spec} - "
                        f"[yellow]{result['samples']:,}[/yellow] samples | "
                        f"Avg: [magenta]{avg_speed:.1f}[/magenta] samples/s"
                    )
                elif result['status'] == 'skipped':
                    total_samples_processed += result['samples']
                    console.print(
                        f"[dim]⏭️  [{completed}/{len(dataset_shards)}][/dim] "
                        f"[cyan]GPU {gpu_id}:[/cyan] {shard_spec} - "
                        f"[dim]skipped[/dim]"
                    )
                else:
                    console.print(
                        f"[yellow]⚠️  [{completed}/{len(dataset_shards)}][/yellow] "
                        f"[cyan]GPU {gpu_id}:[/cyan] {shard_spec} - "
                        f"[red]{result['status']}[/red]"
                    )
                
            except Empty:
                continue
    
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user, shutting down workers...[/yellow]")
        shutdown_event.set()
    
    # Stop progress display
    progress_display_active.clear()
    if progress_thread.is_alive():
        progress_thread.join(timeout=2)
    
    # Wait for all workers to finish
    console.print("\n[cyan]🛑 Waiting for workers to finish...[/cyan]")
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            console.print(f"[yellow]⚠️  Worker {worker.pid} did not terminate gracefully, forcing...[/yellow]")
            worker.terminate()
    
    # Final summary
    elapsed = time.time() - start_time
    console.print()
    
    # Create summary table
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="cyan bold")
    summary_table.add_column("Value", style="green bold")
    
    summary_table.add_row("Processed", f"{completed} shard(s)")
    summary_table.add_row("Total samples", f"{total_samples_processed:,}")
    summary_table.add_row("Total time", f"{elapsed:.1f}s ({elapsed/60:.1f} min)")
    if elapsed > 0:
        summary_table.add_row("Average speed", f"{total_samples_processed/elapsed:.1f} samples/s")
    summary_table.add_row("Output directory", str(output_dir))
    
    console.print(Panel.fit(
        summary_table,
        title="[bold green]📊 PROCESSING COMPLETE[/bold green]",
        border_style="green"
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

