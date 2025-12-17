#!/usr/bin/env python3
"""
Embedding Validation Script

This script validates the generated embeddings to ensure they are correct:
- Checks embedding dimensions
- Verifies number of samples
- Lists all splits and shards
- Compares with source datasets (optional)

Usage:
    python validate_embeddings.py
    python validate_embeddings.py --embeddings-dir /path/to/embeddings
    python validate_embeddings.py --compare-source  # Compare with source datasets
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# =============================================================================
# DEFAULT PATH CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()

DEFAULT_DATASETS_DIR = SCRIPT_DIR / "datasets"
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings_output"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate generated embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_embeddings.py
  
  # Custom embeddings directory
  python validate_embeddings.py --embeddings-dir /data/embeddings
  
  # Compare with source datasets
  python validate_embeddings.py --compare-source
  
  # Verbose output with sample data
  python validate_embeddings.py --verbose
  
  # Custom directories (consistent with download/extract scripts)
  python validate_embeddings.py --datasets-dir /data/datasets --embeddings-dir /data/embeddings
        """
    )
    
    # Path arguments (consistent with download_nemotron_datasets.py and extract_embeddings_parallel_shards.py)
    parser.add_argument(
        '--datasets-dir', type=str, default=str(DEFAULT_DATASETS_DIR),
        help=f'Directory containing source datasets (default: {DEFAULT_DATASETS_DIR})'
    )
    parser.add_argument(
        '--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help=f'Directory for model checkpoints/HF cache (default: {DEFAULT_CHECKPOINTS_DIR})'
    )
    parser.add_argument(
        '--embeddings-dir', type=str, default=str(DEFAULT_EMBEDDINGS_DIR),
        help=f'Directory containing embeddings (default: {DEFAULT_EMBEDDINGS_DIR})'
    )
    
    # Validation options
    parser.add_argument(
        '--compare-source', action='store_true',
        help='Compare embeddings count with source dataset counts'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed information including sample embeddings'
    )
    parser.add_argument(
        '--check-nans', action='store_true',
        help='Check for NaN or Inf values in embeddings (slower)'
    )
    
    return parser.parse_args()


def discover_embedding_files(embeddings_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Discover all embedding parquet files organized by dataset and split.
    
    Returns:
        Dict[dataset_name][split_name] = [list of parquet files]
    """
    files_by_dataset = defaultdict(lambda: defaultdict(list))
    
    if not embeddings_dir.exists():
        return files_by_dataset
    
    # Walk through the embeddings directory
    for dataset_dir in embeddings_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        
        for split_dir in dataset_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            split_name = split_dir.name
            
            # Find all parquet files
            parquet_files = sorted(split_dir.glob("*.parquet"))
            if parquet_files:
                files_by_dataset[dataset_name][split_name] = parquet_files
    
    return files_by_dataset


def validate_parquet_file(filepath: Path, check_nans: bool = False) -> Dict:
    """
    Validate a single parquet file.
    
    Returns:
        Dictionary with validation results
    """
    import pyarrow.parquet as pq
    import numpy as np
    
    result = {
        'path': filepath,
        'valid': True,
        'errors': [],
        'warnings': [],
        'num_samples': 0,
        'embedding_dim': 0,
        'file_size_mb': 0,
    }
    
    try:
        # Get file size
        result['file_size_mb'] = filepath.stat().st_size / (1024 * 1024)
        
        # Read parquet file
        table = pq.read_table(str(filepath))
        df = table.to_pandas()
        
        result['num_samples'] = len(df)
        result['columns'] = list(df.columns)
        
        # Check for embeddings column
        if 'embeddings' not in df.columns:
            result['valid'] = False
            result['errors'].append("Missing 'embeddings' column")
            return result
        
        # Get embedding dimension
        if len(df) > 0:
            first_embedding = df['embeddings'].iloc[0]
            if isinstance(first_embedding, (list, np.ndarray)):
                result['embedding_dim'] = len(first_embedding)
            else:
                result['valid'] = False
                result['errors'].append(f"Invalid embedding type: {type(first_embedding)}")
                return result
        
        # Check for NaN/Inf values if requested
        if check_nans and len(df) > 0:
            embeddings = np.array(df['embeddings'].tolist())
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            
            if nan_count > 0:
                result['warnings'].append(f"Found {nan_count} NaN values")
            if inf_count > 0:
                result['warnings'].append(f"Found {inf_count} Inf values")
        
        # Check for original_index column
        if 'original_index' not in df.columns:
            result['warnings'].append("Missing 'original_index' column")
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error reading file: {str(e)}")
    
    return result


def get_source_dataset_counts(datasets_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Get sample counts from source datasets for comparison.
    
    Returns:
        Dict[dataset_name][split_name] = sample_count
    """
    # Set up HuggingFace cache
    os.environ['HF_DATASETS_CACHE'] = str(datasets_dir)
    
    from datasets import load_dataset
    
    counts = defaultdict(dict)
    
    # Dataset configurations (must match extract_embeddings_parallel_shards.py and download_nemotron_datasets.py)
    dataset_configs = {
        # V1 and V2 datasets
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
        # Llama-Nemotron datasets
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
        # V3 datasets (Post-Training Nano v3 Collection) - Available
        'v3-science': {
            'hf_name': 'nvidia/Nemotron-Science-v1',
            'cache_dir': str(datasets_dir / 'nemotron-v3' / 'science'),
            'config': None
        },
        'v3-instruction-chat': {
            'hf_name': 'nvidia/Nemotron-Instruction-Following-Chat-v1',
            'cache_dir': str(datasets_dir / 'nemotron-v3' / 'instruction-chat'),
            'config': None
        },
        'v3-math-proofs': {
            'hf_name': 'nvidia/Nemotron-Math-Proofs-v1',
            'cache_dir': str(datasets_dir / 'nemotron-v3' / 'math-proofs'),
            'config': None
        },
        # V3 datasets (Post-Training Nano v3 Collection) - Preview (not yet downloadable)
        'v3-rl-blend': {
            'hf_name': 'nvidia/Nemotron-3-Nano-RL-Training-Blend',
            'cache_dir': str(datasets_dir / 'nemotron-v3' / 'rl-blend'),
            'config': None
        },
        'v3-agentic': {
            'hf_name': 'nvidia/Nemotron-Agentic-v1',
            'cache_dir': str(datasets_dir / 'nemotron-v3' / 'agentic'),
            'config': None
        },
        'v3-competitive-programming': {
            'hf_name': 'nvidia/Nemotron-Competitive-Programming-v1',
            'cache_dir': str(datasets_dir / 'nemotron-v3' / 'competitive-programming'),
            'config': None
        },
        'v3-math': {
            'hf_name': 'nvidia/Nemotron-Math-v2',
            'cache_dir': str(datasets_dir / 'nemotron-v3' / 'math-v2'),
            'config': None
        },
    }
    
    for dataset_name, config in dataset_configs.items():
        cache_dir = Path(config['cache_dir'])
        if not cache_dir.exists():
            continue
        
        try:
            load_args = {
                'path': config['hf_name'],
                'cache_dir': config['cache_dir']
            }
            if config['config']:
                load_args['name'] = config['config']
            
            dataset = load_dataset(**load_args)
            
            for split_name in dataset.keys():
                counts[dataset_name][split_name] = len(dataset[split_name])
                
        except Exception as e:
            print(f"  ⚠️  Could not load {dataset_name}: {e}")
    
    return counts


def format_size(size_mb: float) -> str:
    """Format file size for display."""
    if size_mb >= 1024:
        return f"{size_mb/1024:.2f} GB"
    return f"{size_mb:.2f} MB"


def print_validation_report(
    files_by_dataset: Dict,
    validation_results: Dict,
    source_counts: Optional[Dict] = None,
    verbose: bool = False
):
    """Print a formatted validation report."""
    
    print("\n" + "=" * 90)
    print("📊 EMBEDDING VALIDATION REPORT")
    print("=" * 90)
    
    total_samples = 0
    total_files = 0
    total_size_mb = 0
    all_valid = True
    embedding_dims = set()
    
    for dataset_name in sorted(files_by_dataset.keys()):
        splits = files_by_dataset[dataset_name]
        
        print(f"\n{'─' * 90}")
        print(f"📁 Dataset: {dataset_name}")
        print(f"{'─' * 90}")
        
        dataset_samples = 0
        dataset_size = 0
        
        for split_name in sorted(splits.keys()):
            files = splits[split_name]
            
            split_samples = 0
            split_size = 0
            split_dim = None
            split_errors = []
            split_warnings = []
            
            for filepath in files:
                result = validation_results.get(str(filepath), {})
                split_samples += result.get('num_samples', 0)
                split_size += result.get('file_size_mb', 0)
                
                if result.get('embedding_dim'):
                    split_dim = result['embedding_dim']
                    embedding_dims.add(split_dim)
                
                if not result.get('valid', True):
                    all_valid = False
                    split_errors.extend(result.get('errors', []))
                
                split_warnings.extend(result.get('warnings', []))
            
            # Status indicator
            status = "✅" if not split_errors else "❌"
            
            # Source comparison
            source_info = ""
            if source_counts and dataset_name in source_counts:
                if split_name in source_counts[dataset_name]:
                    source_count = source_counts[dataset_name][split_name]
                    diff = split_samples - source_count
                    pct = (split_samples / source_count * 100) if source_count > 0 else 0
                    
                    if abs(diff) <= 1:
                        source_info = f" (source: {source_count:,} ✓)"
                    else:
                        source_info = f" (source: {source_count:,}, diff: {diff:+,}, {pct:.1f}%)"
            
            print(f"  {status} {split_name}:")
            print(f"      Samples: {split_samples:,}{source_info}")
            print(f"      Shards:  {len(files)}")
            print(f"      Size:    {format_size(split_size)}")
            if split_dim:
                print(f"      Dim:     {split_dim}")
            
            if split_errors:
                for err in split_errors:
                    print(f"      ❌ Error: {err}")
            
            if split_warnings and verbose:
                for warn in split_warnings:
                    print(f"      ⚠️  Warning: {warn}")
            
            dataset_samples += split_samples
            dataset_size += split_size
            total_files += len(files)
        
        print(f"  {'─' * 40}")
        print(f"  📈 Dataset total: {dataset_samples:,} samples, {format_size(dataset_size)}")
        
        total_samples += dataset_samples
        total_size_mb += dataset_size
    
    # Summary
    print("\n" + "=" * 90)
    print("📋 SUMMARY")
    print("=" * 90)
    
    print(f"\n  Total datasets:    {len(files_by_dataset)}")
    print(f"  Total splits:      {sum(len(s) for s in files_by_dataset.values())}")
    print(f"  Total files:       {total_files}")
    print(f"  Total samples:     {total_samples:,}")
    print(f"  Total size:        {format_size(total_size_mb)}")
    print(f"  Embedding dims:    {sorted(embedding_dims) if embedding_dims else 'N/A'}")
    
    if all_valid:
        print(f"\n  ✅ All embeddings validated successfully!")
    else:
        print(f"\n  ❌ Some embeddings have errors - check details above")
    
    print("\n" + "=" * 90)


def show_sample_embeddings(validation_results: Dict, num_samples: int = 3):
    """Show sample embeddings from the first valid file."""
    import numpy as np
    import pyarrow.parquet as pq
    
    print("\n" + "=" * 90)
    print("🔍 SAMPLE EMBEDDINGS")
    print("=" * 90)
    
    for filepath_str, result in validation_results.items():
        if result.get('valid') and result.get('num_samples', 0) > 0:
            filepath = Path(filepath_str)
            
            print(f"\nFile: {filepath.name}")
            print(f"From: {filepath.parent.parent.name}/{filepath.parent.name}")
            
            try:
                table = pq.read_table(str(filepath))
                df = table.to_pandas()
                
                for i in range(min(num_samples, len(df))):
                    embedding = np.array(df['embeddings'].iloc[i])
                    print(f"\n  Sample {i+1}:")
                    print(f"    Shape: {embedding.shape}")
                    print(f"    Dtype: {embedding.dtype}")
                    print(f"    Min:   {embedding.min():.6f}")
                    print(f"    Max:   {embedding.max():.6f}")
                    print(f"    Mean:  {embedding.mean():.6f}")
                    print(f"    Std:   {embedding.std():.6f}")
                    print(f"    Norm:  {np.linalg.norm(embedding):.6f}")
                    print(f"    First 5: [{', '.join(f'{x:.4f}' for x in embedding[:5])}]")
                
            except Exception as e:
                print(f"  Error reading samples: {e}")
            
            break  # Only show from first valid file
    
    print("\n" + "=" * 90)


def main():
    """Main function."""
    args = parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    datasets_dir = Path(args.datasets_dir)
    
    print("=" * 90)
    print("🔍 Embedding Validation Script")
    print("=" * 90)
    print(f"\n  Embeddings directory: {embeddings_dir}")
    print(f"  Datasets directory:   {datasets_dir}")
    print(f"  Compare with source:  {args.compare_source}")
    print(f"  Check NaN/Inf:        {args.check_nans}")
    print(f"  Verbose:              {args.verbose}")
    
    # Check if embeddings directory exists
    if not embeddings_dir.exists():
        print(f"\n❌ Embeddings directory not found: {embeddings_dir}")
        print("   Run extract_embeddings_parallel_shards.py first to generate embeddings.")
        sys.exit(1)
    
    # Discover embedding files
    print("\n📂 Discovering embedding files...")
    files_by_dataset = discover_embedding_files(embeddings_dir)
    
    if not files_by_dataset:
        print(f"\n❌ No embedding files found in: {embeddings_dir}")
        sys.exit(1)
    
    total_files = sum(len(f) for splits in files_by_dataset.values() for f in splits.values())
    print(f"   Found {total_files} parquet file(s) across {len(files_by_dataset)} dataset(s)")
    
    # Validate each file
    print("\n🔬 Validating embedding files...")
    validation_results = {}
    
    for dataset_name, splits in files_by_dataset.items():
        for split_name, files in splits.items():
            for filepath in files:
                print(f"   Checking: {dataset_name}/{split_name}/{filepath.name}", end="\r")
                result = validate_parquet_file(filepath, check_nans=args.check_nans)
                validation_results[str(filepath)] = result
    
    print(" " * 80)  # Clear the line
    
    # Get source dataset counts if requested
    source_counts = None
    if args.compare_source:
        print("\n📊 Loading source dataset counts for comparison...")
        source_counts = get_source_dataset_counts(datasets_dir)
    
    # Print report
    print_validation_report(
        files_by_dataset,
        validation_results,
        source_counts,
        verbose=args.verbose
    )
    
    # Show sample embeddings if verbose
    if args.verbose:
        show_sample_embeddings(validation_results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

