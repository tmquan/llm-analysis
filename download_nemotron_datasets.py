#!/usr/bin/env python3
"""
NVIDIA Nemotron Post-Training Datasets Downloader

This script downloads NVIDIA's Nemotron post-training datasets, which are used
for fine-tuning large language models.

Features:
- Auto-detects problematic datasets and uses JSONL streaming mode
- Automatically converts JSONL to Parquet format
- Handles authentication for gated datasets

Available Datasets:
1. Nemotron-Post-Training-Dataset-v1: Original version with chat, code, math, stem, and tool_calling splits
2. Nemotron-Post-Training-Dataset-v2: Updated version with improvements
3. Llama-Nemotron-Post-Training-Dataset: Comprehensive dataset with SFT and RL subsets
4. Nemotron v3 Collection (Post-Training Nano v3):
   - nvidia/Nemotron-3-Nano-RL-Training-Blend  (⚠️ auto-JSONL)
   - nvidia/Nemotron-Science-v1
   - nvidia/Nemotron-Instruction-Following-Chat-v1
   - nvidia/Nemotron-Math-Proofs-v1
   - nvidia/Nemotron-Agentic-v1  (⚠️ auto-JSONL)
   - nvidia/Nemotron-Competitive-Programming-v1  (⚠️ auto-JSONL)
   - nvidia/Nemotron-Math-v2  (⚠️ auto-JSONL)

Usage:
    python download_nemotron_datasets.py [--v1] [--v2] [--llama-sft] [--llama-rl] [--v3] [--all]
    
    Custom paths:
    python download_nemotron_datasets.py --datasets-dir /path/to/datasets
    
    JSONL mode (manual, keeps JSONL without converting):
    python download_nemotron_datasets.py --jsonl --llama-rl
    
    Disable auto-conversion:
    python download_nemotron_datasets.py --no-convert --v3
    
    If no flags are provided, all datasets will be downloaded.
    Problematic datasets are automatically handled with JSONL→Parquet conversion.
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# DEFAULT PATH CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()

# Default paths (can be overridden via CLI arguments)
DEFAULT_DATASETS_DIR = SCRIPT_DIR / "datasets"
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

# Global variables that will be set after parsing arguments
DATASETS_DIR = None
CHECKPOINTS_DIR = None


def parse_path_args():
    """
    Parse only path-related arguments first.
    This is needed because HuggingFace environment variables must be set
    BEFORE importing the datasets library.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--datasets-dir', type=str, default=str(DEFAULT_DATASETS_DIR),
        help=f'Directory for downloaded datasets (default: {DEFAULT_DATASETS_DIR})'
    )
    parser.add_argument(
        '--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help=f'Directory for model checkpoints/HF cache (default: {DEFAULT_CHECKPOINTS_DIR})'
    )
    
    # Parse known args only (ignore others for now)
    args, _ = parser.parse_known_args()
    return args


def setup_environment(datasets_dir: Path, checkpoints_dir: Path):
    """
    Set up HuggingFace environment variables and create directories.
    Must be called BEFORE importing datasets library.
    """
    global DATASETS_DIR, CHECKPOINTS_DIR
    
    DATASETS_DIR = datasets_dir
    CHECKPOINTS_DIR = checkpoints_dir
    
    # Ensure directories exist
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set HuggingFace cache directories
    # Models/hub go to checkpoints (in case any models are downloaded)
    os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)
    os.environ['HF_MODULES_CACHE'] = str(CHECKPOINTS_DIR / "modules")
    
    # Datasets go to datasets folder
    os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)


def display_environment():
    """Display environment information."""
    print(f"📁 Datasets directory: {DATASETS_DIR}")
    print(f"🤖 Checkpoints directory: {CHECKPOINTS_DIR}")
    if USE_JSONL:
        print(f"📄 Mode: JSONL streaming (saves to {DATASETS_DIR}/jsonl/)")
    else:
        print(f"📄 Mode: Auto (uses JSONL+convert for {len(PROBLEMATIC_DATASETS)} known problematic datasets)")
    if AUTO_CONVERT:
        print(f"🔄 Auto-convert: JSONL → Parquet enabled")
    print("✅ Environment setup complete\n")


# =============================================================================
# PARSE PATH ARGUMENTS AND SETUP ENVIRONMENT BEFORE IMPORTS
# =============================================================================
_path_args = parse_path_args()
setup_environment(
    datasets_dir=Path(_path_args.datasets_dir),
    checkpoints_dir=Path(_path_args.checkpoints_dir)
)

# Now safe to import datasets library
from datasets import load_dataset

# Get HF token from environment variable (set in .bashrc or shell)
HF_TOKEN = os.environ.get('HF_TOKEN', True)  # Fall back to True to use cached token

# Global flags (set by CLI)
VERBOSE = False
USE_JSONL = False  # Use streaming mode and save as JSONL
AUTO_CONVERT = True  # Auto-convert JSONL to parquet for problematic datasets

# Known problematic datasets that need JSONL mode
PROBLEMATIC_DATASETS = {
    "nvidia/Llama-Nemotron-Post-Training-Dataset:RL",  # config-specific
    "nvidia/Nemotron-3-Nano-RL-Training-Blend",
    "nvidia/Nemotron-Agentic-v1",
    "nvidia/Nemotron-Competitive-Programming-v1",
    "nvidia/Nemotron-Math-v2",
}


def is_problematic_dataset(load_fn: str, config: str = None) -> bool:
    """Check if a dataset is known to have parquet issues."""
    if config:
        key = f"{load_fn}:{config}"
        if key in PROBLEMATIC_DATASETS:
            return True
    return load_fn in PROBLEMATIC_DATASETS


def download_raw_parquet_lenient(repo: str, output_dir: str, dataset_name: str, config: str = None):
    """
    Download raw parquet files and read them with multiple fallback methods:
    1. Fastparquet (more lenient, pure Python)
    2. Pandas with error handling
    3. PyArrow row-group by row-group
    
    This is a fallback for severely corrupted datasets.
    """
    import requests
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔧 Attempting raw parquet download with lenient reading...")
    
    try:
        # Use HuggingFace API to list files
        api_url = f"https://huggingface.co/api/datasets/{repo}/tree/main"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN and HF_TOKEN != True else {}
        
        response = requests.get(api_url, headers=headers)
        if response.status_code != 200:
            print(f"   ❌ Failed to list repo files: {response.status_code}")
            return 0
        
        files_info = response.json()
        
        # Filter for parquet files
        if config:
            parquet_files = [f['path'] for f in files_info if f['path'].endswith('.parquet') and config.lower() in f['path'].lower()]
        else:
            parquet_files = [f['path'] for f in files_info if f['path'].endswith('.parquet')]
        
        if not parquet_files:
            print(f"   No parquet files found")
            return 0
        
        print(f"   Found {len(parquet_files)} parquet files")
        
        total_rows = 0
        all_records = []
        
        # Create temp directory for downloads
        temp_dir = output_path / "_temp_parquet"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for pq_file in parquet_files:
            print(f"\n   Processing: {pq_file}")
            
            try:
                # Download the file directly
                download_url = f"https://huggingface.co/datasets/{repo}/resolve/main/{pq_file}"
                local_path = temp_dir / Path(pq_file).name
                
                print(f"   Downloading from {download_url[:60]}...")
                response = requests.get(download_url, headers=headers, stream=True)
                if response.status_code != 200:
                    print(f"   ❌ Download failed: {response.status_code}")
                    continue
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"   Downloaded {local_path.stat().st_size / 1024 / 1024:.1f} MB")
                
                file_records = read_parquet_lenient(str(local_path))
                if file_records:
                    all_records.extend(file_records)
                    print(f"   ✅ Read {len(file_records):,} rows from {Path(pq_file).name}")
                    total_rows += len(file_records)
                
                # Clean up temp file
                local_path.unlink()
                
            except Exception as e:
                print(f"   ❌ Error with file {pq_file}: {str(e)[:100]}")
                if VERBOSE:
                    traceback.print_exc()
                continue
        
        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except:
            pass
        
        # Save all records to JSONL
        if all_records:
            output_file = output_path / "train.jsonl"
            print(f"\n   Writing {len(all_records):,} records to {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in tqdm(all_records, desc="   Writing", unit=" rows"):
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"   ✅ Saved {len(all_records):,} total rows")
            return len(all_records)
        
        return 0
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:200]}")
        if VERBOSE:
            traceback.print_exc()
        return 0


def read_parquet_lenient(file_path: str) -> list:
    """
    Try multiple methods to read a parquet file, from most to least strict.
    """
    records = []
    
    # Method 1: Try fastparquet (more lenient than pyarrow)
    try:
        import fastparquet
        print(f"   Trying fastparquet...")
        pf = fastparquet.ParquetFile(file_path)
        df = pf.to_pandas()
        records = df.to_dict('records')
        print(f"   ✅ Fastparquet succeeded: {len(records):,} rows")
        return records
    except ImportError:
        print(f"   ⚠️ fastparquet not installed, trying other methods...")
    except Exception as e:
        print(f"   ⚠️ Fastparquet failed: {str(e)[:80]}")
    
    # Method 2: Try pandas with pyarrow but catch errors
    try:
        import pandas as pd
        print(f"   Trying pandas.read_parquet...")
        df = pd.read_parquet(file_path, engine='pyarrow')
        records = df.to_dict('records')
        print(f"   ✅ Pandas succeeded: {len(records):,} rows")
        return records
    except Exception as e:
        print(f"   ⚠️ Pandas failed: {str(e)[:80]}")
    
    # Method 3: PyArrow row-group by row-group
    try:
        import pyarrow.parquet as pq
        print(f"   Trying PyArrow row-group reading...")
        pq_reader = pq.ParquetFile(file_path)
        num_row_groups = pq_reader.metadata.num_row_groups
        print(f"   File has {num_row_groups} row groups")
        
        for rg_idx in range(num_row_groups):
            try:
                table = pq_reader.read_row_group(rg_idx)
                rg_records = table.to_pylist()
                records.extend(rg_records)
            except Exception as e:
                error_str = str(e)
                if "Check failed" in error_str:
                    print(f"   ⚠️ Row group {rg_idx} corrupted, skipping...")
                else:
                    print(f"   ⚠️ Row group {rg_idx} error: {error_str[:60]}")
                continue
        
        if records:
            print(f"   ✅ PyArrow row-group read: {len(records):,} rows")
        return records
    except Exception as e:
        print(f"   ⚠️ PyArrow row-group failed: {str(e)[:80]}")
    
    # Method 4: Try polars (another parquet reader)
    try:
        import polars as pl
        print(f"   Trying polars...")
        df = pl.read_parquet(file_path)
        records = df.to_dicts()
        print(f"   ✅ Polars succeeded: {len(records):,} rows")
        return records
    except ImportError:
        print(f"   ⚠️ polars not installed")
    except Exception as e:
        print(f"   ⚠️ Polars failed: {str(e)[:80]}")
    
    return records


def convert_jsonl_to_parquet(jsonl_dir: str, parquet_dir: str, dataset_name: str):
    """
    Convert JSONL files to Parquet format.
    
    Args:
        jsonl_dir: Directory containing JSONL files
        parquet_dir: Directory to save Parquet files
        dataset_name: Name of the dataset for display
    
    Returns:
        True if successful, False otherwise
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    jsonl_path = Path(jsonl_dir)
    parquet_path = Path(parquet_dir)
    
    if not jsonl_path.exists():
        print(f"   ⚠️ JSONL directory not found: {jsonl_path}")
        return False
    
    jsonl_files = list(jsonl_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"   ⚠️ No JSONL files found in {jsonl_path}")
        return False
    
    parquet_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔄 Converting {dataset_name} from JSONL to Parquet...")
    
    for jsonl_file in jsonl_files:
        split_name = jsonl_file.stem
        output_file = parquet_path / f"{split_name}.parquet"
        
        print(f"   Converting {jsonl_file.name} -> {output_file.name}")
        
        try:
            # Read JSONL and convert to parquet
            records = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"   Reading {split_name}", unit=" rows"):
                    if line.strip():
                        records.append(json.loads(line))
            
            if not records:
                print(f"   ⚠️ No records in {jsonl_file.name}")
                continue
            
            # Convert to PyArrow table and write as parquet
            table = pa.Table.from_pylist(records)
            pq.write_table(table, output_file, compression='snappy')
            
            print(f"   ✅ Saved {len(records):,} rows to {output_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error converting {jsonl_file.name}: {str(e)[:150]}")
            if VERBOSE:
                traceback.print_exc()
            continue
    
    return True


def safe_download_jsonl(dataset_name: str, load_fn: str, output_dir: str, config: str = None, max_samples: int = None):
    """
    Download dataset using streaming mode and save as JSONL.
    
    This method processes data lazily and may bypass some parquet parsing issues.
    
    Args:
        dataset_name: Human-readable name for display
        load_fn: The HuggingFace dataset identifier
        output_dir: Directory to save JSONL files
        config: Optional dataset configuration/subset name
        max_samples: Maximum samples per split (for testing)
    
    Returns:
        dict with split info if successful, None if failed
    """
    from datasets import load_dataset
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load with streaming mode
        if config:
            ds = load_dataset(load_fn, config, streaming=True, token=HF_TOKEN)
        else:
            ds = load_dataset(load_fn, streaming=True, token=HF_TOKEN)
        
        # Get available splits
        if hasattr(ds, 'keys'):
            splits = list(ds.keys())
        else:
            splits = ['train']
            ds = {'train': ds}
        
        print(f"   Found splits: {splits}")
        
        total_saved = 0
        split_info = {}
        
        for split_name in splits:
            split_data = ds[split_name]
            output_file = output_path / f"{split_name}.jsonl"
            
            print(f"\n   Processing split: {split_name}")
            print(f"   Output: {output_file}")
            
            count = 0
            errors = 0
            
            with open(output_file, 'w', encoding='utf-8') as f:
                try:
                    iterator = iter(split_data)
                    pbar = tqdm(desc=f"   {split_name}", unit=" samples")
                    
                    while True:
                        try:
                            item = next(iterator)
                            # Convert to JSON and write
                            json_line = json.dumps(item, ensure_ascii=False)
                            f.write(json_line + '\n')
                            count += 1
                            pbar.update(1)
                            
                            if max_samples and count >= max_samples:
                                print(f"\n   Reached max_samples limit: {max_samples}")
                                break
                                
                        except StopIteration:
                            break
                        except Exception as e:
                            errors += 1
                            error_str = str(e)
                            # Check for Arrow-level corruption errors
                            if "Check failed" in error_str or "ArrowInvalid" in error_str:
                                if errors <= 3:
                                    print(f"\n   ⚠️ Arrow data corruption at sample ~{count}, attempting to continue...")
                                # Try to skip corrupted batch by continuing
                                continue
                            elif errors <= 5:
                                print(f"\n   ⚠️ Error on sample {count}: {error_str[:100]}")
                            continue
                    
                    pbar.close()
                            
                except Exception as e:
                    error_str = str(e)
                    if "Check failed" in error_str:
                        print(f"\n   ❌ Fatal Arrow corruption error - data file is corrupted")
                        print(f"   📁 Saved {count:,} samples before corruption to {output_file.name}")
                        # Mark as partial
                        partial_marker = output_path / f"{split_name}.partial"
                        partial_marker.write_text(f"Partial download: {count} samples saved before Arrow corruption error")
                    else:
                        print(f"\n   ❌ Stream error: {error_str[:200]}")
            
            if count > 0:
                print(f"   ✅ Saved {count:,} samples to {output_file.name}")
            if errors > 0:
                print(f"   ⚠️ Skipped {errors} samples due to errors")
            total_saved += count
            split_info[split_name] = count
        
        if total_saved > 0:
            # Check if any splits were partial
            partial_files = list(output_path.glob("*.partial"))
            if partial_files:
                print(f"\n⚠️ Download of {dataset_name} PARTIALLY completed (JSONL mode)!")
                print(f"   Total samples saved: {total_saved:,}")
                print(f"   ⚠️ {len(partial_files)} split(s) had corruption - partial data saved")
            else:
                print(f"\n✅ Download of {dataset_name} completed (JSONL mode)!")
                print(f"   Total samples: {total_saved:,}")
            print(f"   Location: {output_dir}\n")
        
        # Return a dict-like object for compatibility with display_summary
        return type('DatasetInfo', (), {
            'keys': lambda self: split_info.keys(),
            '_split_info': split_info,
            '__getitem__': lambda self, k: type('Split', (), {'__len__': lambda s: split_info[k]})()
        })()
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"\n❌ Error downloading {dataset_name} (JSONL mode)")
        print(f"   Type: {error_type}")
        print(f"   Error: {error_msg[:200]}...")
        
        if VERBOSE:
            print(f"\n   Full traceback:")
            traceback.print_exc()
        else:
            print(f"   (Use --verbose for full traceback)")
        
        return None


def safe_download(dataset_name: str, load_fn, cache_dir: str, config: str = None):
    """
    Wrapper function to safely download a dataset with proper error handling.
    
    Args:
        dataset_name: Human-readable name for display
        load_fn: The HuggingFace dataset identifier
        cache_dir: Directory to cache the dataset
        config: Optional dataset configuration/subset name
    
    Returns:
        Dataset if successful, None if failed
    """
    try:
        kwargs = {
            "cache_dir": cache_dir,
            "token": HF_TOKEN
        }
        if config:
            dataset = load_dataset(load_fn, config, **kwargs)
        else:
            dataset = load_dataset(load_fn, **kwargs)
        
        print(f"\n✅ Download of {dataset_name} completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {cache_dir}\n")
        
        return dataset
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Determine error category and provide appropriate message
        is_data_error = (
            "DatasetGenerationError" in error_type or
            "ArrowInvalid" in error_msg or
            "An error occurred while generating" in error_msg or
            "Couldn't cast array" in error_msg
        )
        is_auth_error = (
            "gated" in error_msg.lower() or 
            "authentication" in error_msg.lower() or
            "token" in error_msg.lower()
        )
        is_permission_error = isinstance(e, PermissionError)
        is_connection_error = isinstance(e, (ConnectionError, TimeoutError))
        
        if is_data_error:
            print(f"\n❌ Data generation error for {dataset_name}")
            print(f"   This is likely a data corruption issue in the HuggingFace repository.")
            print(f"   Please report this to NVIDIA on the dataset's HuggingFace page.")
            if not VERBOSE:
                print(f"   Error: {error_msg[:150]}...")
        elif is_auth_error:
            print(f"\n❌ Authentication error for {dataset_name}")
            print(f"   This dataset may be gated. Make sure you:")
            print(f"   1. Have accepted the license on HuggingFace")
            print(f"   2. Are logged in with: huggingface-cli login")
            print(f"   3. Have HF_TOKEN environment variable set")
        elif is_permission_error:
            print(f"\n❌ Permission denied for {dataset_name}")
            print(f"   Check your file permissions and disk space.")
        elif is_connection_error:
            print(f"\n❌ Connection error for {dataset_name}")
            print(f"   Please check your internet connection and try again.")
        else:
            print(f"\n❌ Error downloading {dataset_name}")
            print(f"   Type: {error_type}")
            if not VERBOSE:
                print(f"   Error: {error_msg[:150]}...")
        
        # Only print full traceback if verbose mode is enabled
        if VERBOSE:
            print(f"\n   Full traceback:")
            traceback.print_exc()
        else:
            print(f"   (Use --verbose for full traceback)")
        
        return None


def download_dataset(dataset_name: str, load_fn: str, cache_dir: str, config: str = None):
    """
    Universal download function that chooses the right method based on flags and dataset type.
    
    - If USE_JSONL is True, always use JSONL streaming mode
    - If dataset is in PROBLEMATIC_DATASETS, go DIRECTLY to raw parquet (skip streaming to avoid crash)
    - Otherwise, use normal parquet download
    """
    is_problematic = is_problematic_dataset(load_fn, config)
    
    if is_problematic and not USE_JSONL:
        # For problematic datasets, skip streaming entirely (it crashes!)
        # Go directly to raw parquet download with lenient readers
        jsonl_dir = str(Path(cache_dir).parent / "jsonl" / Path(cache_dir).name)
        Path(jsonl_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"   ⚠️ Known problematic dataset - skipping streaming, using direct parquet download")
        raw_count = download_raw_parquet_lenient(load_fn, jsonl_dir, dataset_name, config)
        
        if raw_count > 0:
            result = type('DatasetInfo', (), {
                'keys': lambda self: ['train'],
                '_split_info': {'train': raw_count},
                '__getitem__': lambda self, k: type('Split', (), {'__len__': lambda s: raw_count})()
            })()
            
            # Auto-convert to parquet if enabled
            if AUTO_CONVERT:
                convert_jsonl_to_parquet(jsonl_dir, cache_dir, dataset_name)
            
            return result
        return None
        
    elif USE_JSONL:
        # User explicitly requested JSONL streaming mode
        jsonl_dir = str(Path(cache_dir).parent / "jsonl" / Path(cache_dir).name)
        result = safe_download_jsonl(dataset_name, load_fn, jsonl_dir, config)
        
        # Check if download was partial (has .partial marker files)
        partial_files = list(Path(jsonl_dir).glob("*.partial"))
        
        # If partial due to corruption, try raw parquet fallback
        if partial_files:
            print(f"\n   🔄 Streaming hit corruption, trying raw parquet reading...")
            raw_count = download_raw_parquet_lenient(load_fn, jsonl_dir, dataset_name, config)
            if raw_count > 0:
                for pf in partial_files:
                    pf.unlink()
                result = type('DatasetInfo', (), {
                    'keys': lambda self: ['train'],
                    '_split_info': {'train': raw_count},
                    '__getitem__': lambda self, k: type('Split', (), {'__len__': lambda s: raw_count})()
                })()
        
        return result
    else:
        # Normal dataset - use standard download
        return safe_download(dataset_name, load_fn, cache_dir, config)


def download_nemotron_v1():
    """Download Nemotron Post-Training Dataset v1."""
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Post-Training-Dataset-v1{'  [JSONL mode]' if USE_JSONL else ''}...")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="Nemotron v1",
        load_fn="nvidia/Nemotron-Post-Training-Dataset-v1",
        cache_dir=str(DATASETS_DIR / "nemotron-v1")
    )


def download_nemotron_v2():
    """Download Nemotron Post-Training Dataset v2."""
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Post-Training-Dataset-v2{'  [JSONL mode]' if USE_JSONL else ''}...")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="Nemotron v2",
        load_fn="nvidia/Nemotron-Post-Training-Dataset-v2",
            cache_dir=str(DATASETS_DIR / "nemotron-v2")
        )


def download_llama_nemotron_sft():
    """Download Llama-Nemotron Post-Training Dataset (SFT subset)."""
    print("=" * 80)
    print(f"🔽 Downloading Llama-Nemotron-Post-Training-Dataset (SFT subset){'  [JSONL mode]' if USE_JSONL else ''}...")
    print("   Splits: math, code, science, chat, safety")
    print("   ⚠️  This is a LARGE dataset and may take significant time and disk space!")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="Llama-Nemotron SFT",
        load_fn="nvidia/Llama-Nemotron-Post-Training-Dataset",
        cache_dir=str(DATASETS_DIR / "llama-nemotron"),
        config="SFT"
    )


def download_llama_nemotron_rl():
    """Download Llama-Nemotron Post-Training Dataset (RL subset)."""
    is_problematic = is_problematic_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "RL")
    mode_str = " [JSONL mode]" if USE_JSONL else (" [Direct Parquet→JSONL]" if is_problematic else "")
    
    print("=" * 80)
    print(f"🔽 Downloading Llama-Nemotron-Post-Training-Dataset (RL subset){mode_str}...")
    print("   ⚠️  This is a LARGE dataset and may take significant time and disk space!")
    if is_problematic and not USE_JSONL:
        print("   ℹ️  Using direct parquet download (fastparquet) to bypass Arrow crashes")
    elif USE_JSONL:
        print("   ℹ️  Using JSONL streaming mode")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="Llama-Nemotron RL",
        load_fn="nvidia/Llama-Nemotron-Post-Training-Dataset",
        cache_dir=str(DATASETS_DIR / "llama-nemotron"),
        config="RL"
    )


# =============================================================================
# NEMOTRON V3 DATASETS (Post-Training Nano v3 Collection)
# =============================================================================

def download_nemotron_v3_rl_blend():
    """Download Nemotron-3-Nano-RL-Training-Blend dataset."""
    is_problematic = is_problematic_dataset("nvidia/Nemotron-3-Nano-RL-Training-Blend")
    mode_str = " [JSONL mode]" if USE_JSONL else (" [Direct Parquet→JSONL]" if is_problematic else "")
    
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-3-Nano-RL-Training-Blend (v3){mode_str}...")
    if is_problematic and not USE_JSONL:
        print("   ℹ️  Using direct parquet download (fastparquet) to bypass Arrow crashes")
    elif USE_JSONL:
        print("   ℹ️  Using JSONL streaming mode")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="v3 RL Blend",
        load_fn="nvidia/Nemotron-3-Nano-RL-Training-Blend",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "rl-blend")
        )


def download_nemotron_v3_science():
    """Download Nemotron-Science-v1 dataset."""
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Science-v1 (v3){'  [JSONL mode]' if USE_JSONL else ''}...")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="v3 Science",
        load_fn="nvidia/Nemotron-Science-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "science")
        )


def download_nemotron_v3_instruction_chat():
    """Download Nemotron-Instruction-Following-Chat-v1 dataset."""
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Instruction-Following-Chat-v1 (v3){'  [JSONL mode]' if USE_JSONL else ''}...")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="v3 Instruction Chat",
        load_fn="nvidia/Nemotron-Instruction-Following-Chat-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "instruction-chat")
        )


def download_nemotron_v3_math_proofs():
    """Download Nemotron-Math-Proofs-v1 dataset."""
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Math-Proofs-v1 (v3){'  [JSONL mode]' if USE_JSONL else ''}...")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="v3 Math Proofs",
        load_fn="nvidia/Nemotron-Math-Proofs-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "math-proofs")
        )


def download_nemotron_v3_agentic():
    """Download Nemotron-Agentic-v1 dataset."""
    is_problematic = is_problematic_dataset("nvidia/Nemotron-Agentic-v1")
    mode_str = " [JSONL mode]" if USE_JSONL else (" [Direct Parquet→JSONL]" if is_problematic else "")
    
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Agentic-v1 (v3){mode_str}...")
    if is_problematic and not USE_JSONL:
        print("   ℹ️  Using direct parquet download (fastparquet) to bypass Arrow crashes")
    elif USE_JSONL:
        print("   ℹ️  Using JSONL streaming mode")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="v3 Agentic",
        load_fn="nvidia/Nemotron-Agentic-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "agentic")
        )


def download_nemotron_v3_competitive_programming():
    """Download Nemotron-Competitive-Programming-v1 dataset."""
    is_problematic = is_problematic_dataset("nvidia/Nemotron-Competitive-Programming-v1")
    mode_str = " [JSONL mode]" if USE_JSONL else (" [Direct Parquet→JSONL]" if is_problematic else "")
    
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Competitive-Programming-v1 (v3){mode_str}...")
    if is_problematic and not USE_JSONL:
        print("   ℹ️  Using direct parquet download (fastparquet) to bypass Arrow crashes")
    elif USE_JSONL:
        print("   ℹ️  Using JSONL streaming mode")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="v3 Competitive Programming",
        load_fn="nvidia/Nemotron-Competitive-Programming-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "competitive-programming")
        )


def download_nemotron_v3_math():
    """Download Nemotron-Math-v2 dataset."""
    is_problematic = is_problematic_dataset("nvidia/Nemotron-Math-v2")
    mode_str = " [JSONL mode]" if USE_JSONL else (" [Direct Parquet→JSONL]" if is_problematic else "")
    
    print("=" * 80)
    print(f"🔽 Downloading Nemotron-Math-v2 (v3){mode_str}...")
    if is_problematic and not USE_JSONL:
        print("   ℹ️  Using direct parquet download (fastparquet) to bypass Arrow crashes")
    elif USE_JSONL:
        print("   ℹ️  Using JSONL streaming mode")
    print("=" * 80)
    
    return download_dataset(
        dataset_name="v3 Math v2",
        load_fn="nvidia/Nemotron-Math-v2",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "math-v2")
        )


def download_all_v3():
    """Download all Nemotron v3 datasets."""
    print("=" * 80)
    print("🔽 Downloading ALL Nemotron v3 Datasets (Post-Training Nano v3 Collection)...")
    print("   ⚠️  This includes 7 datasets and may take significant time and disk space!")
    print("=" * 80)
    
    v3_datasets = {}
    
    v3_datasets['v3-rl-blend'] = download_nemotron_v3_rl_blend()
    v3_datasets['v3-science'] = download_nemotron_v3_science()
    v3_datasets['v3-instruction-chat'] = download_nemotron_v3_instruction_chat()
    v3_datasets['v3-math-proofs'] = download_nemotron_v3_math_proofs()
    v3_datasets['v3-agentic'] = download_nemotron_v3_agentic()
    v3_datasets['v3-competitive-programming'] = download_nemotron_v3_competitive_programming()
    v3_datasets['v3-math'] = download_nemotron_v3_math()
    
    return v3_datasets


def display_summary(datasets_dict):
    """Display a summary of all downloaded datasets."""
    print("\n" + "=" * 80)
    print("📋 DATASET DOWNLOAD SUMMARY")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for name, dataset in datasets_dict.items():
        if dataset:
            successful += 1
            print(f"\n{name}:")
            print(f"  ✅ Downloaded successfully")
            print(f"  Splits: {', '.join(dataset.keys())}")
            print(f"  Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
            for split in dataset.keys():
                print(f"    - {split}: {len(dataset[split]):,} samples")
        else:
            failed += 1
            print(f"\n{name}:")
            print(f"  ❌ Download failed or skipped")
    
    print("\n" + "-" * 80)
    print(f"📊 Results: {successful} successful, {failed} failed")
    if failed > 0:
        print(f"   ℹ️  Use --verbose flag to see full error tracebacks")
    print("=" * 80)


def display_sample(dataset):
    """Display a sample from the SFT dataset."""
    if not dataset or 'math' not in dataset:
        return
    
    print("\n" + "=" * 80)
    print("🔍 Sample from Llama-Nemotron SFT (Math split)")
    print("=" * 80)
    
    sample = dataset['math'][0]
    for key, value in sample.items():
        print(f"\n{key}:")
        print("-" * 40)
        if isinstance(value, str) and len(value) > 500:
            print(value[:500] + "...")
        else:
            print(value)
    
    print("\n" + "=" * 80)


def main():
    """Main function to handle CLI arguments and coordinate downloads."""
    parser = argparse.ArgumentParser(
        description="Download NVIDIA Nemotron Post-Training Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets (default)
  python download_nemotron_datasets.py
  
  # Download only v1
  python download_nemotron_datasets.py --v1
  
  # Download v1 and v2
  python download_nemotron_datasets.py --v1 --v2
  
  # Download only Llama-Nemotron SFT subset
  python download_nemotron_datasets.py --llama-sft
  
  # Download all v3 datasets (Post-Training Nano v3 collection)
  python download_nemotron_datasets.py --v3
  
  # Download specific v3 datasets
  python download_nemotron_datasets.py --v3-science --v3-math
  
  # Custom directories
  python download_nemotron_datasets.py --datasets-dir /data/datasets --checkpoints-dir /data/checkpoints
  
  # JSONL mode - for datasets with parquet errors (streaming mode)
  python download_nemotron_datasets.py --jsonl --llama-rl
  python download_nemotron_datasets.py --jsonl --v3-rl-blend --v3-agentic --v3-math
        """
    )
    
    # Path arguments (already parsed, but include for help text)
    parser.add_argument(
        '--datasets-dir', type=str, default=str(DEFAULT_DATASETS_DIR),
        help=f'Directory for downloaded datasets (default: {DEFAULT_DATASETS_DIR})'
    )
    parser.add_argument(
        '--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help=f'Directory for model checkpoints/HF cache (default: {DEFAULT_CHECKPOINTS_DIR})'
    )
    
    # Dataset selection arguments
    parser.add_argument('--v1', action='store_true', help='Download Nemotron v1 dataset')
    parser.add_argument('--v2', action='store_true', help='Download Nemotron v2 dataset')
    parser.add_argument('--llama-sft', action='store_true', help='Download Llama-Nemotron SFT subset')
    parser.add_argument('--llama-rl', action='store_true', help='Download Llama-Nemotron RL subset')
    
    # V3 dataset arguments
    parser.add_argument('--v3', action='store_true', help='Download ALL Nemotron v3 datasets (Post-Training Nano v3)')
    parser.add_argument('--v3-rl-blend', action='store_true', help='Download Nemotron-3-Nano-RL-Training-Blend')
    parser.add_argument('--v3-science', action='store_true', help='Download Nemotron-Science-v1')
    parser.add_argument('--v3-instruction-chat', action='store_true', help='Download Nemotron-Instruction-Following-Chat-v1')
    parser.add_argument('--v3-math-proofs', action='store_true', help='Download Nemotron-Math-Proofs-v1')
    parser.add_argument('--v3-agentic', action='store_true', help='Download Nemotron-Agentic-v1')
    parser.add_argument('--v3-competitive-programming', action='store_true', help='Download Nemotron-Competitive-Programming-v1')
    parser.add_argument('--v3-math', action='store_true', help='Download Nemotron-Math-v2')
    
    parser.add_argument('--all', action='store_true', help='Download all datasets (default if no flags provided)')
    parser.add_argument('--show-sample', action='store_true', help='Display a sample from the downloaded datasets')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show full error tracebacks on failure')
    parser.add_argument('--jsonl', action='store_true', 
                        help='Use streaming mode and save as JSONL (bypasses parquet errors)')
    parser.add_argument('--no-convert', action='store_true',
                        help='Do not auto-convert JSONL to parquet for problematic datasets')
    
    args = parser.parse_args()
    
    # Set global flags
    global VERBOSE, USE_JSONL, AUTO_CONVERT
    VERBOSE = args.verbose
    USE_JSONL = args.jsonl
    AUTO_CONVERT = not args.no_convert
    
    # Check if any v3 specific flags are set
    v3_specific = (args.v3_rl_blend or args.v3_science or args.v3_instruction_chat or 
                   args.v3_math_proofs or args.v3_agentic or args.v3_competitive_programming or 
                   args.v3_math)
    
    # If no specific flags are provided, download all
    download_all = args.all or not (args.v1 or args.v2 or args.llama_sft or args.llama_rl or 
                                    args.v3 or v3_specific)
    
    print("=" * 80)
    print("🚀 NVIDIA Nemotron Post-Training Datasets Downloader")
    print("=" * 80)
    print()
    
    # Display environment info
    display_environment()
    
    # Dictionary to store downloaded datasets
    datasets = {}
    
    # Download requested datasets
    if download_all or args.v1:
        datasets['Nemotron v1'] = download_nemotron_v1()
    
    if download_all or args.v2:
        datasets['Nemotron v2'] = download_nemotron_v2()
    
    if download_all or args.llama_sft:
        datasets['Llama-Nemotron SFT'] = download_llama_nemotron_sft()
    
    if download_all or args.llama_rl:
        datasets['Llama-Nemotron RL'] = download_llama_nemotron_rl()
    
    # V3 datasets - download all if --v3 or --all, or individual ones if specified
    if download_all or args.v3:
        v3_datasets = download_all_v3()
        datasets.update(v3_datasets)
    else:
        # Individual v3 dataset downloads
        if args.v3_rl_blend:
            datasets['v3-rl-blend'] = download_nemotron_v3_rl_blend()
        if args.v3_science:
            datasets['v3-science'] = download_nemotron_v3_science()
        if args.v3_instruction_chat:
            datasets['v3-instruction-chat'] = download_nemotron_v3_instruction_chat()
        if args.v3_math_proofs:
            datasets['v3-math-proofs'] = download_nemotron_v3_math_proofs()
        if args.v3_agentic:
            datasets['v3-agentic'] = download_nemotron_v3_agentic()
        if args.v3_competitive_programming:
            datasets['v3-competitive-programming'] = download_nemotron_v3_competitive_programming()
        if args.v3_math:
            datasets['v3-math'] = download_nemotron_v3_math()
    
    # Display summary
    display_summary(datasets)
    
    # Display sample if requested
    if args.show_sample and 'Llama-Nemotron SFT' in datasets:
        display_sample(datasets['Llama-Nemotron SFT'])
    
    print("\n✅ All requested downloads completed!")
    print(f"📁 Datasets location: {DATASETS_DIR}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)



