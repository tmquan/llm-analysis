#!/usr/bin/env python3
"""
NVIDIA Nemotron Post-Training Datasets Downloader

This script downloads NVIDIA's Nemotron post-training datasets, which are used
for fine-tuning large language models.

Available Datasets:
1. Nemotron-Post-Training-Dataset-v1: Original version with chat, code, math, stem, and tool_calling splits
2. Nemotron-Post-Training-Dataset-v2: Updated version with improvements
3. Llama-Nemotron-Post-Training-Dataset: Comprehensive dataset with SFT and RL subsets

Usage:
    python download_nemotron_datasets.py [--v1] [--v2] [--llama-sft] [--llama-rl] [--all]
    
    If no flags are provided, all datasets will be downloaded.
"""

import os
import sys
import argparse
from pathlib import Path

# Set HuggingFace cache directories BEFORE importing datasets
SCRIPT_DIR = Path(__file__).parent.absolute()
DATASETS_DIR = SCRIPT_DIR / "datasets"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

# Ensure directories exist
DATASETS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Set environment variables for HuggingFace cache
# Models/hub go to checkpoints (in case any models are downloaded)
os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)
os.environ['HF_MODULES_CACHE'] = str(CHECKPOINTS_DIR / "modules")

# Datasets go to datasets folder
os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)

# Now import datasets library
from datasets import load_dataset


def setup_environment():
    """Display environment information."""
    print(f"📁 Datasets will be cached in: {DATASETS_DIR}")
    print(f"🤖 Models will be cached in: {CHECKPOINTS_DIR}")
    print("✅ Environment setup complete\n")


def download_nemotron_v1():
    """Download Nemotron Post-Training Dataset v1."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Post-Training-Dataset-v1...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v1",
            cache_dir="./datasets/nemotron-v1"
        )
        
        print(f"\n✅ Download of v1 completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {os.path.abspath('datasets/nemotron-v1')}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v1: {e}")
        return None


def download_nemotron_v2():
    """Download Nemotron Post-Training Dataset v2."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Post-Training-Dataset-v2...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            cache_dir="./datasets/nemotron-v2"
        )
        
        print(f"\n✅ Download of v2 completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {os.path.abspath('datasets/nemotron-v2')}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v2: {e}")
        return None


def download_llama_nemotron_sft():
    """Download Llama-Nemotron Post-Training Dataset (SFT subset)."""
    print("=" * 80)
    print("🔽 Downloading Llama-Nemotron-Post-Training-Dataset (SFT subset)...")
    print("   Splits: math, code, science, chat, safety")
    print("   ⚠️  This is a LARGE dataset and may take significant time and disk space!")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "SFT",
            cache_dir="./datasets/llama-nemotron"
        )
        
        print(f"\n✅ Download of SFT subset completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {os.path.abspath('datasets/llama-nemotron')}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading SFT subset: {e}")
        return None


def download_llama_nemotron_rl():
    """Download Llama-Nemotron Post-Training Dataset (RL subset)."""
    print("=" * 80)
    print("🔽 Downloading Llama-Nemotron-Post-Training-Dataset (RL subset)...")
    print("   ⚠️  This is a LARGE dataset and may take significant time and disk space!")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "RL",
            cache_dir="./datasets/llama-nemotron"
        )
        
        print(f"\n✅ Download of RL subset completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {os.path.abspath('datasets/llama-nemotron')}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading RL subset: {e}")
        return None


def display_summary(datasets_dict):
    """Display a summary of all downloaded datasets."""
    print("\n" + "=" * 80)
    print("📋 DATASET DOWNLOAD SUMMARY")
    print("=" * 80)
    
    for name, dataset in datasets_dict.items():
        if dataset:
            print(f"\n{name}:")
            print(f"  ✅ Downloaded successfully")
            print(f"  Splits: {', '.join(dataset.keys())}")
            print(f"  Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
            for split in dataset.keys():
                print(f"    - {split}: {len(dataset[split]):,} samples")
        else:
            print(f"\n{name}:")
            print(f"  ❌ Download failed or skipped")
    
    print("\n" + "=" * 80)


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
        """
    )
    
    parser.add_argument('--v1', action='store_true', help='Download Nemotron v1 dataset')
    parser.add_argument('--v2', action='store_true', help='Download Nemotron v2 dataset')
    parser.add_argument('--llama-sft', action='store_true', help='Download Llama-Nemotron SFT subset')
    parser.add_argument('--llama-rl', action='store_true', help='Download Llama-Nemotron RL subset')
    parser.add_argument('--all', action='store_true', help='Download all datasets (default if no flags provided)')
    parser.add_argument('--show-sample', action='store_true', help='Display a sample from the downloaded datasets')
    
    args = parser.parse_args()
    
    # If no specific flags are provided, download all
    download_all = args.all or not (args.v1 or args.v2 or args.llama_sft or args.llama_rl)
    
    print("=" * 80)
    print("🚀 NVIDIA Nemotron Post-Training Datasets Downloader")
    print("=" * 80)
    print()
    
    # Setup environment
    setup_environment()
    
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
    
    # Display summary
    display_summary(datasets)
    
    # Display sample if requested
    if args.show_sample and 'Llama-Nemotron SFT' in datasets:
        display_sample(datasets['Llama-Nemotron SFT'])
    
    print("\n✅ All requested downloads completed!")
    print(f"📁 Datasets location: {os.path.abspath('datasets')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)



