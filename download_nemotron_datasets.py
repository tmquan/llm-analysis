#!/usr/bin/env python3
"""
NVIDIA Nemotron Post-Training Datasets Downloader

This script downloads NVIDIA's Nemotron post-training datasets, which are used
for fine-tuning large language models.

Available Datasets:
1. Nemotron-Post-Training-Dataset-v1: Original version with chat, code, math, stem, and tool_calling splits
2. Nemotron-Post-Training-Dataset-v2: Updated version with improvements
3. Llama-Nemotron-Post-Training-Dataset: Comprehensive dataset with SFT and RL subsets
4. Nemotron v3 Collection (Post-Training Nano v3):
   - nvidia/Nemotron-3-Nano-RL-Training-Blend
   - nvidia/Nemotron-Science-v1
   - nvidia/Nemotron-Instruction-Following-Chat-v1
   - nvidia/Nemotron-Math-Proofs-v1
   - nvidia/Nemotron-Agentic-v1
   - nvidia/Nemotron-Competitive-Programming-v1
   - nvidia/Nemotron-Math-v2

Usage:
    python download_nemotron_datasets.py [--v1] [--v2] [--llama-sft] [--llama-rl] [--v3] [--all]
    
    Custom paths:
    python download_nemotron_datasets.py --datasets-dir /path/to/datasets --checkpoints-dir /path/to/checkpoints
    
    If no flags are provided, all datasets will be downloaded.
"""

import os
import sys
import argparse
from pathlib import Path

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


def download_nemotron_v1():
    """Download Nemotron Post-Training Dataset v1."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Post-Training-Dataset-v1...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v1")
        )
        
        print(f"\n✅ Download of v1 completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v1'}\n")
        
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
            cache_dir=str(DATASETS_DIR / "nemotron-v2")
        )
        
        print(f"\n✅ Download of v2 completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v2'}\n")
        
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
            cache_dir=str(DATASETS_DIR / "llama-nemotron")
        )
        
        print(f"\n✅ Download of SFT subset completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'llama-nemotron'}\n")
        
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
            cache_dir=str(DATASETS_DIR / "llama-nemotron")
        )
        
        print(f"\n✅ Download of RL subset completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'llama-nemotron'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading RL subset: {e}")
        return None


# =============================================================================
# NEMOTRON V3 DATASETS (Post-Training Nano v3 Collection)
# =============================================================================

def download_nemotron_v3_rl_blend():
    """Download Nemotron-3-Nano-RL-Training-Blend dataset."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-3-Nano-RL-Training-Blend (v3)...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-3-Nano-RL-Training-Blend",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "rl-blend")
        )
        
        print(f"\n✅ Download of v3 RL Blend completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v3' / 'rl-blend'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v3 RL Blend: {e}")
        return None


def download_nemotron_v3_science():
    """Download Nemotron-Science-v1 dataset."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Science-v1 (v3)...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Science-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "science")
        )
        
        print(f"\n✅ Download of v3 Science completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v3' / 'science'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v3 Science: {e}")
        return None


def download_nemotron_v3_instruction_chat():
    """Download Nemotron-Instruction-Following-Chat-v1 dataset."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Instruction-Following-Chat-v1 (v3)...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Instruction-Following-Chat-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "instruction-chat")
        )
        
        print(f"\n✅ Download of v3 Instruction Chat completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v3' / 'instruction-chat'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v3 Instruction Chat: {e}")
        return None


def download_nemotron_v3_math_proofs():
    """Download Nemotron-Math-Proofs-v1 dataset."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Math-Proofs-v1 (v3)...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Math-Proofs-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "math-proofs")
        )
        
        print(f"\n✅ Download of v3 Math Proofs completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v3' / 'math-proofs'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v3 Math Proofs: {e}")
        return None


def download_nemotron_v3_agentic():
    """Download Nemotron-Agentic-v1 dataset."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Agentic-v1 (v3)...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Agentic-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "agentic")
        )
        
        print(f"\n✅ Download of v3 Agentic completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v3' / 'agentic'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v3 Agentic: {e}")
        return None


def download_nemotron_v3_competitive_programming():
    """Download Nemotron-Competitive-Programming-v1 dataset."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Competitive-Programming-v1 (v3)...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Competitive-Programming-v1",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "competitive-programming")
        )
        
        print(f"\n✅ Download of v3 Competitive Programming completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v3' / 'competitive-programming'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v3 Competitive Programming: {e}")
        return None


def download_nemotron_v3_math():
    """Download Nemotron-Math-v2 dataset."""
    print("=" * 80)
    print("🔽 Downloading Nemotron-Math-v2 (v3)...")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Math-v2",
            cache_dir=str(DATASETS_DIR / "nemotron-v3" / "math-v2")
        )
        
        print(f"\n✅ Download of v3 Math v2 completed!")
        print(f"   Dataset splits: {list(dataset.keys())}")
        print(f"   Total samples: {sum(len(dataset[split]) for split in dataset.keys()):,}")
        print(f"   Location: {DATASETS_DIR / 'nemotron-v3' / 'math-v2'}\n")
        
        return dataset
    except Exception as e:
        print(f"❌ Error downloading v3 Math v2: {e}")
        return None


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
  
  # Download all v3 datasets (Post-Training Nano v3 collection)
  python download_nemotron_datasets.py --v3
  
  # Download specific v3 datasets
  python download_nemotron_datasets.py --v3-science --v3-math
  
  # Custom directories
  python download_nemotron_datasets.py --datasets-dir /data/datasets --checkpoints-dir /data/checkpoints
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
    
    args = parser.parse_args()
    
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



