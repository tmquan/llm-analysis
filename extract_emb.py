#!/usr/bin/env python3
"""
NVIDIA NIM Embedding Extractor

This script extracts embeddings from the NVIDIA NIM API using the datasets
downloaded by download_nemotron_datasets.py.

The script loads samples from Nemotron datasets and generates embeddings
using the local NIM API endpoint.

Usage:
    python extract_emb.py --dataset nemotron-v1 --split chat --num-samples 10
    python extract_emb.py --dataset llama-nemotron-sft --split math --output embeddings.json
    python extract_emb.py --test  # Quick API test with sample data
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    print("❌ Error: 'requests' library not found!")
    print("   Install it with: pip install requests")
    sys.exit(1)

try:
    from datasets import load_from_disk
except ImportError:
    print("❌ Error: 'datasets' library not found!")
    print("   Install it with: pip install datasets")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None  # NumPy is optional


CLOUD_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"
CLOUD_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
LOCAL_API_URL = "http://localhost:8000/v1/embeddings"
LOCAL_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
EMBEDDING_DIMENSION = 2048  # Expected embedding dimension for this model


class EmbeddingExtractor:
    """Extracts embeddings from NVIDIA NIM API (local or cloud)."""
    
    def __init__(self, api_url=LOCAL_API_URL, model=LOCAL_MODEL, timeout=30, api_key=None, use_cloud=False):
        # If use_cloud is True, switch to cloud URL and model unless explicitly overridden
        if use_cloud and api_url == LOCAL_API_URL:
            api_url = CLOUD_API_URL
        if use_cloud and model == LOCAL_MODEL:
            model = CLOUD_MODEL
            
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.getenv("NGC_API_KEY")
        self.use_cloud = use_cloud or "integrate.api.nvidia.com" in api_url
        
        # Validate API key for cloud mode
        if self.use_cloud and not self.api_key:
            print("⚠️  Warning: Cloud mode requires NGC_API_KEY")
            print("   Set it with: export NGC_API_KEY=your_key")
            print("   Or pass with: --api-key your_key")
        
    def _get_headers(self):
        """Get request headers based on local or cloud mode."""
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # Add authorization for cloud API
        if self.use_cloud and self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def check_api_health(self, verbose=False):
        """Check if the NIM API is responsive."""
        try:
            # Prepare request payload
            payload = {
                "input": ["test"],
                "model": self.model,
                "input_type": "query"
            }
            
            if verbose:
                print(f"   Sending payload: {json.dumps(payload, indent=2)}")
            
            # Try a simple embedding request
            response = requests.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=5
            )
            
            if verbose:
                print(f"   HTTP Status: {response.status_code}")
                if response.status_code not in [200, 422]:
                    try:
                        error_data = response.json()
                        print(f"   Error: {json.dumps(error_data, indent=2)}")
                    except:
                        print(f"   Response: {response.text[:500]}")
            
            return response.status_code in [200, 422]
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"   Connection error: {e}")
            return False
        except Exception as e:
            if verbose:
                print(f"   Unexpected error: {e}")
            return False
    
    def extract_embedding(self, text: str, input_type: str = "query") -> Optional[List[float]]:
        """
        Extract embedding for a single text.
        
        Args:
            text: Input text to embed
            input_type: Either "query" or "passage"
            
        Returns:
            Embedding vector as list of floats, or None on failure
        """
        try:
            payload = {
                "input": [text],
                "model": self.model,
                "input_type": input_type
            }
            
            response = requests.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['data'][0]['embedding']
            else:
                try:
                    error_msg = response.json().get('detail', response.text)
                except:
                    error_msg = response.text
                print(f"❌ API Error {response.status_code}: {error_msg}")
                print(f"   Payload sent: {json.dumps(payload, indent=2)}")
                print(f"   Response: {response.text[:500]}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def extract_batch(self, texts: List[str], input_type: str = "query", 
                     batch_size: int = 32, show_progress: bool = True) -> List[Optional[List[float]]]:
        """
        Extract embeddings for multiple texts with batching.
        
        Args:
            texts: List of input texts
            input_type: Either "query" or "passage"
            batch_size: Number of texts to process per request
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                payload = {
                    "input": batch,
                    "model": self.model,
                    "input_type": input_type
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    batch_embeddings = [item['embedding'] for item in data['data']]
                    embeddings.extend(batch_embeddings)
                    
                    if show_progress:
                        progress = min(i + batch_size, total)
                        print(f"   Progress: {progress}/{total} ({100*progress//total}%)", end='\r')
                else:
                    try:
                        error_msg = response.json().get('detail', response.text)
                    except:
                        error_msg = response.text
                    print(f"\n❌ API Error {response.status_code}: {error_msg}")
                    print(f"   Payload sent: {json.dumps(payload, indent=2)[:500]}")
                    print(f"   Response: {response.text[:500]}")
                    embeddings.extend([None] * len(batch))
                    
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Batch request failed: {e}")
                embeddings.extend([None] * len(batch))
        
        if show_progress:
            print()  # New line after progress
        
        return embeddings


class DatasetLoader:
    """Loads Nemotron datasets from disk."""
    
    DATASET_MAPPING = {
        'nemotron-v1': ('nvidia/Nemotron-Post-Training-Dataset-v1', './datasets/nemotron-v1'),
        'nemotron-v2': ('nvidia/Nemotron-Post-Training-Dataset-v2', './datasets/nemotron-v2'),
        'llama-nemotron-sft': ('nvidia/Llama-Nemotron-Post-Training-Dataset', './datasets/llama-nemotron'),
        'llama-nemotron-rl': ('nvidia/Llama-Nemotron-Post-Training-Dataset', './datasets/llama-nemotron'),
    }
    
    def __init__(self, datasets_dir='./datasets'):
        self.datasets_dir = Path(datasets_dir)
        # Import here to show better error messages
        try:
            from datasets import load_dataset
            self.load_dataset_fn = load_dataset
        except ImportError:
            print("❌ datasets library not found!")
            print("   Install with: pip install datasets")
            sys.exit(1)
    
    def list_available_datasets(self):
        """List all available datasets and their splits."""
        print("\n" + "=" * 80)
        print("📁 Available Datasets")
        print("=" * 80)
        
        found_any = False
        for name, (hf_path, cache_dir) in self.DATASET_MAPPING.items():
            full_path = Path(cache_dir)
            if full_path.exists():
                found_any = True
                print(f"\n✅ {name}")
                print(f"   HuggingFace path: {hf_path}")
                print(f"   Cache: {full_path.absolute()}")
                
                # Try to load and show splits
                try:
                    # Try loading to get split info
                    if 'Llama-Nemotron' in hf_path:
                        # Has SFT and RL configs
                        print(f"   Configs: SFT (math, code, science, chat, safety), RL")
                    else:
                        # Try to detect splits
                        print(f"   Typical splits: chat, code, math, stem, tool_calling")
                except Exception as e:
                    print(f"   (Could not read splits: {e})")
        
        if not found_any:
            print("\n❌ No datasets found!")
            print("   Run download_nemotron_datasets.py first to download datasets\n")
        else:
            print("\n" + "=" * 80)
    
    def load_dataset(self, dataset_name: str, split: Optional[str] = None):
        """
        Load a dataset from HuggingFace cache.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'nemotron-v1')
            split: Optional split name (e.g., 'chat', 'math')
            
        Returns:
            Loaded dataset or None on failure
        """
        if dataset_name not in self.DATASET_MAPPING:
            print(f"❌ Unknown dataset: {dataset_name}")
            print(f"   Available: {', '.join(self.DATASET_MAPPING.keys())}")
            return None
        
        hf_path, cache_dir = self.DATASET_MAPPING[dataset_name]
        
        # Check if cache directory exists
        if not Path(cache_dir).exists():
            print(f"❌ Dataset not found at: {cache_dir}")
            print(f"   Run: python download_nemotron_datasets.py to download it")
            return None
        
        try:
            # Load dataset using HuggingFace's load_dataset with cache
            print(f"   Loading from HuggingFace cache: {hf_path}")
            
            # For Llama-Nemotron, need to specify config
            if 'Llama-Nemotron' in hf_path:
                # Assume SFT for now (most common)
                dataset = self.load_dataset_fn(hf_path, "SFT", cache_dir=cache_dir)
            else:
                dataset = self.load_dataset_fn(hf_path, cache_dir=cache_dir)
            
            # If split is specified, extract it
            if split and hasattr(dataset, 'keys'):
                if split in dataset.keys():
                    return dataset[split]
                else:
                    print(f"❌ Split '{split}' not found. Available: {list(dataset.keys())}")
                    return None
            
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            print(f"   Trying alternative loading method...")
            
            # Try loading without cache_dir specification
            try:
                if 'Llama-Nemotron' in hf_path:
                    dataset = self.load_dataset_fn(hf_path, "SFT")
                else:
                    dataset = self.load_dataset_fn(hf_path)
                
                if split and hasattr(dataset, 'keys'):
                    if split in dataset.keys():
                        return dataset[split]
                
                return dataset
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                return None
    
    def extract_texts_from_samples(self, samples: List[Dict[str, Any]], 
                                   max_length: int = 512) -> List[str]:
        """
        Extract text content from dataset samples.
        
        Args:
            samples: List of dataset samples
            max_length: Maximum text length
            
        Returns:
            List of text strings
        """
        texts = []
        
        for sample in samples:
            # Try to find text content in various fields
            text = None
            
            # Common field names for text content
            for field in ['text', 'content', 'prompt', 'question', 'instruction', 'input']:
                if field in sample:
                    text = sample[field]
                    break
            
            # If no direct text field, try 'messages' structure (chat format)
            if not text and 'messages' in sample:
                messages = sample['messages']
                if isinstance(messages, list) and len(messages) > 0:
                    # Extract user message
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            text = msg.get('content', '')
                            break
            
            # If still no text, try to concatenate all string values
            if not text:
                string_values = [str(v) for v in sample.values() if isinstance(v, str) and len(str(v)) > 10]
                if string_values:
                    text = string_values[0]
            
            # Truncate if too long
            if text:
                text = str(text)[:max_length]
                texts.append(text)
        
        return texts


def run_api_test(extractor: EmbeddingExtractor):
    """Run a simple API test."""
    print("=" * 80)
    print("🧪 Testing NVIDIA NIM API")
    print("=" * 80)
    
    # Show mode
    mode = "☁️  CLOUD" if extractor.use_cloud else "🏠 LOCAL"
    print(f"\n📍 Mode: {mode}")
    print(f"   Endpoint: {extractor.api_url}")
    print(f"   Model: {extractor.model}")
    if extractor.use_cloud:
        print(f"   API Key: {'✅ Set' if extractor.api_key else '❌ Missing'}")
    
    # Check API health
    print("\n🔍 Checking API health...")
    if not extractor.check_api_health(verbose=True):
        print("❌ API is not responsive!")
        if extractor.use_cloud:
            print("   Check your NGC_API_KEY is valid")
            print("   Get one at: https://org.ngc.nvidia.com/setup/api-key")
        else:
            print("   Make sure the NIM container is running:")
            print("   python deploy_nim.py status")
        print()
        return False
    
    print("✅ API is responsive!")
    
    # Test sample embeddings
    test_texts = [
        "Hello world",
        "What is artificial intelligence?",
        "Python is a programming language"
    ]
    
    print(f"\n📝 Extracting embeddings for {len(test_texts)} test samples...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Sample {i}: \"{text[:50]}...\"" if len(text) > 50 else f"\n   Sample {i}: \"{text}\"")
        embedding = extractor.extract_embedding(text)
        
        if embedding:
            print(f"   ✅ Embedding dimension: {len(embedding)}")
            print(f"   ✅ First 5 values: {embedding[:5]}")
        else:
            print(f"   ❌ Failed to extract embedding")
    
    print("\n" + "=" * 80)
    print("✅ API test completed!\n")
    return True


def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from NVIDIA NIM API using Nemotron datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test API (cloud or local)
  python extract_emb.py --test --cloud
  
  # Extract from specific dataset splits
  python extract_emb.py --v1-chat --cloud --num-samples 100
  python extract_emb.py --llama-sft-math --cloud --num-samples 500
  
  # Extract from all Nemotron v1 splits
  python extract_emb.py --v1-all --cloud
  
  # Extract from multiple datasets
  python extract_emb.py --v1-chat --v2-code --llama-sft-math --cloud
  
  # Extract everything (not recommended, very large)
  python extract_emb.py --all --cloud --num-samples 100
  
  # Custom output location
  python extract_emb.py --llama-sft-math --cloud --output embeddings_math.json
        """
    )
    
    # Mode selection
    mode_group = parser.add_argument_group('🌐 API Mode Selection')
    mode_group.add_argument('--cloud', action='store_true', 
                           help='Use NVIDIA Cloud API (requires NGC_API_KEY)')
    mode_group.add_argument('--local', action='store_true', 
                           help='Use local NIM container (default)')
    mode_group.add_argument('--api-url', help=f'Custom API URL (overrides --cloud/--local)')
    mode_group.add_argument('--api-key', help='NGC API key for cloud mode (or set NGC_API_KEY env var)')
    
    # Operation mode
    parser.add_argument('--test', action='store_true', help='Run a quick API test')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    
    # Dataset selection (similar to download_nemotron_datasets.py)
    data_group = parser.add_argument_group('📊 Dataset Selection (Nemotron-style flags)')
    
    # Nemotron v1 splits
    data_group.add_argument('--v1-chat', action='store_true', help='Extract from Nemotron v1 chat split')
    data_group.add_argument('--v1-code', action='store_true', help='Extract from Nemotron v1 code split')
    data_group.add_argument('--v1-math', action='store_true', help='Extract from Nemotron v1 math split')
    data_group.add_argument('--v1-stem', action='store_true', help='Extract from Nemotron v1 stem split')
    data_group.add_argument('--v1-tool', action='store_true', help='Extract from Nemotron v1 tool_calling split')
    data_group.add_argument('--v1-all', action='store_true', help='Extract from all Nemotron v1 splits')
    
    # Nemotron v2 splits
    data_group.add_argument('--v2-chat', action='store_true', help='Extract from Nemotron v2 chat split')
    data_group.add_argument('--v2-code', action='store_true', help='Extract from Nemotron v2 code split')
    data_group.add_argument('--v2-math', action='store_true', help='Extract from Nemotron v2 math split')
    data_group.add_argument('--v2-stem', action='store_true', help='Extract from Nemotron v2 stem split')
    data_group.add_argument('--v2-tool', action='store_true', help='Extract from Nemotron v2 tool_calling split')
    data_group.add_argument('--v2-all', action='store_true', help='Extract from all Nemotron v2 splits')
    
    # Llama-Nemotron SFT splits
    data_group.add_argument('--llama-sft-math', action='store_true', help='Extract from Llama-Nemotron SFT math split')
    data_group.add_argument('--llama-sft-code', action='store_true', help='Extract from Llama-Nemotron SFT code split')
    data_group.add_argument('--llama-sft-science', action='store_true', help='Extract from Llama-Nemotron SFT science split')
    data_group.add_argument('--llama-sft-chat', action='store_true', help='Extract from Llama-Nemotron SFT chat split')
    data_group.add_argument('--llama-sft-safety', action='store_true', help='Extract from Llama-Nemotron SFT safety split')
    data_group.add_argument('--llama-sft-all', action='store_true', help='Extract from all Llama-Nemotron SFT splits')
    
    # Extract everything
    data_group.add_argument('--all', action='store_true', help='Extract from ALL datasets (warning: very large!)')
    
    # Processing options
    proc_group = parser.add_argument_group('⚙️  Processing Options')
    proc_group.add_argument('--num-samples', type=int, default=100, 
                           help='Number of samples per split (default: 100, use -1 for all)')
    proc_group.add_argument('--output', default='embeddings.jsonl',
                           help='Output file path (default: embeddings.jsonl)')
    proc_group.add_argument('--model', help=f'Model name (auto-selected based on mode)')
    proc_group.add_argument('--input-type', choices=['query', 'passage'], default='passage', 
                           help='Input type for embeddings (default: passage)')
    proc_group.add_argument('--batch-size', type=int, default=64, help='Batch size for API requests (default: 64)')
    proc_group.add_argument('--max-text-length', type=int, default=8192, help='Maximum text length (default: 8192)')
    
    args = parser.parse_args()
    
    # Determine mode
    use_cloud = args.cloud or (args.api_url and "integrate.api.nvidia.com" in args.api_url)
    
    # Set defaults based on mode
    if args.api_url:
        api_url = args.api_url
    else:
        api_url = CLOUD_API_URL if use_cloud else LOCAL_API_URL
    
    model = args.model or (CLOUD_MODEL if use_cloud else LOCAL_MODEL)
    
    # Create extractor
    extractor = EmbeddingExtractor(
        api_url=api_url, 
        model=model,
        api_key=args.api_key,
        use_cloud=use_cloud
    )
    
    # Handle --test
    if args.test:
        success = run_api_test(extractor)
        sys.exit(0 if success else 1)
    
    # Handle --list
    if args.list:
        loader = DatasetLoader()
        loader.list_available_datasets()
        sys.exit(0)
    
    # Determine which datasets/splits to process
    datasets_to_process = []
    
    # Nemotron v1
    if args.v1_all or args.all:
        datasets_to_process.extend([
            ('nemotron-v1', 'chat'),
            ('nemotron-v1', 'code'),
            ('nemotron-v1', 'math'),
            ('nemotron-v1', 'stem'),
            ('nemotron-v1', 'tool_calling'),
        ])
    else:
        if args.v1_chat: datasets_to_process.append(('nemotron-v1', 'chat'))
        if args.v1_code: datasets_to_process.append(('nemotron-v1', 'code'))
        if args.v1_math: datasets_to_process.append(('nemotron-v1', 'math'))
        if args.v1_stem: datasets_to_process.append(('nemotron-v1', 'stem'))
        if args.v1_tool: datasets_to_process.append(('nemotron-v1', 'tool_calling'))
    
    # Nemotron v2
    if args.v2_all or args.all:
        datasets_to_process.extend([
            ('nemotron-v2', 'chat'),
            ('nemotron-v2', 'code'),
            ('nemotron-v2', 'math'),
            ('nemotron-v2', 'stem'),
            ('nemotron-v2', 'tool_calling'),
        ])
    else:
        if args.v2_chat: datasets_to_process.append(('nemotron-v2', 'chat'))
        if args.v2_code: datasets_to_process.append(('nemotron-v2', 'code'))
        if args.v2_math: datasets_to_process.append(('nemotron-v2', 'math'))
        if args.v2_stem: datasets_to_process.append(('nemotron-v2', 'stem'))
        if args.v2_tool: datasets_to_process.append(('nemotron-v2', 'tool_calling'))
    
    # Llama-Nemotron SFT
    if args.llama_sft_all or args.all:
        datasets_to_process.extend([
            ('llama-nemotron-sft', 'math'),
            ('llama-nemotron-sft', 'code'),
            ('llama-nemotron-sft', 'science'),
            ('llama-nemotron-sft', 'chat'),
            ('llama-nemotron-sft', 'safety'),
        ])
    else:
        if args.llama_sft_math: datasets_to_process.append(('llama-nemotron-sft', 'math'))
        if args.llama_sft_code: datasets_to_process.append(('llama-nemotron-sft', 'code'))
        if args.llama_sft_science: datasets_to_process.append(('llama-nemotron-sft', 'science'))
        if args.llama_sft_chat: datasets_to_process.append(('llama-nemotron-sft', 'chat'))
        if args.llama_sft_safety: datasets_to_process.append(('llama-nemotron-sft', 'safety'))
    
    # Check if any datasets were selected
    if not datasets_to_process:
        print("❌ Error: Please specify at least one dataset to process\n")
        print("Examples:")
        print("  python extract_emb.py --v1-chat --cloud")
        print("  python extract_emb.py --llama-sft-math --cloud")
        print("  python extract_emb.py --v1-all --cloud")
        print("\nUse --help for all options\n")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 NVIDIA NIM Embedding Extraction")
    print("=" * 80)
    
    # Show mode
    mode = "☁️  CLOUD API" if extractor.use_cloud else "🏠 LOCAL NIM"
    print(f"\n📍 Mode: {mode}")
    print(f"   Endpoint: {extractor.api_url}")
    print(f"   Model: {extractor.model}")
    if extractor.use_cloud:
        print(f"   API Key: {'✅ Set' if extractor.api_key else '❌ Missing'}")
    
    print(f"\n📊 Datasets to process: {len(datasets_to_process)}")
    for dataset_name, split in datasets_to_process:
        print(f"   - {dataset_name}/{split}")
    print(f"   Samples per split: {args.num_samples if args.num_samples > 0 else 'ALL'}")
    print(f"   Output: {args.output}")
    
    # Check API
    print("\n🔍 Checking NIM API...")
    if not extractor.check_api_health(verbose=True):
        print("❌ API is not responsive!")
        if extractor.use_cloud:
            print("   Check your NGC_API_KEY is valid")
            print("   Get one at: https://org.ngc.nvidia.com/setup/api-key")
        else:
            print("   Make sure the NIM container is running:")
            print("   python deploy_nim.py status")
        print()
        sys.exit(1)
    print("✅ API is ready!")
    
    # Process all datasets
    loader = DatasetLoader()
    all_embeddings = []
    total_processed = 0
    
    for dataset_idx, (dataset_name, split) in enumerate(datasets_to_process, 1):
        print(f"\n{'='*80}")
        print(f"📁 Processing dataset {dataset_idx}/{len(datasets_to_process)}: {dataset_name}/{split}")
        print(f"{'='*80}")
        
        # Load dataset
        dataset = loader.load_dataset(dataset_name, split)
        
        if dataset is None:
            print(f"⚠️  Skipping {dataset_name}/{split} - not found")
            continue
        
        # Handle dataset with splits
        if hasattr(dataset, 'keys'):
            splits = list(dataset.keys())
            if split in splits:
                dataset = dataset[split]
            else:
                print(f"⚠️  Split '{split}' not found in {dataset_name}, skipping")
                continue
        
        # Determine number of samples
        num_samples = args.num_samples if args.num_samples > 0 else len(dataset)
        num_samples = min(num_samples, len(dataset))
        
        print(f"\n📊 Extracting {num_samples} samples from {dataset_name}/{split}")
        samples = [dataset[i] for i in range(num_samples)]
        
        # Extract texts
        print(f"   Extracting text content...")
        texts = loader.extract_texts_from_samples(samples, max_length=args.max_text_length)
        print(f"   ✅ Extracted {len(texts)} texts")
        
        if not texts:
            print(f"⚠️  No text content found in {dataset_name}/{split}, skipping")
            continue
        
        # Extract embeddings
        print(f"\n🔮 Generating embeddings (batch_size={args.batch_size})...")
        embeddings = extractor.extract_batch(
            texts, 
            input_type=args.input_type,
            batch_size=args.batch_size,
            show_progress=True
        )
        
        # Count successful embeddings
        successful = sum(1 for emb in embeddings if emb is not None)
        print(f"   ✅ Successfully generated {successful}/{len(embeddings)} embeddings")
        
        if successful == 0:
            print(f"⚠️  No embeddings generated for {dataset_name}/{split}, skipping")
            continue
        
        # Add to all_embeddings with rich metadata for UMAP
        for i, (text, embedding, sample) in enumerate(zip(texts, embeddings, samples)):
            if embedding is not None:  # Only include successful embeddings
                # Extract additional metadata from sample
                metadata = {
                    'dataset': dataset_name,
                    'split': split,
                    'index_in_split': i,
                    'global_index': total_processed,
                    'text': text,
                    'text_length': len(text),
                    'embedding': embedding,
                }
                
                # Add original sample fields (truncated for size)
                for key, value in sample.items():
                    if key not in ['text', 'embedding']:  # Avoid duplication
                        if isinstance(value, str):
                            metadata[f'original_{key}'] = value[:200] if len(value) > 200 else value
                        elif isinstance(value, (int, float, bool)):
                            metadata[f'original_{key}'] = value
                        elif isinstance(value, list) and len(value) > 0:
                            # For chat/message lists, extract info
                            if isinstance(value[0], dict):
                                metadata[f'original_{key}_count'] = len(value)
                            else:
                                metadata[f'original_{key}'] = str(value)[:200]
                
                all_embeddings.append(metadata)
                total_processed += 1
        
        print(f"   Added {successful} samples to collection (total: {total_processed})")
    
    # Check if we got any embeddings
    if not all_embeddings:
        print("\n❌ No embeddings were successfully generated!")
        sys.exit(1)
    
    # Prepare output with rich metadata for UMAP visualization
    import time
    
    # Create metadata header
    metadata = {
        'type': 'metadata',
        'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': extractor.model,
        'api_mode': 'cloud' if extractor.use_cloud else 'local',
        'input_type': args.input_type,
        'embedding_dimension': len(all_embeddings[0]['embedding']) if all_embeddings else EMBEDDING_DIMENSION,
        'total_samples': len(all_embeddings),
        'datasets_processed': len(datasets_to_process),
        'max_text_length': args.max_text_length,
        'batch_size': args.batch_size,
        'dataset_splits': [
            {'dataset': ds, 'split': sp} for ds, sp in datasets_to_process
        ],
        'format': 'jsonl',
        'format_description': 'First line contains metadata, subsequent lines contain embeddings'
    }
    
    # Save results in JSONL format
    output_path = Path(args.output)
    print(f"\n💾 Saving {len(all_embeddings)} embeddings to: {output_path} (JSONL format)")
    
    with open(output_path, 'w') as f:
        # Write metadata as first line
        f.write(json.dumps(metadata) + '\n')
        
        # Write each embedding as a separate line
        for emb_data in all_embeddings:
            emb_data['type'] = 'embedding'  # Mark as embedding entry
            f.write(json.dumps(emb_data) + '\n')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved successfully!")
    print(f"   Format: JSONL (JSON Lines)")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Total embeddings: {len(all_embeddings)}")
    print(f"   Embedding dimension: {metadata['embedding_dimension']}")
    
    # Show summary by split
    print(f"\n📊 Summary by split:")
    split_counts = {}
    for emb in all_embeddings:
        key = f"{emb['dataset']}/{emb['split']}"
        split_counts[key] = split_counts.get(key, 0) + 1
    
    for split_key, count in sorted(split_counts.items()):
        print(f"   {split_key}: {count} samples")
    
    print("\n" + "=" * 80)
    print("✅ Embedding extraction completed!")
    print(f"\n💡 How to load JSONL file:")
    print(f"   import json")
    print(f"   with open('{output_path}') as f:")
    print(f"       metadata = json.loads(f.readline())")
    print(f"       embeddings = [json.loads(line) for line in f]")
    print(f"\n   Or using pandas:")
    print(f"       import pandas as pd")
    print(f"       df = pd.read_json('{output_path}', lines=True)")
    print(f"       metadata = df[df['type'] == 'metadata'].iloc[0]")
    print(f"       embeddings = df[df['type'] == 'embedding']")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

