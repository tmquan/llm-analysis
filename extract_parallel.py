#!/usr/bin/env python3
"""
Standalone Multi-GPU Parallel Extraction
All GPUs work on the same split, each processing different samples
"""

import os
import sys
import json
import time
import math
from pathlib import Path
import multiprocessing as mp

# Set HuggingFace cache directories
SCRIPT_DIR = Path(__file__).parent.absolute()
DATASETS_DIR = SCRIPT_DIR / "datasets"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

# HuggingFace datasets go to ./datasets
# HuggingFace hub (models) go to ./checkpoints
os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(CHECKPOINTS_DIR)
os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("   Install: pip install sentence-transformers torch datasets pyarrow")
    sys.exit(1)


def extract_text(sample, max_length=8192):
    """Extract text from a sample"""
    # Try common fields
    for field in ['text', 'content', 'prompt', 'question', 'instruction']:
        if field in sample:
            text = str(sample[field])[:max_length]
            return text
    
    # Try messages structure
    if 'messages' in sample and isinstance(sample['messages'], list):
        for msg in sample['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return str(msg.get('content', ''))[:max_length]
    
    # Fallback: first string value
    for v in sample.values():
        if isinstance(v, str) and len(v) > 10:
            return str(v)[:max_length]
    
    return ""


def worker(gpu_id, dataset_name, split, start_idx, end_idx, output_dir, config, chunk_idx, total_chunks):
    """Worker process for one GPU processing one chunk"""
    # Set HuggingFace cache directories
    script_dir = Path(__file__).parent.absolute()
    datasets_dir = script_dir / "datasets"
    checkpoints_dir = script_dir / "checkpoints"
    
    os.environ['HF_HOME'] = str(checkpoints_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(checkpoints_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(checkpoints_dir)
    os.environ['HF_DATASETS_CACHE'] = str(datasets_dir)
    
    # Set GPU BEFORE any CUDA imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Chunk {chunk_idx+1}/{total_chunks}: samples {start_idx}-{end_idx}")
    
    try:
        # Now import torch (CUDA will initialize with correct device)
        import torch
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
        
        # Load model
        print(f"[GPU {gpu_id}] Loading model...")
        model = SentenceTransformer(
            config['model'],
            trust_remote_code=True,
            device='cuda'
        )
        
        # Load dataset
        print(f"[GPU {gpu_id}] Loading dataset...")
        dataset_paths = {
            'nemotron-v1': 'nvidia/Nemotron-Post-Training-Dataset-v1',
            'nemotron-v2': 'nvidia/Nemotron-Post-Training-Dataset-v2',
            'llama-nemotron-sft': 'nvidia/Llama-Nemotron-Post-Training-Dataset'
        }
        
        if dataset_name == 'llama-nemotron-sft':
            ds = load_dataset(dataset_paths[dataset_name], "SFT", cache_dir=f'./datasets/{dataset_name}')
        else:
            ds = load_dataset(dataset_paths[dataset_name], cache_dir=f'./datasets/{dataset_name}')
        
        if split in ds.keys():
            ds = ds[split]
        else:
            print(f"[GPU {gpu_id}] ❌ Split '{split}' not found")
            return False
        
        # Extract samples
        samples = [ds[i] for i in range(start_idx, end_idx)]
        texts = [extract_text(s, config['max_text_length']) for s in samples]
        texts = [t for t in texts if t]  # Filter empty
        
        if not texts:
            print(f"[GPU {gpu_id}] ❌ No texts extracted")
            return False
        
        print(f"[GPU {gpu_id}] Generating embeddings for {len(texts)} samples...")
        
        # Generate embeddings in batches
        allembeddings = []
        batch_size = config['batch_size']
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if config['input_type'] == 'query':
                embs = model.encode_query(batch)
            else:
                embs = model.encode_document(batch)
            allembeddings.extend([emb.tolist() for emb in embs])
            
            if i % (batch_size * 10) == 0:
                print(f"[GPU {gpu_id}]   {i}/{len(texts)} ({100*i//len(texts)}%)")
        
        print(f"[GPU {gpu_id}] Generated {len(allembeddings)} embeddings")
        
        # Prepare records
        records = []
        for i, (sample, text, emb) in enumerate(zip(samples, texts, allembeddings)):
            rec = {}
            # Copy original fields
            for k, v in sample.items():
                if isinstance(v, (list, dict)):
                    rec[k] = json.dumps(v)
                else:
                    rec[k] = str(v) if v is not None else ""
            
            # Add metadata
            rec['embedding'] = emb
            rec['dataset'] = dataset_name
            rec['split'] = split
            rec['text'] = text
            rec['text_length'] = str(len(text))
            rec['lang'] = 'en'
            rec['chunk_id'] = str(chunk_idx)
            records.append(rec)
        
        # Save to parquet - name by chunk number
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_file = output_dir / f"data-{chunk_idx+1:05d}-of-{total_chunks:05d}.parquet"
        
        # Build schema
        emb_dim = len(records[0]['embedding'])
        fields = [pa.field('embedding', pa.list_(pa.float32(), emb_dim))]
        
        for key in records[0].keys():
            if key != 'embedding':
                fields.append(pa.field(key, pa.string()))
        
        schema = pa.schema(fields)
        
        # Build arrays
        arrays = []
        for field in fields:
            if field.name == 'embedding':
                arrays.append(pa.array([r['embedding'] for r in records], type=field.type))
            else:
                arrays.append(pa.array([r[field.name] for r in records], type=pa.string()))
        
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, shard_file, compression='snappy')
        
        size_mb = shard_file.stat().st_size / (1024 * 1024)
        print(f"[GPU {gpu_id}] ✅ Saved {shard_file.name}: {len(records)} samples, {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_split(dataset_name, split, num_gpus, config):
    """Process one split using all GPUs in parallel, working on chunks"""
    print(f"\n{'='*80}")
    print(f"📁 Processing: {dataset_name}/{split}")
    print(f"   Using {num_gpus} GPUs in parallel")
    print(f"{'='*80}\n")
    
    # Load dataset metadata
    from datasets import load_dataset
    
    dataset_paths = {
        'nemotron-v1': 'nvidia/Nemotron-Post-Training-Dataset-v1',
        'nemotron-v2': 'nvidia/Nemotron-Post-Training-Dataset-v2',
        'llama-nemotron-sft': 'nvidia/Llama-Nemotron-Post-Training-Dataset'
    }
    
    try:
        if dataset_name == 'llama-nemotron-sft':
            ds = load_dataset(dataset_paths[dataset_name], "SFT", cache_dir=f'./datasets/{dataset_name}')
        else:
            ds = load_dataset(dataset_paths[dataset_name], cache_dir=f'./datasets/{dataset_name}')
        
        if split not in ds.keys():
            print(f"❌ Split '{split}' not found. Available: {list(ds.keys())}")
            return False
        
        total_samples = len(ds[split])
        
        # Determine chunk size and total chunks
        if config['num_samples'] > 0:
            chunk_size = config['num_samples']
        else:
            # If -1, divide into chunks that fit num_gpus
            chunk_size = math.ceil(total_samples / num_gpus)
        
        total_chunks = math.ceil(total_samples / chunk_size)
        
        print(f"📊 Total samples: {total_samples:,}")
        print(f"   Chunk size: {chunk_size:,}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   GPUs: {num_gpus}")
        print(f"   Batches: {math.ceil(total_chunks / num_gpus)}\n")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Output directory
    if dataset_name == 'llama-nemotron-sft':
        output_dir = Path(config['output']) / f"Nemotron-Post-Training-Dataset-llama-sft-{split}"
    else:
        output_dir = Path(config['output']) / f"Nemotron-Post-Training-Dataset-{dataset_name}-{split}"
    
    # Process chunks in batches (8 chunks at a time using 8 GPUs)
    all_success = True
    
    for batch_start in range(0, total_chunks, num_gpus):
        batch_end = min(batch_start + num_gpus, total_chunks)
        batch_chunks = batch_end - batch_start
        
        print(f"🔄 Processing chunks {batch_start+1}-{batch_end} of {total_chunks}")
        
        # Launch workers for this batch
        processes = []
        for i in range(batch_chunks):
            chunk_idx = batch_start + i
            gpu_id = i  # GPU 0-7 for each batch
            
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            
            if start_idx >= total_samples:
                break
            
            p = mp.Process(
                target=worker,
                args=(gpu_id, dataset_name, split, start_idx, end_idx, output_dir, config, chunk_idx, total_chunks)
            )
            p.start()
            processes.append(p)
            time.sleep(0.5)  # Stagger startup
        
        # Wait for this batch to complete
        for p in processes:
            p.join()
            if p.exitcode != 0:
                all_success = False
        
        print(f"✅ Batch {batch_start//num_gpus + 1} complete\n")
    
    if all_success:
        # Save metadata
        metadata = {
            'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': config['model'],
            'dataset': dataset_name,
            'split': split,
            'total_samples': total_samples,
            'chunk_size': chunk_size,
            'total_chunks': total_chunks,
            'num_gpus': num_gpus,
            'embedding_dimension': 8192,
            'format': 'parquet_sharded'
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        shards = list(output_dir.glob("data-*.parquet"))
        size = sum(f.stat().st_size for f in shards) / (1024 * 1024)
        print(f"\n✅ Completed: {dataset_name}/{split}")
        print(f"   Total chunks: {len(shards)}")
        print(f"   Total size: {size:.2f} MB\n")
        return True
    
    print(f"\n❌ Failed: {dataset_name}/{split}\n")
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU parallel extraction (chunk-based)")
    parser.add_argument('--splits', nargs='+', required=True, 
                       help='Format: v1:chat v2:math llama-sft:safety')
    parser.add_argument('--num-gpus', type=int, default=8, 
                       help='Number of GPUs to use (default: 8)')
    parser.add_argument('--num-samples', type=int, default=10000, 
                       help='Samples per chunk (default: 10000, use -1 to auto-divide by num_gpus)')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for inference (default: 64)')
    parser.add_argument('--output', default='embeddings_output', 
                       help='Output directory (default: embeddings_output)')
    parser.add_argument('--model', default='nvidia/llama-embed-nemotron-8b', 
                       help='Model to use')
    parser.add_argument('--max-text-length', type=int, default=8192, 
                       help='Max text length (default: 8192)')
    parser.add_argument('--input-type', default='document', choices=['document', 'query'],
                       help='Input type (default: document)')
    
    args = parser.parse_args()
    
    config = {
        'model': args.model,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'output': args.output,
        'max_text_length': args.max_text_length,
        'input_type': args.input_type,
        'num_gpus': args.num_gpus
    }
    
    # Required for multiprocessing with CUDA
    mp.set_start_method('spawn', force=True)
    
    print("="*80)
    print("🚀 Multi-GPU Parallel Extraction (Chunk-Based)")
    print("="*80)
    print(f"🎮 GPUs: {args.num_gpus}")
    print(f"📦 Model: {args.model}")
    print(f"📏 Chunk size: {args.num_samples if args.num_samples > 0 else 'auto'} samples")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"💾 Output: {args.output}/")
    
    dataset_map = {
        'v1': 'nemotron-v1',
        'v2': 'nemotron-v2',
        'llama-sft': 'llama-nemotron-sft'
    }
    
    success = 0
    failed = 0
    
    for split_spec in args.splits:
        try:
            version, split = split_spec.split(':')
            
            if version not in dataset_map:
                print(f"\n❌ Unknown version: {version}")
                failed += 1
                continue
            
            if process_split(dataset_map[version], split, args.num_gpus, config):
                success += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\n❌ Error processing {split_spec}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print(f"✅ Successful: {success}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print("="*80)
