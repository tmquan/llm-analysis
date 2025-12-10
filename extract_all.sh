#!/bin/bash
# Extract all splits using multi-GPU parallel strategy (chunk-based)

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nemotron

# Set HuggingFace cache directories
# - datasets folder: for HuggingFace datasets
# - checkpoints folder: for HuggingFace models/hub
export HF_HOME="$(pwd)/checkpoints"
export HUGGINGFACE_HUB_CACHE="$(pwd)/checkpoints"
export TRANSFORMERS_CACHE="$(pwd)/checkpoints"
export HF_DATASETS_CACHE="$(pwd)/datasets"

NUM_GPUS=${1:-8}
CHUNK_SIZE=${2:-1000000}  # Samples per chunk

echo "🚀 Multi-GPU Extraction (Chunk-Based)"
echo "Using $NUM_GPUS GPUs"
echo "Chunk size: $CHUNK_SIZE samples per chunk"
echo "📁 Datasets cache: $HF_DATASETS_CACHE"
echo "🤖 Models cache: $HF_HOME"
echo ""

# All 14 splits
SPLITS=(
    "v1:chat" "v1:code" "v1:math" "v1:stem" "v1:tool" 
    "v2:chat" "v2:code" "v2:math" "v2:stem"
    "llama-sft:math" "llama-sft:code" "llama-sft:science" "llama-sft:chat" "llama-sft:safety"
)

# Process each split
for split in "${SPLITS[@]}"; do
    echo "Processing: $split"
    
    python extract_parallel.py \
        --splits "$split" \
        --num-gpus $NUM_GPUS \
        --batch-size 64 \
        --num-samples $CHUNK_SIZE \
        --output embeddings_output
    
    [ $? -eq 0 ] && echo "✅ $split" || echo "❌ $split"
    echo ""
done

echo "✅ All done!"
echo ""
echo "Results:"
ls -lh embeddings_output/*/

