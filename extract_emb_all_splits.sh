#!/bin/bash
# Script to extract embeddings from all Nemotron dataset splits
# Each split is saved to a separate JSON file
#
# Model: nvidia/llama-3.2-nemoretriever-300m-embed-v2
# Embedding Dimension: 2048
#
# Usage:
#   ./extract_all_splits.sh [--cloud|--local]
#
# Examples:
#   ./extract_all_splits.sh --cloud              # Use cloud API
#   ./extract_all_splits.sh --local              # Use local NIM
#   ./extract_all_splits.sh                      # Default: local

set -e  # Exit on error

# Parse mode argument
MODE_FLAG="${1:---local}"  # Default to --local if no argument

echo "=========================================="
echo "🚀 Extracting All Nemotron Dataset Splits"
echo "=========================================="
echo ""
echo "Model: nvidia/llama-3.2-nemoretriever-300m-embed-v2"
echo "Embedding Dimension: 2048"
echo "Mode: $MODE_FLAG"
echo "Output directory: current directory"
echo ""

# Default settings
NUM_SAMPLES=-1  # Adjust as needed
BATCH_SIZE=1024
INPUT_TYPE="passage"
MAX_TEXT_LENGTH=8192

# Nemotron v1 splits
echo "📦 Processing Nemotron v1 splits..."
splits_v1=("chat" "code" "math" "stem" "tool")
for split in "${splits_v1[@]}"; do
    output_file="/raid/Nemotron-Post-Training-Dataset-v1-${split}.jsonl"
    echo ""
    echo "  → Extracting v1-${split} to ${output_file}"
    python extract_emb.py \
        --v1-${split} \
        $MODE_FLAG \
        --num-samples $NUM_SAMPLES \
        --batch-size $BATCH_SIZE \
        --input-type $INPUT_TYPE \
        --max-text-length $MAX_TEXT_LENGTH \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully saved to ${output_file}"
    else
        echo "  ❌ Failed to extract v1-${split}"
    fi
done

# Nemotron v2 splits
echo ""
echo "📦 Processing Nemotron v2 splits..."
splits_v2=("chat" "code" "math" "stem" "tool")
for split in "${splits_v2[@]}"; do
    output_file="/raid/Nemotron-Post-Training-Dataset-v2-${split}.jsonl"
    echo ""
    echo "  → Extracting v2-${split} to ${output_file}"
    python extract_emb.py \
        --v2-${split} \
        $MODE_FLAG \
        --num-samples $NUM_SAMPLES \
        --batch-size $BATCH_SIZE \
        --input-type $INPUT_TYPE \
        --max-text-length $MAX_TEXT_LENGTH \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully saved to ${output_file}"
    else
        echo "  ❌ Failed to extract v2-${split}"
    fi
done

# Llama-Nemotron SFT splits
echo ""
echo "📦 Processing Llama-Nemotron SFT splits..."
splits_llama=("math" "code" "science" "chat" "safety")
for split in "${splits_llama[@]}"; do
    output_file="/raid/Nemotron-Post-Training-Dataset-llama-sft-${split}.jsonl"
    echo ""
    echo "  → Extracting llama-sft-${split} to ${output_file}"
    python extract_emb.py \
        --llama-sft-${split} \
        $MODE_FLAG \
        --num-samples $NUM_SAMPLES \
        --batch-size $BATCH_SIZE \
        --input-type $INPUT_TYPE \
        --max-text-length $MAX_TEXT_LENGTH \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully saved to ${output_file}"
    else
        echo "  ❌ Failed to extract llama-sft-${split}"
    fi
done

echo ""
echo "=========================================="
echo "✅ All extractions completed!"
echo "=========================================="
echo ""
echo "📊 Generated files:"
ls -lh Nemotron-Post-Training-Dataset-*.jsonl 2>/dev/null || echo "  (No files found - all extractions may have failed)"
echo ""
echo "💡 Total size:"
du -ch Nemotron-Post-Training-Dataset-*.jsonl 2>/dev/null | tail -1 || echo "  0B"
echo ""

