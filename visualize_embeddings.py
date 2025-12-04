#!/usr/bin/env python3
"""
UMAP Visualization for NVIDIA NIM Embeddings

This script creates beautiful UMAP visualizations from the embeddings
extracted by extract_emb.py.

Usage:
    python visualize_embeddings.py embeddings.json
    python visualize_embeddings.py embeddings.json --output umap_plot.png
"""

import json
import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("❌ NumPy not found. Install with: pip install numpy")
    exit(1)

try:
    import umap
except ImportError:
    print("❌ UMAP not found. Install with: pip install umap-learn")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("❌ Matplotlib not found. Install with: pip install matplotlib")
    exit(1)


def load_embeddings(json_path):
    """Load embeddings from JSON file."""
    print(f"📂 Loading embeddings from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract embeddings and metadata
    embeddings = []
    labels = []
    texts = []
    datasets = []
    
    for item in data['embeddings']:
        if item['embedding'] is not None:
            embeddings.append(item['embedding'])
            labels.append(item['split'])
            texts.append(item['text'][:100])  # Truncate for hover
            datasets.append(item['dataset'])
    
    print(f"✅ Loaded {len(embeddings)} embeddings")
    print(f"   Dimension: {len(embeddings[0])}")
    print(f"   Splits: {set(labels)}")
    print(f"   Datasets: {set(datasets)}")
    
    return np.array(embeddings), labels, texts, datasets, data['metadata']


def create_umap_visualization(embeddings, labels, texts, datasets, metadata, output_path=None, n_neighbors=15, min_dist=0.1):
    """Create UMAP visualization."""
    print(f"\n🔮 Running UMAP dimensionality reduction...")
    print(f"   n_neighbors: {n_neighbors}, min_dist: {min_dist}")
    
    # Run UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    print(f"✅ UMAP completed!")
    
    # Create color map
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each label separately for legend
    for label in unique_labels:
        mask = np.array(labels) == label
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[label_to_color[label]],
            label=label,
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Styling
    ax.set_title(f'UMAP Projection of {metadata["model"]} Embeddings\n'
                 f'{metadata["total_samples"]} samples from {metadata["datasets_processed"]} datasets',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(title='Dataset Splits', loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Add metadata text
    info_text = (f"Model: {metadata['model']}\n"
                 f"Embedding dim: {metadata['embedding_dimension']}\n"
                 f"Extraction: {metadata['extraction_time']}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved visualization to: {output_path}")
    else:
        print(f"\n👁️  Displaying plot...")
        plt.show()
    
    return embedding_2d


def create_interactive_html(embeddings_2d, labels, texts, datasets, metadata, output_path):
    """Create interactive HTML visualization with Plotly."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print("⚠️  Plotly not found. Install with: pip install plotly")
        print("   Skipping interactive visualization")
        return
    
    print(f"\n🌐 Creating interactive HTML visualization...")
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'split': labels,
        'dataset': datasets,
        'text': texts
    })
    
    # Create interactive plot
    fig = px.scatter(
        df, x='x', y='y', color='split',
        hover_data=['text', 'dataset'],
        title=f'Interactive UMAP: {metadata["model"]} Embeddings ({metadata["total_samples"]} samples)',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        width=1200, height=800
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig.update_layout(
        plot_bgcolor='#f8f9fa',
        font=dict(size=12),
        legend=dict(title='Dataset Splits', font=dict(size=10))
    )
    
    fig.write_html(output_path)
    print(f"✅ Saved interactive visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize embeddings with UMAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  python visualize_embeddings.py embeddings.json
  
  # Save to file
  python visualize_embeddings.py embeddings.json --output umap_plot.png
  
  # Create interactive HTML
  python visualize_embeddings.py embeddings.json --interactive umap.html
  
  # Adjust UMAP parameters
  python visualize_embeddings.py embeddings.json --n-neighbors 30 --min-dist 0.05
        """
    )
    
    parser.add_argument('input', help='Input JSON file with embeddings')
    parser.add_argument('--output', '-o', help='Output image file (PNG, PDF, SVG)')
    parser.add_argument('--interactive', help='Output interactive HTML file')
    parser.add_argument('--n-neighbors', type=int, default=15, help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter (default: 0.1)')
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        exit(1)
    
    print("=" * 80)
    print("🎨 UMAP Embedding Visualization")
    print("=" * 80)
    
    # Load data
    embeddings, labels, texts, datasets, metadata = load_embeddings(args.input)
    
    # Create UMAP visualization
    embeddings_2d = create_umap_visualization(
        embeddings, labels, texts, datasets, metadata,
        output_path=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
    
    # Create interactive HTML if requested
    if args.interactive:
        create_interactive_html(embeddings_2d, labels, texts, datasets, metadata, args.interactive)
    
    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

