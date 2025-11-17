#!/usr/bin/env python3
"""
PDF to JSON Converter using NVIDIA Nemotron Parse API
======================================================

This script extracts structured content from PDF files using NVIDIA's Nemotron Parse API.
The extraction includes layout detection with bounding boxes, text extraction, and semantic
classification of document elements (titles, sections, tables, figures, etc.).

Requirements:
    - PyMuPDF (fitz): For rendering PDF pages to images
    - Pillow (PIL): For image manipulation
    - requests: For API calls to NVIDIA
    - pandas: For data manipulation and export
    - rich: For beautiful progress bars and console output
    - python-dotenv: For loading credentials from environment

Setup:
    1. Sign up for NVIDIA API: https://build.nvidia.com/
    2. Set environment variable:
       - NVIDIA_API_KEY: Your NVIDIA API key
    3. Install dependencies: pip install pymupdf pillow requests pandas rich python-dotenv

Usage:
    python pdf_to_json.py
    
    The script will:
    - Read PDFs from data/pdf directory
    - Extract layout and content using NVIDIA Nemotron Parse API
    - Save results to data/out/{pdf_name}/ directory:
        • page_{N}.json: Structured JSON with blocks and bounding boxes
        • page_{N}.png: Annotated image with layout visualization
        • page_{N}.tsv: Tabular data for easy analysis
"""

import os
import re
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from io import BytesIO

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from dotenv import load_dotenv


# ============================================================================
# Configuration
# ============================================================================

# Initialize rich console for beautiful output
console = Console()

# Color scheme for different layout element types
CLASS_COLORS = {
    "Title": "#D32F2F",           # Red
    "Section-header": "#E91E63",  # Pink
    "Text": "#4CAF50",            # Green
    "List-item": "#1976D2",       # Blue
    "Caption": "#607D8B",         # Blue Grey
    "Table": "#03A9F4",           # Light Blue
    "Figure": "#6D4C41",          # Brown
    "Picture": "#6D4C41",         # Brown
    "Formula": "#FF9800",         # Orange
    "Page-header": "#424242",     # Dark Grey
    "Page-footer": "#424242",     # Dark Grey  
    "Page-number": "#424242",     # Dark Grey
    "Footnote": "#00BCD4",        # Cyan
    "Biography": "#512DA8",       # Deep Purple
    "TOC": "#FFC107",             # Amber
    "DEFAULT": "#757575",         # Medium Grey
}


# ============================================================================
# Color Utilities
# ============================================================================

def get_text_color(hex_color: str) -> str:
    """
    Calculate optimal text color (black or white) based on background color.
    
    Uses the relative luminance formula from WCAG 2.0 to determine if
    a background color is light or dark, then returns an appropriate
    contrasting text color.
    
    Args:
        hex_color: Background color in hex format (e.g., "#FF5733")
    
    Returns:
        "#000000" for dark text or "#FFFFFF" for light text
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate relative luminance
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    
    # Return black for light backgrounds, white for dark backgrounds
    return "#000000" if luminance > 0.5 else "#FFFFFF"


# ============================================================================
# PDF to Image Conversion
# ============================================================================

def pdf_page_to_image(
    pdf_path: Path,
    page_index: int,
    dpi: int = 300,
    target_size: Tuple[int, int] = (1536, 2048),
) -> Image.Image:
    """
    Convert a PDF page to a high-resolution image with consistent dimensions.
    
    This function:
    1. Opens the PDF and selects the specified page
    2. Renders the page at the specified DPI
    3. Resizes to fit within target dimensions while preserving aspect ratio
    4. Centers the result on a white canvas
    
    Args:
        pdf_path: Path to the PDF file
        page_index: Zero-based page index
        dpi: Resolution for rendering (default: 300 for high quality)
        target_size: Canvas size as (width, height) in pixels
    
    Returns:
        PIL Image object with the rendered page
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    
    # Calculate zoom factor from DPI
    # PDF default is 72 DPI, so zoom = target_dpi / 72
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    # Render the page to a pixmap (raster image)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert pixmap to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    
    # Resize to fit target dimensions while preserving aspect ratio
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create a white canvas and center the image
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    x = (target_size[0] - img.width) // 2
    y = (target_size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    
    return canvas


# ============================================================================
# Image Encoding
# ============================================================================

def encode_image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a PIL Image to base64 string for API transmission.
    
    Args:
        image: PIL Image object
        fmt: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64-encoded string representation of the image
    """
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================================
# NVIDIA Nemotron Parse API Integration
# ============================================================================

def call_nemotron_parse(
    base64_image: str,
    api_key: str,
    api_url: str = "https://integrate.api.nvidia.com/v1"
) -> List[Dict[str, Any]]:
    """
    Call NVIDIA Nemotron Parse API to extract layout and content from image.
    
    This function sends a base64-encoded image to NVIDIA's Nemotron Parse model,
    which analyzes the document layout and returns structured blocks with:
    - Type classification (title, text, table, figure, etc.)
    - Bounding box coordinates (normalized 0-1)
    - Extracted text content
    
    Args:
        base64_image: Base64-encoded image string
        api_key: NVIDIA API key
        api_url: NVIDIA API base URL
    
    Returns:
        List of block dictionaries, each containing:
        - type: Element type (e.g., "Title", "Text", "Table")
        - bbox: Bounding box as {xmin, ymin, xmax, ymax}
        - text: Extracted text content
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    # Construct the message with the image
    message = [{
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
            }
        }]
    }]
    
    # Define the tool that the model should use
    tools = [{"type": "function", "function": {"name": "markdown_bbox"}}]
    
    # Prepare the API request payload
    payload = {
        "model": "nvidia/nemotron-parse",
        "messages": message,
        "max_tokens": 4096,
        "tools": tools,
        "temperature": 0.1,
        "top_p": 0.9,
        # "frequency_penalty": 0.0,
        # "presence_penalty": 0.0,
        # "stop": None,
        # "stream": False,
    }
    
    try:
        # Make the API request
        response = requests.post(
            f"{api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=180,  # 3 minutes timeout
        )
        response.raise_for_status()
        response_json = response.json()
        
        # Extract the tool call arguments which contain the parsed blocks
        tool_call = response_json.get("choices", [{}])[0] \
            .get("message", {}) \
            .get("tool_calls", [{}])[0]
        
        if not tool_call:
            return []
        
        arguments_str = tool_call.get("function", {}).get("arguments", "[]")
        
        try:
            parsed_args = json.loads(arguments_str)
            
            # Handle nested list structure
            if isinstance(parsed_args, list) and len(parsed_args) > 0 and \
               isinstance(parsed_args[0], list):
                return parsed_args[0]
            
            return parsed_args if isinstance(parsed_args, list) else []
        
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not decode JSON arguments[/yellow]")
            return []
    
    except requests.exceptions.HTTPError as e:
        console.print(f"[red]API HTTP Error: {e}[/red]")
        return []
    
    except requests.exceptions.Timeout:
        console.print("[red]API request timed out (>180s)[/red]")
        return []
    
    except Exception as e:
        console.print(f"[red]Unexpected API error: {e}[/red]")
        return []


# ============================================================================
# Layout Visualization
# ============================================================================

def draw_layout_annotations(
    image: Image.Image,
    blocks: List[Dict[str, Any]]
) -> Image.Image:
    """
    Draw bounding boxes and labels on an image to visualize layout detection.
    
    This creates an annotated version of the image where each detected block
    is highlighted with:
    - A colored bounding box (color based on element type)
    - A label showing the block index and type
    
    Args:
        image: Source PIL Image
        blocks: List of block dictionaries with bbox and type information
    
    Returns:
        New PIL Image with annotations drawn
    """
    # Create a copy to avoid modifying the original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    width, height = annotated.size
    
    # Calculate sizes based on image dimensions
    box_thickness = max(2, int(width / 600))
    font_size = max(10, int(width / 80))
    
    # Try to load a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    
    # Draw each block
    for i, block in enumerate(blocks):
        bbox = block.get("bbox", {})
        
        # Extract normalized coordinates (0-1 range)
        xmin = bbox.get("xmin", 0)
        ymin = bbox.get("ymin", 0)
        xmax = bbox.get("xmax", 0)
        ymax = bbox.get("ymax", 0)
        
        # Ensure coordinates are within valid range [0, 1] with small margin tolerance
        xmin = max(0, min(1, xmin))
        ymin = max(0, min(1, ymin))
        xmax = max(0, min(1, xmax))
        ymax = max(0, min(1, ymax))
        
        # Skip invalid bounding boxes after clamping
        # Check that coordinates form a proper rectangle with minimum size
        min_dimension = 0.001  # Minimum 0.1% of dimension
        if xmax <= xmin + min_dimension or ymax <= ymin + min_dimension:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        x0, y0 = xmin * width, ymin * height
        x1, y1 = xmax * width, ymax * height
        
        # Final safety check: ensure minimum pixel dimensions
        min_pixels = 1
        if x1 <= x0 + min_pixels or y1 <= y0 + min_pixels:
            continue
        
        # Get color for this element type
        element_type = block.get("type", "DEFAULT")
        color = CLASS_COLORS.get(element_type, CLASS_COLORS["DEFAULT"])
        text_color = get_text_color(color)
        
        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=box_thickness)
        
        # Prepare label
        label = f"{i}: {element_type}"
        
        # Calculate label dimensions
        try:
            # Newer Pillow versions
            tb = draw.textbbox((0, 0), label, font=font)
            text_width = tb[2] - tb[0]
            text_height = tb[3] - tb[1]
        except AttributeError:
            # Older Pillow versions
            text_width, text_height = draw.textsize(label, font=font)
        
        # Position label above the box if there's space, otherwise at top
        padding = 6
        if y0 > text_height + padding * 2:
            label_y_top = y0 - text_height - padding * 2
        else:
            label_y_top = 0
        
        label_y_bottom = label_y_top + text_height + padding
        
        # # Draw label background
        # label_bg = (x0, label_y_top, x0 + text_width + padding * 2, label_y_bottom)
        # draw.rectangle(label_bg, fill=color)
        
        # # Draw label text
        # draw.text((x0 + padding, label_y_top + padding // 2), label, 
        #           fill=text_color, font=font)
    
    return annotated


# ============================================================================
# Data Conversion Utilities
# ============================================================================

def blocks_to_dataframe(blocks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert blocks to a pandas DataFrame for easy analysis and export.
    
    Args:
        blocks: List of block dictionaries from API
    
    Returns:
        DataFrame with columns: id, type, text, bbox
    """
    rows = []
    for i, block in enumerate(blocks):
        # Truncate long text for readability
        text = (block.get("text") or "").replace("\n", " ")
        if len(text) > 200:
            text = text[:200] + "..."
        
        rows.append({
            "id": i,
            "type": block.get("type", "DEFAULT"),
            "text": text,
            "bbox": json.dumps(block.get("bbox", {})),
        })
    
    return pd.DataFrame(rows)


def blocks_to_detailed_records(
    blocks: List[Dict[str, Any]],
    page_number: int,
    page_size_px: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """
    Convert blocks to detailed records with both normalized and pixel coordinates.
    
    This format is useful for downstream processing and includes:
    - Page number for multi-page documents
    - Both normalized (0-1) and pixel coordinates
    - Full text content
    
    Args:
        blocks: List of block dictionaries from API
        page_number: 1-based page number
        page_size_px: Image dimensions as (width, height)
    
    Returns:
        List of enriched record dictionaries
    """
    width, height = page_size_px
    records = []
    
    for i, block in enumerate(blocks):
        bbox = block.get("bbox", {})
        
        # Extract normalized coordinates
        xmin = bbox.get("xmin", 0)
        ymin = bbox.get("ymin", 0)
        xmax = bbox.get("xmax", 0)
        ymax = bbox.get("ymax", 0)
        
        # Validate and clamp coordinates to [0, 1] range
        xmin = max(0, min(1, xmin))
        ymin = max(0, min(1, ymin))
        xmax = max(0, min(1, xmax))
        ymax = max(0, min(1, ymax))
        
        # Ensure proper rectangle (swap if needed)
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        
        # Convert to pixel coordinates
        x0 = int(xmin * width)
        y0 = int(ymin * height)
        x1 = int(xmax * width)
        y1 = int(ymax * height)
        
        records.append({
            "page": page_number,
            "block_id": i,
            "type": block.get("type", "DEFAULT"),
            "text": block.get("text", ""),
            "bbox_normalized": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            },
            "bbox_pixels": {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            },
            "is_valid": xmax > xmin and ymax > ymin,
        })
    
    return records


# ============================================================================
# PDF Processing Pipeline
# ============================================================================

def process_pdf_page(
    pdf_path: Path,
    page_index: int,
    output_page_dir: Path,
    api_key: str,
    dpi: int = 120,
    target_size: Tuple[int, int] = (1536, 2048),
) -> Tuple[bool, Optional[int]]:
    """
    Process a single PDF page: extract layout, create visualizations, save outputs.
    
    This is the main processing pipeline for one page:
    1. Convert PDF page to image
    2. Send to NVIDIA API for layout extraction
    3. Create annotated visualization
    4. Save JSON, PNG, and TSV outputs
    
    Args:
        pdf_path: Path to the PDF file
        page_index: Zero-based page index
        output_page_dir: Directory to save page outputs
        api_key: NVIDIA API key
        dpi: Image rendering resolution
        target_size: Target image dimensions
    
    Returns:
        Tuple of (success: bool, num_blocks: int or None)
    """
    page_number = page_index + 1  # 1-based for user-facing output
    
    try:
        # Step 1: Convert PDF page to image
        image = pdf_page_to_image(pdf_path, page_index, dpi=dpi, target_size=target_size)
        
        # Step 2: Encode image for API
        base64_image = encode_image_to_base64(image, fmt="PNG")
        
        # Step 3: Call NVIDIA API to extract layout
        blocks = call_nemotron_parse(base64_image, api_key)
        
        if not blocks:
            console.print(f"[yellow]Warning: No blocks found for page {page_number}[/yellow]")
            return False, 0
        
        # Step 4: Create annotated visualization
        annotated_image = draw_layout_annotations(image, blocks)
        
        # Step 5: Convert to different formats
        df = blocks_to_dataframe(blocks)
        records = blocks_to_detailed_records(blocks, page_number, image.size)
        
        # Step 6: Save all outputs
        output_page_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON (detailed records)
        json_path = output_page_dir / f"page_{page_number}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        # Save PNG (annotated image)
        png_path = output_page_dir / f"page_{page_number}.png"
        annotated_image.save(png_path, "PNG")
        
        # # Save TSV (tabular data)
        # tsv_path = output_page_dir / f"page_{page_number}.tsv"
        # df.to_csv(tsv_path, sep='\t', index=False)
        
        return True, len(blocks)
    
    except Exception as e:
        console.print(f"[red]Error processing page {page_number}: {e}[/red]")
        return False, None


def process_single_pdf(
    pdf_path: Path,
    output_base_dir: Path,
    api_key: str,
    progress: Progress,
    task_id: int,
) -> Dict[str, Any]:
    """
    Process all pages of a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base output directory
        api_key: NVIDIA API key
        progress: Rich progress bar instance
        task_id: Progress bar task ID
    
    Returns:
        Dictionary with processing statistics
    """
    pdf_name = pdf_path.stem
    output_pdf_dir = output_base_dir / pdf_name
    
    # Get total number of pages
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        console.print(f"[red]Error opening {pdf_path.name}: {e}[/red]")
        return {
            "pdf_name": pdf_name,
            "success": False,
            "pages_processed": 0,
            "pages_total": 0,
            "total_blocks": 0,
        }
    
    # Process each page
    pages_processed = 0
    total_blocks = 0
    
    for page_idx in range(num_pages):
        progress.update(
            task_id,
            description=f"[cyan]Processing: {pdf_path.name} (page {page_idx + 1}/{num_pages})"
        )
        
        success, num_blocks = process_pdf_page(
            pdf_path,
            page_idx,
            output_pdf_dir,
            api_key,
        )
        
        if success and num_blocks is not None:
            pages_processed += 1
            total_blocks += num_blocks
    
    return {
        "pdf_name": pdf_name,
        "success": pages_processed == num_pages,
        "pages_processed": pages_processed,
        "pages_total": num_pages,
        "total_blocks": total_blocks,
    }


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main entry point: Process all PDFs in the input directory.
    
    This orchestrates the entire pipeline:
    1. Load API credentials
    2. Find all PDF files
    3. Process each PDF with progress tracking
    4. Display summary statistics
    """
    # Display header
    console.print("\n[bold magenta]PDF to JSON Converter[/bold magenta]")
    console.print("[dim]Using NVIDIA Nemotron Parse API[/dim]\n")
    
    # Step 1: Load API credentials
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        console.print(
            "[red]Error: NVIDIA API key not found![/red]\n"
            "Please set the NVIDIA_API_KEY environment variable.\n"
            "You can set it in a .env file or as a system environment variable.\n\n"
            "Get your API key at: https://build.nvidia.com/"
        )
        return
    
    console.print("[green]✓[/green] API key loaded successfully\n")
    
    # Step 2: Setup directories
    input_dir = Path("data/pdf")
    output_dir = Path("data/json")
    
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory '{input_dir}' does not exist![/red]")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Output directory: {output_dir}\n")
    
    # Step 3: Find all PDF files
    pdf_files = sorted(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
        return
    
    console.print(f"[cyan]Found {len(pdf_files)} PDF file(s)[/cyan]\n")
    
    # Step 4: Process all PDFs with progress tracking
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        overall_task = progress.add_task(
            "[cyan]Processing PDFs...",
            total=len(pdf_files)
        )
        
        for pdf_path in pdf_files:
            result = process_single_pdf(
                pdf_path,
                output_dir,
                api_key,
                progress,
                overall_task,
            )
            results.append(result)
            progress.advance(overall_task)
    
    # Step 5: Display summary
    console.print("\n" + "="*70)
    console.print("[bold green]✓ Processing Complete![/bold green]\n")
    
    total_pages = sum(r["pages_total"] for r in results)
    total_processed = sum(r["pages_processed"] for r in results)
    total_blocks = sum(r["total_blocks"] for r in results)
    successful_pdfs = sum(1 for r in results if r["success"])
    
    console.print(f"  • PDFs processed: {successful_pdfs}/{len(pdf_files)}")
    console.print(f"  • Pages processed: {total_processed}/{total_pages}")
    console.print(f"  • Total blocks extracted: {total_blocks}")
    console.print(f"  • Output directory: {output_dir}\n")
    
    # Show per-PDF summary
    if len(results) > 1:
        console.print("[bold]Per-PDF Summary:[/bold]")
        for result in results:
            status = "✓" if result["success"] else "✗"
            color = "green" if result["success"] else "red"
            console.print(
                f"  [{color}]{status}[/{color}] {result['pdf_name']}: "
                f"{result['pages_processed']}/{result['pages_total']} pages, "
                f"{result['total_blocks']} blocks"
            )
        console.print()
    
    console.print("="*70)


if __name__ == "__main__":
    main()

