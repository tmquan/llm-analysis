#!/usr/bin/env python3
"""
PDF to PNG Converter
====================

This script converts all PDF files in a source directory to PNG images.
Each page of each PDF is saved as a separate PNG file.

Requirements:
    - pymupdf (fitz): For PDF rendering
    - rich: For beautiful progress bars and console output

Usage:
    python pdf_to_png.py
"""

import fitz  # PyMuPDF
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


# Initialize rich console for beautiful output
console = Console()


def convert_pdf_to_png(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300
) -> int:
    """
    Convert a single PDF file to PNG images (one per page).
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory where PNG files will be saved
        dpi: Resolution for the output images (default: 300)
    
    Returns:
        Number of pages successfully converted
    
    The output files are named: {pdf_basename}_page_{page_number}.png
    """
    try:
        # Open the PDF document
        pdf_document = fitz.open(pdf_path)
        pdf_basename = pdf_path.stem  # Get filename without extension
        
        # Get total number of pages
        total_pages = len(pdf_document)
        
        # Convert each page to PNG
        for page_num in range(total_pages):
            # Get the page
            page = pdf_document[page_num]
            
            # Calculate zoom factor for desired DPI
            # Standard PDF resolution is 72 DPI
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to an image (pixmap)
            pix = page.get_pixmap(matrix=mat)
            
            # Construct output filename
            output_filename = f"{pdf_basename}_page_{page_num + 1:03d}.png"
            output_path = output_dir / output_filename
            
            # Save the image
            pix.save(str(output_path))
        
        # Close the document
        pdf_document.close()
        
        return total_pages
    
    except Exception as e:
        console.print(f"[red]Error processing {pdf_path.name}: {e}[/red]")
        return 0


def main():
    """
    Main function: Convert all PDF files in the input directory to PNG images.
    
    This is the main entry point that orchestrates the entire conversion process:
    1. Displays welcome header
    2. Validates input directory
    3. Creates output directory
    4. Processes all PDFs with a progress bar
    5. Displays summary
    """
    # Define input and output directories
    input_dir = Path("data/pdf")
    output_dir = Path("data/png")
    
    # DPI setting: Higher values = better quality but larger file sizes
    # 300 DPI is standard for high-quality prints
    # 150 DPI is often sufficient for screen viewing
    dpi = 300
    
    # Display header
    console.print("\n[bold magenta]PDF to PNG Converter[/bold magenta]")
    console.print("[dim]Using PyMuPDF (fitz) for rendering[/dim]\n")
    
    # Ensure input directory exists
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory '{input_dir}' does not exist![/red]")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Output directory: {output_dir}")
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {input_dir}[/yellow]")
        return
    
    console.print(f"[cyan]Found {len(pdf_files)} PDF file(s)[/cyan]\n")
    
    # Setup progress bar with multiple columns for detailed tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        # Create overall task for tracking all PDFs
        overall_task = progress.add_task(
            "[cyan]Converting PDFs...",
            total=len(pdf_files)
        )
        
        total_pages_converted = 0
        
        # Process each PDF file
        for pdf_path in pdf_files:
            # Update progress description to show current file
            progress.update(
                overall_task,
                description=f"[cyan]Converting: {pdf_path.name}"
            )
            
            # Convert the PDF
            pages_converted = convert_pdf_to_png(pdf_path, output_dir, dpi)
            total_pages_converted += pages_converted
            
            # Update progress
            progress.advance(overall_task)
    
    # Print summary
    console.print("\n" + "="*60)
    console.print(f"[bold green]✓ Conversion Complete![/bold green]")
    console.print(f"  • PDFs processed: {len(pdf_files)}")
    console.print(f"  • Total pages converted: {total_pages_converted}")
    console.print(f"  • Output directory: {output_dir}")
    console.print("="*60)


if __name__ == "__main__":
    main()

