#!/usr/bin/env python3
"""
PDF to XML Converter using Adobe PDF Services API
==================================================

This script extracts structured content from PDF files using Adobe's PDF Services API.
The extraction includes text, tables, and structural information in XML/JSON format.

Requirements:
    - pdfservices-sdk: Adobe's official PDF Services SDK (Python 3.10+ required)
    - rich: For beautiful progress bars and console output
    - python-dotenv: For loading credentials from environment

Setup:
    1. Sign up for Adobe PDF Services API: https://developer.adobe.com/document-services/
    2. Create a credentials file or set environment variables:
       - ADOBE_CLIENT_ID: Your Adobe API client ID
       - ADOBE_CLIENT_SECRET: Your Adobe API client secret
    3. Install dependencies: pip install pdfservices-sdk rich python-dotenv
    
Note:
    - Requires Python 3.10 or higher
    - Package name: pdfservices-sdk (imports as: from adobe.pdfservices.operation...)

Usage:
    python pdf_to_xml.py
    
    The script will:
    - Read PDFs from data/pdf directory
    - Extract structured content using Adobe API
    - Save results to data/xml directory
"""

import os
import time
import zipfile
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

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


# Initialize rich console for beautiful output
console = Console()


def load_adobe_credentials() -> Optional[ServicePrincipalCredentials]:
    """
    Load Adobe PDF Services API credentials from environment variables.
    
    This function looks for two environment variables:
    - ADOBE_CLIENT_ID: Your Adobe API client ID
    - ADOBE_CLIENT_SECRET: Your Adobe API client secret
    
    These can be set in a .env file or as system environment variables.
    
    Returns:
        ServicePrincipalCredentials object if credentials are found, None otherwise
    """
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    client_id = os.getenv("ADOBE_CLIENT_ID")
    client_secret = os.getenv("ADOBE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        console.print(
            "[red]Error: Adobe credentials not found![/red]\n"
            "Please set the following environment variables:\n"
            "  • ADOBE_CLIENT_ID\n"
            "  • ADOBE_CLIENT_SECRET\n\n"
            "You can set them in a .env file or as system environment variables."
        )
        return None
    
    # Create and return credentials object
    credentials = ServicePrincipalCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    
    return credentials


def extract_pdf_content(
    pdf_path: Path,
    output_dir: Path,
    credentials: ServicePrincipalCredentials
) -> Tuple[bool, Optional[Dict]]:
    """
    Extract structured content from a PDF using Adobe PDF Services API.
    
    This function performs the following steps:
    1. Creates a PDF Services client with your credentials
    2. Uploads the PDF file to Adobe's servers
    3. Requests extraction of text, tables, and structure
    4. Downloads the results (a ZIP file)
    5. Extracts and saves the JSON content
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory where extracted content will be saved
        credentials: Adobe API credentials
    
    Returns:
        Tuple of (success: bool, extracted_data: dict or None)
        
    The output includes:
    - {pdf_basename}.json: Structured JSON with all extracted content
    - {pdf_basename}.zip: Original ZIP archive from Adobe (optional)
    """
    try:
        # Step 1: Create PDF Services instance with credentials
        pdf_services = PDFServices(credentials=credentials)
        
        # Step 2: Read the input PDF file
        with open(pdf_path, 'rb') as file:
            input_stream = file.read()
        
        # Step 3: Upload the PDF to Adobe's servers
        # This creates a "StreamAsset" that Adobe can process
        input_asset = pdf_services.upload(
            input_stream=input_stream,
            mime_type=PDFServicesMediaType.PDF
        )
        
        # Step 4: Configure extraction parameters
        # We're requesting extraction of:
        # - TEXT: All text content with positioning
        # - TABLES: Table structures and data
        extract_pdf_params = ExtractPDFParams(
            elements_to_extract=[
                ExtractElementType.TEXT,
                ExtractElementType.TABLES
            ]
        )
        
        # Step 5: Create and submit the extraction job
        extract_pdf_job = ExtractPDFJob(
            input_asset=input_asset,
            extract_pdf_params=extract_pdf_params
        )
        
        # Get the job's location URL for polling
        location = pdf_services.submit(extract_pdf_job)
        
        # Step 6: Poll for job completion
        # Adobe processes the PDF asynchronously, so we need to wait
        pdf_services_response = pdf_services.get_job_result(
            location,
            ExtractPDFResult
        )
        
        # Step 7: Get the result asset
        result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
        
        # Step 8: Download the result
        # Adobe returns a ZIP file containing:
        # - structuredData.json: The extracted content
        # - Other files depending on extraction options
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        
        # Step 9: Save and extract the ZIP file
        pdf_basename = pdf_path.stem
        
        # Save the ZIP file temporarily
        zip_path = output_dir / f"{pdf_basename}.zip"
        with open(zip_path, "wb") as zip_file:
            zip_file.write(stream_asset.get_input_stream())
        
        # Step 10: Extract the JSON content from the ZIP
        extracted_data = None
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Look for the structured data JSON file
            if 'structuredData.json' in zip_ref.namelist():
                json_content = zip_ref.read('structuredData.json')
                extracted_data = json.loads(json_content)
                
                # Save the JSON separately for easier access
                json_path = output_dir / f"{pdf_basename}.json"
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(extracted_data, json_file, indent=2, ensure_ascii=False)
        
        # Optional: Remove the ZIP file to save space
        # Uncomment the next line if you don't need to keep the ZIP
        # zip_path.unlink()
        
        return True, extracted_data
    
    except ServiceApiException as e:
        console.print(f"[red]Adobe API Error for {pdf_path.name}: {e}[/red]")
        return False, None
    
    except ServiceUsageException as e:
        console.print(f"[red]Usage Error for {pdf_path.name}: {e}[/red]")
        console.print("[yellow]Check your Adobe API quota/limits[/yellow]")
        return False, None
    
    except SdkException as e:
        console.print(f"[red]SDK Error for {pdf_path.name}: {e}[/red]")
        return False, None
    
    except Exception as e:
        console.print(f"[red]Unexpected error processing {pdf_path.name}: {e}[/red]")
        return False, None


def display_extraction_info(data: Dict) -> None:
    """
    Display a summary of extracted content (for demonstration purposes).
    
    This function shows what kind of information Adobe's API extracted,
    helping you understand the structure of the results.
    
    Args:
        data: The extracted data dictionary from Adobe API
    """
    if not data:
        return
    
    # Count elements
    elements = data.get('elements', [])
    text_elements = [e for e in elements if e.get('Path', '').endswith('P')]
    table_elements = [e for e in elements if e.get('Path', '').endswith('Table')]
    
    console.print(Panel(
        f"[cyan]Total Elements:[/cyan] {len(elements)}\n"
        f"[cyan]Text Elements:[/cyan] {len(text_elements)}\n"
        f"[cyan]Tables:[/cyan] {len(table_elements)}",
        title="Extraction Summary",
        border_style="blue"
    ))


def main():
    """
    Main function: Extract content from all PDF files in the input directory.
    
    This is the main entry point that orchestrates the entire extraction process:
    1. Displays welcome header
    2. Loads Adobe API credentials
    3. Validates input directory
    4. Creates output directory
    5. Processes all PDFs with a progress bar
    6. Displays summary
    """
    # Display header
    console.print("\n[bold magenta]PDF to XML/JSON Extractor[/bold magenta]")
    console.print("[dim]Using Adobe PDF Services API[/dim]\n")
    
    # Step 1: Load credentials
    console.print("[cyan]Loading Adobe API credentials...[/cyan]")
    credentials = load_adobe_credentials()
    
    if not credentials:
        return
    
    console.print("[green]✓[/green] Credentials loaded successfully\n")
    
    # Step 2: Define input and output directories
    input_dir = Path("data/pdf")
    output_dir = Path("data/xml")
    
    # Step 3: Ensure input directory exists
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
    
    # Step 4: Setup progress bar and process all PDFs
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
            "[cyan]Extracting PDFs...",
            total=len(pdf_files)
        )
        
        successful_extractions = 0
        failed_extractions = 0
        
        # Process each PDF file
        for pdf_path in pdf_files:
            # Update progress description to show current file
            progress.update(
                overall_task,
                description=f"[cyan]Extracting: {pdf_path.name}"
            )
            
            # Extract the PDF content
            success, data = extract_pdf_content(pdf_path, output_dir, credentials)
            
            if success:
                successful_extractions += 1
            else:
                failed_extractions += 1
            
            # Update progress
            progress.advance(overall_task)
            
            # Small delay to respect API rate limits
            # Adjust this based on your Adobe API plan
            time.sleep(0.5)
    
    # Step 5: Print summary
    console.print("\n" + "="*60)
    console.print(f"[bold green]✓ Extraction Complete![/bold green]")
    console.print(f"  • PDFs processed: {len(pdf_files)}")
    console.print(f"  • Successful: {successful_extractions}")
    if failed_extractions > 0:
        console.print(f"  • Failed: [red]{failed_extractions}[/red]")
    console.print(f"  • Output directory: {output_dir}")
    console.print("="*60)


if __name__ == "__main__":
    main()

