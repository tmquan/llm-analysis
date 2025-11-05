#!/usr/bin/env python3
"""
NVIDIA GPU Datasheet Downloader
================================

This script downloads NVIDIA GPU datasheets from various sources.
It handles different scenarios:
- Direct PDF downloads
- Landing pages with embedded PDF URLs
- Widen.net hosted files requiring URL extraction

Requirements:
    - requests: For HTTP requests
    - beautifulsoup4: For HTML parsing
    - rich: For beautiful progress bars and console output

Usage:
    python download_gpu_datasheets.py
"""

import os
import re
import json
import requests
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.table import Table


# Initialize rich console for beautiful output
console = Console()


# Output directory for downloaded PDFs
OUTPUT_DIR = Path("data/pdf")
OUTPUT_DIR.mkdir(exist_ok=True)


# GPU datasheet configuration
# Each entry contains the name, URL (direct or landing page), and output filename
GPU_DATASHEETS = {
    "A100_80GB": {
        "name": "A100 80GB (SXM/PCIe)",
        "url": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf",
        "filename": "nvidia-a100-datasheet.pdf"
    },
    "H100_80GB": {
        "name": "H100 80GB (SXM/NVL)",
        "url": "https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet",
        "filename": "nvidia-h100-datasheet.pdf"
    },
    "H200_141GB": {
        "name": "H200 141GB (SXM/NVL)",
        "url": "https://resources.nvidia.com/en-us-hopper-architecture/hpc-datasheet-sc23",
        "filename": "nvidia-h200-datasheet.pdf"
    },
    "GH200": {
        "name": "GH200 Grace Hopper Superchip",
        "url": "https://resources.nvidia.com/en-us-grace-cpu/grace-hopper-superchip",
        "filename": "nvidia-gh200-grace-hopper-datasheet.pdf"
    },
    "B200_B300": {
        "name": "B200/B300 Blackwell",
        "url": "https://resources.nvidia.com/en-us-blackwell-architecture",
        "filename": "nvidia-b200-b300-blackwell-datasheet.pdf"
    },
    "GB200_GB300": {
        "name": "GB200/GB300 Blackwell Ultra",
        "url": "https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet",
        "filename": "nvidia-gb200-gb300-blackwell-ultra-datasheet.pdf"
    }
}


def get_session() -> requests.Session:
    """
    Create a requests session with browser-like headers.
    
    This helps avoid being blocked by servers that check for bot traffic.
    The User-Agent mimics a modern web browser.
    
    Returns:
        Configured requests.Session object
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    return session


def extract_pdf_download_url_from_widen(html_content: bytes | str, widen_url: str) -> str | None:
    """
    Extract the actual PDF download URL from a Widen.net viewer page.
    
    Widen.net is a digital asset management platform used by NVIDIA.
    Their viewer pages contain the actual PDF URL embedded in the HTML.
    
    Args:
        html_content: HTML content from widen.net (bytes or string)
        widen_url: The widen.net URL for resolving relative paths
    
    Returns:
        Direct PDF download URL if found, None otherwise
    
    Strategy:
        1. First, look for download links (preferred - original quality)
        2. Fallback to viewerPdfUrl (preview quality)
    """
    # Ensure we're working with a string
    if isinstance(html_content, bytes):
        html_content = html_content.decode('utf-8', errors='ignore')
    
    # Method 1: Look for the download link pattern (preferred)
    # This gives us the original PDF with full quality
    # Example: /content/vuzumiozpb/original/h100-datasheet-2287922.pdf?u=9ikupj&use=mhpfe&download=true
    download_pattern = r'href="(/content/[^"]+\.pdf[^"]*download=true[^"]*)"'
    match = re.search(download_pattern, html_content)
    
    if match:
        relative_url = match.group(1)
        # Convert HTML entities (&amp; -> &)
        relative_url = relative_url.replace('&amp;', '&')
        
        # Construct full URL
        parsed = urlparse(widen_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        full_url = urljoin(base_url, relative_url)
        
        console.print(f"    [dim]→ Found download URL[/dim]")
        return full_url
    
    # Method 2: Look for viewerPdfUrl in JavaScript (fallback)
    # This is a preview version but still usable
    viewer_pattern = r"viewerPdfUrl\s*=\s*'([^']*)'"
    match = re.search(viewer_pattern, html_content)
    
    if match:
        viewer_url = match.group(1)
        console.print(f"    [dim]→ Found viewer PDF URL[/dim]")
        return viewer_url
    
    return None


def extract_pdf_url_from_html(html_content: bytes | str, base_url: str) -> str | None:
    """
    Extract PDF URL from NVIDIA resources landing page.
    
    NVIDIA's resources pages are dynamic and embed the PDF URL in various ways:
    - JSON data (window.__PATHFACTORY__)
    - iframe src attributes
    - data-source-url attributes
    
    Args:
        html_content: HTML content (bytes or string)
        base_url: Base URL for reference
    
    Returns:
        PDF or PDF viewer URL if found, None otherwise
    """
    # Ensure we're working with a string
    if isinstance(html_content, bytes):
        html_content = html_content.decode('utf-8', errors='ignore')
    
    # Method 1: Extract from embedded JSON (window.__PATHFACTORY__)
    # This is the most reliable method for NVIDIA resources pages
    json_pattern = r'window\.__PATHFACTORY__\s*=\s*({.*?});'
    match = re.search(json_pattern, html_content, re.DOTALL)
    
    if match:
        try:
            json_str = match.group(1)
            data = json.loads(json_str)
            
            # Check currentContent for sourceUrl
            if 'global' in data and 'currentContent' in data['global']:
                source_url = data['global']['currentContent'].get('sourceUrl')
                if source_url:
                    console.print(f"    [dim]→ Found URL in JSON (currentContent)[/dim]")
                    return source_url
            
            # Check experienceContent array for datasheet entries
            if 'global' in data and 'experienceContent' in data['global']:
                for content in data['global']['experienceContent']:
                    if 'sourceUrl' in content and 'datasheet' in content.get('title', '').lower():
                        source_url = content['sourceUrl']
                        console.print(f"    [dim]→ Found URL in JSON (experienceContent)[/dim]")
                        return source_url
        except json.JSONDecodeError as e:
            console.print(f"    [yellow]⚠ Could not parse JSON: {e}[/yellow]")
    
    # Method 2: Look for iframe with nvdam.widen.net
    soup = BeautifulSoup(html_content, 'html.parser')
    iframe = soup.find('iframe', src=re.compile(r'nvdam\.widen\.net'))
    if iframe and iframe.get('src'):
        console.print(f"    [dim]→ Found URL in iframe[/dim]")
        return iframe['src']
    
    # Method 3: Look for data-source-url attribute
    element = soup.find(attrs={'data-source-url': True})
    if element:
        source_url = element['data-source-url']
        console.print(f"    [dim]→ Found URL in data-source-url attribute[/dim]")
        return source_url
    
    return None


def download_pdf(url: str, output_path: Path, session: requests.Session) -> tuple[bool, float, str]:
    """
    Download a PDF file from a direct URL.
    
    Args:
        url: Direct URL to the PDF
        output_path: Path where the file should be saved
        session: Configured requests session
    
    Returns:
        Tuple of (success, file_size_mb, content_type_or_error)
    """
    try:
        response = session.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        # Verify it's a PDF-like content
        if 'pdf' in content_type or 'application/octet-stream' in content_type or 'widen.net' in url:
            # Verify PDF magic bytes (%PDF at the start)
            if response.content[:4] != b'%PDF':
                return False, 0, f"Not a PDF (starts with {response.content[:4]})"
            
            # Save the file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            return True, file_size_mb, content_type
        else:
            return False, 0, content_type
            
    except Exception as e:
        return False, 0, str(e)


def download_gpu_datasheet(
    url: str,
    output_path: Path,
    gpu_name: str,
    session: requests.Session
) -> tuple[bool, float]:
    """
    Download a GPU datasheet, handling various URL formats.
    
    This function implements a multi-step download strategy:
    1. Try direct PDF download
    2. If landing page, extract PDF URL
    3. If Widen.net, extract download URL from viewer page
    4. Download the final PDF
    
    Args:
        url: URL to download from (direct or landing page)
        output_path: Path to save the PDF
        gpu_name: Name of the GPU for logging
        session: Configured requests session
    
    Returns:
        Tuple of (success, file_size_mb)
    """
    try:
        console.print(f"  [cyan]Fetching:[/cyan] {gpu_name}")
        
        # Step 1: Fetch the URL
        response = session.get(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        # Case 1: Direct PDF download
        if 'pdf' in content_type or url.endswith('.pdf'):
            console.print(f"    [dim]→ Direct PDF detected[/dim]")
            
            # Verify PDF magic bytes
            if response.content[:4] != b'%PDF':
                console.print(f"    [red]✗ Invalid PDF file[/red]")
                return False, 0
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            console.print(f"    [green]✓ Downloaded ({file_size_mb:.2f} MB)[/green]")
            return True, file_size_mb
        
        # Case 2: Landing page - extract PDF URL
        elif 'html' in content_type:
            console.print(f"    [dim]→ Landing page detected, extracting PDF URL...[/dim]")
            
            pdf_url = extract_pdf_url_from_html(response.content, url)
            
            if not pdf_url:
                console.print(f"    [red]✗ Could not extract PDF URL[/red]")
                # Save HTML for debugging
                html_path = output_path.with_suffix('.html')
                with open(html_path, 'wb') as f:
                    f.write(response.content)
                console.print(f"    [dim]→ Saved HTML for inspection: {html_path}[/dim]")
                return False, 0
            
            # Case 2a: PDF URL is on Widen.net
            if 'widen.net' in pdf_url:
                console.print(f"    [dim]→ Widen.net URL detected, fetching viewer page...[/dim]")
                
                widen_response = session.get(pdf_url, timeout=30)
                widen_response.raise_for_status()
                
                # Extract the actual download URL
                download_url = extract_pdf_download_url_from_widen(widen_response.content, pdf_url)
                
                if not download_url:
                    console.print(f"    [red]✗ Could not extract download URL from Widen.net[/red]")
                    return False, 0
                
                # Download the PDF
                console.print(f"    [dim]→ Downloading from Widen.net...[/dim]")
                success, file_size_mb, result = download_pdf(download_url, output_path, session)
                
                if success:
                    console.print(f"    [green]✓ Downloaded ({file_size_mb:.2f} MB)[/green]")
                    return True, file_size_mb
                else:
                    console.print(f"    [red]✗ Download failed: {result}[/red]")
                    return False, 0
            
            # Case 2b: Direct PDF URL
            else:
                console.print(f"    [dim]→ Downloading from extracted URL...[/dim]")
                success, file_size_mb, result = download_pdf(pdf_url, output_path, session)
                
                if success:
                    console.print(f"    [green]✓ Downloaded ({file_size_mb:.2f} MB)[/green]")
                    return True, file_size_mb
                else:
                    console.print(f"    [red]✗ Download failed: {result}[/red]")
                    return False, 0
        
        else:
            console.print(f"    [red]✗ Unexpected content type: {content_type}[/red]")
            return False, 0
            
    except requests.exceptions.RequestException as e:
        console.print(f"    [red]✗ Network error: {e}[/red]")
        return False, 0
    except Exception as e:
        console.print(f"    [red]✗ Error: {e}[/red]")
        return False, 0


def download_all_datasheets() -> None:
    """
    Main function to download all GPU datasheets with progress tracking.
    """
    # Display header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]NVIDIA GPU Datasheet Downloader[/bold cyan]\n"
        "[dim]Downloads datasheets from NVIDIA resources[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    # Initialize session
    session = get_session()
    
    # Track results
    successful_downloads = []
    failed_downloads = []
    skipped_files = []
    total_size_mb = 0.0
    
    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        # Create overall task
        overall_task = progress.add_task(
            "[cyan]Downloading datasheets...",
            total=len(GPU_DATASHEETS)
        )
        
        # Process each GPU datasheet
        for gpu_id, info in GPU_DATASHEETS.items():
            output_path = OUTPUT_DIR / info["filename"]
            
            # Update progress description
            progress.update(
                overall_task,
                description=f"[cyan]Processing: {info['name']}"
            )
            
            # Skip if already downloaded
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                console.print(f"  [yellow]⊘ Skipping:[/yellow] {info['name']} [dim](already exists, {file_size_mb:.2f} MB)[/dim]")
                skipped_files.append((info['name'], file_size_mb))
                total_size_mb += file_size_mb
                progress.advance(overall_task)
                continue
            
            # Download the datasheet
            success, file_size_mb = download_gpu_datasheet(
                info["url"],
                output_path,
                info["name"],
                session
            )
            
            if success:
                successful_downloads.append((info['name'], file_size_mb))
                total_size_mb += file_size_mb
            else:
                failed_downloads.append((info['name'], info['url']))
            
            # Be respectful to the server
            progress.advance(overall_task)
            time.sleep(1)
    
    # Display summary table
    console.print()
    console.print("[bold]Summary[/bold]")
    
    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Status", style="dim", width=12)
    table.add_column("GPU", style="cyan")
    table.add_column("Size", justify="right", style="green")
    
    # Add successful downloads
    for name, size_mb in successful_downloads:
        table.add_row("✓ Downloaded", name, f"{size_mb:.2f} MB")
    
    # Add skipped files
    for name, size_mb in skipped_files:
        table.add_row("⊘ Skipped", name, f"{size_mb:.2f} MB")
    
    # Add failed downloads
    for name, url in failed_downloads:
        table.add_row("[red]✗ Failed[/red]", name, "-")
    
    console.print(table)
    console.print()
    
    # Print statistics
    console.print(f"[bold]Statistics:[/bold]")
    console.print(f"  • Total files: {len(GPU_DATASHEETS)}")
    console.print(f"  • [green]Downloaded: {len(successful_downloads)}[/green]")
    console.print(f"  • [yellow]Skipped: {len(skipped_files)}[/yellow]")
    console.print(f"  • [red]Failed: {len(failed_downloads)}[/red]")
    console.print(f"  • Total size: {total_size_mb:.2f} MB")
    console.print(f"  • Output directory: [cyan]{OUTPUT_DIR.absolute()}[/cyan]")
    
    # Show failure details if any
    if failed_downloads:
        console.print()
        console.print("[yellow]⚠ Failed downloads may require manual download from a web browser.[/yellow]")
        console.print("[yellow]  Landing pages may use JavaScript or require authentication.[/yellow]")
    
    console.print()


def main():
    """
    Entry point for the script.
    """
    download_all_datasheets()


if __name__ == "__main__":
    main()
