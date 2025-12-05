#!/usr/bin/env python3
"""
NVIDIA NIM Deployment Manager

This script manages the lifecycle of NVIDIA NIM Docker containers.

Usage:
    python docker_nim.py start [--api-key KEY] [--cache-dir PATH] [--port PORT]
    python docker_nim.py stop
    python docker_nim.py restart [--api-key KEY]
    python docker_nim.py status
    python docker_nim.py logs [-f]
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


CONTAINER_NAME = "nvidia-nim"
DEFAULT_IMAGE = "nvcr.io/nvidia/nemo-microservices/llama-3.2-nemoretriever-1b-vlm-embed-v1:latest"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/nim")
DEFAULT_PORT = 8000


class NIMDeployer:
    """Manages NVIDIA NIM Docker container lifecycle."""
    
    def __init__(self, api_key=None, cache_dir=DEFAULT_CACHE_DIR, port=DEFAULT_PORT, image=DEFAULT_IMAGE):
        self.api_key = api_key or os.getenv("NGC_API_KEY")
        self.cache_dir = Path(cache_dir).expanduser()
        self.port = port
        self.image = image
        self.container_name = CONTAINER_NAME
        
    def _run_command(self, cmd, capture_output=True, check=False):
        """Run a shell command and return the result."""
        try:
            if capture_output:
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    check=check
                )
                return result
            else:
                return subprocess.run(cmd, shell=True, check=check)
        except subprocess.CalledProcessError as e:
            return e
    
    def _get_container_id(self):
        """Get the container ID if running."""
        result = self._run_command(
            f"docker ps -q -f name={self.container_name}"
        )
        return result.stdout.strip() if result.returncode == 0 else None
    
    def _get_container_status(self):
        """Get container status (running, exited, or not found)."""
        result = self._run_command(
            f"docker ps -a -f name={self.container_name} --format '{{{{.Status}}}}'"
        )
        return result.stdout.strip() if result.returncode == 0 else None
    
    def status(self):
        """Check the status of the NIM container."""
        print("=" * 80)
        print("🔍 Checking NVIDIA NIM Container Status")
        print("=" * 80)
        
        container_id = self._get_container_id()
        status = self._get_container_status()
        
        if not status:
            print(f"\n❌ Container '{self.container_name}' not found")
            print(f"   Use 'docker_nim.py start' to create and start the container\n")
            return False
        
        if container_id:
            print(f"\n✅ Container '{self.container_name}' is RUNNING")
            print(f"   Container ID: {container_id}")
            print(f"   Status: {status}")
            print(f"   API Endpoint: http://localhost:{self.port}/v1/embeddings")
            
            # Test if API is responsive
            test_result = self._run_command(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{self.port}/v1/embeddings -X POST -H 'Content-Type: application/json' -d '{{}}' --max-time 2",
                check=False
            )
            if test_result.stdout.strip() in ['200', '422', '400']:
                print(f"   API Status: ✅ Responsive")
            else:
                print(f"   API Status: ⏳ Starting up (may take a few minutes)")
        else:
            print(f"\n⚠️  Container '{self.container_name}' exists but is NOT RUNNING")
            print(f"   Status: {status}")
            print(f"   Use 'docker_nim.py start' to start it")
        
        print()
        return container_id is not None
    
    def start(self):
        """Start the NIM container."""
        print("=" * 80)
        print("🚀 Starting NVIDIA NIM Container")
        print("=" * 80)
        
        # Check if already running
        if self._get_container_id():
            print("\n⚠️  Container is already running!")
            self.status()
            return
        
        # Validate API key
        if not self.api_key:
            print("\n❌ Error: NGC_API_KEY not found!")
            print("   Set it using: export NGC_API_KEY=<your_key>")
            print("   Or pass it with: --api-key <your_key>\n")
            sys.exit(1)
        
        # Create cache directory
        print(f"\n📁 Setting up cache directory: {self.cache_dir}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if container exists but is stopped
        status = self._get_container_status()
        if status and "Exited" in status:
            print(f"\n🔄 Found stopped container, removing it...")
            self._run_command(f"docker rm {self.container_name}")
        
        # Pull the image if not present
        print(f"\n🔽 Checking for image: {self.image}")
        image_check = self._run_command(f"docker image inspect {self.image}")
        if image_check.returncode != 0:
            print(f"   Image not found locally, pulling from registry...")
            print(f"   ⏳ This may take several minutes...")
            pull_result = self._run_command(f"docker pull {self.image}", capture_output=False)
            if pull_result.returncode != 0:
                print(f"\n❌ Failed to pull image. Check your NGC_API_KEY and internet connection.\n")
                sys.exit(1)
        else:
            print(f"   ✅ Image found locally")
        
        # Start the container
        print(f"\n🐳 Starting Docker container...")
        print(f"   Container name: {self.container_name}")
        print(f"   Port: {self.port}")
        print(f"   Cache directory: {self.cache_dir}")
        
        docker_cmd = f"""
        docker run -d --rm \\
            --name {self.container_name} \\
            --gpus all \\
            --shm-size=16GB \\
            -e NGC_API_KEY={self.api_key} \\
            -v "{self.cache_dir}:/opt/nim/.cache" \\
            -u $(id -u) \\
            -p {self.port}:8000 \\
            {self.image}
        """
        
        result = self._run_command(docker_cmd.strip())
        
        if result.returncode == 0:
            print(f"\n✅ Container started successfully!")
            print(f"   Container ID: {result.stdout.strip()}")
            print(f"\n⏳ Model is loading... This may take 2-5 minutes.")
            print(f"   API will be available at: http://localhost:{self.port}/v1/embeddings")
            print(f"\n💡 Use 'docker_nim.py status' to check if the API is ready")
            print(f"   Use 'docker_nim.py logs -f' to watch the startup logs\n")
        else:
            print(f"\n❌ Failed to start container!")
            print(f"   Error: {result.stderr}")
            print(f"\n💡 Common issues:")
            print(f"   - NVIDIA Docker runtime not installed (nvidia-docker2)")
            print(f"   - GPU not available")
            print(f"   - Port {self.port} already in use\n")
            sys.exit(1)
    
    def stop(self):
        """Stop the NIM container."""
        print("=" * 80)
        print("🛑 Stopping NVIDIA NIM Container")
        print("=" * 80)
        
        container_id = self._get_container_id()
        
        if not container_id:
            print(f"\n⚠️  Container '{self.container_name}' is not running\n")
            return
        
        print(f"\n🐳 Stopping container: {container_id}")
        result = self._run_command(f"docker stop {self.container_name}")
        
        if result.returncode == 0:
            print(f"✅ Container stopped successfully!\n")
        else:
            print(f"❌ Failed to stop container: {result.stderr}\n")
            sys.exit(1)
    
    def restart(self):
        """Restart the NIM container."""
        print("=" * 80)
        print("🔄 Restarting NVIDIA NIM Container")
        print("=" * 80)
        print()
        
        if self._get_container_id():
            self.stop()
            time.sleep(2)
        
        self.start()
    
    def logs(self, follow=False):
        """Show container logs."""
        print("=" * 80)
        print("📋 NVIDIA NIM Container Logs")
        print("=" * 80)
        print()
        
        if not self._get_container_id():
            print(f"❌ Container '{self.container_name}' is not running\n")
            return
        
        follow_flag = "-f" if follow else ""
        cmd = f"docker logs {follow_flag} {self.container_name}"
        
        if follow:
            print("📡 Following logs (Ctrl+C to exit)...\n")
        
        self._run_command(cmd, capture_output=False)


def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Manage NVIDIA NIM Docker container for LLama-3.2-NV-EmbedQA-1B-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the NIM container
  python docker_nim.py start
  
  # Start with custom API key
  python docker_nim.py start --api-key nvapi-xxx
  
  # Check container status
  python docker_nim.py status
  
  # Stop the container
  python docker_nim.py stop
  
  # Restart the container
  python docker_nim.py restart
  
  # View logs
  python docker_nim.py logs
  
  # Follow logs in real-time
  python docker_nim.py logs -f
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the NIM container')
    start_parser.add_argument('--api-key', help='NGC API key (or set NGC_API_KEY env var)')
    start_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help=f'Cache directory (default: {DEFAULT_CACHE_DIR})')
    start_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to expose (default: {DEFAULT_PORT})')
    start_parser.add_argument('--image', default=DEFAULT_IMAGE, help='Docker image to use')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop the NIM container')
    
    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart the NIM container')
    restart_parser.add_argument('--api-key', help='NGC API key (or set NGC_API_KEY env var)')
    restart_parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help=f'Cache directory (default: {DEFAULT_CACHE_DIR})')
    restart_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to expose (default: {DEFAULT_PORT})')
    
    # Status command
    subparsers.add_parser('status', help='Check container status')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View container logs')
    logs_parser.add_argument('-f', '--follow', action='store_true', help='Follow log output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create deployer instance
    deployer = NIMDeployer(
        api_key=getattr(args, 'api_key', None),
        cache_dir=getattr(args, 'cache_dir', DEFAULT_CACHE_DIR),
        port=getattr(args, 'port', DEFAULT_PORT),
        image=getattr(args, 'image', DEFAULT_IMAGE)
    )
    
    # Execute command
    if args.command == 'start':
        deployer.start()
    elif args.command == 'stop':
        deployer.stop()
    elif args.command == 'restart':
        deployer.restart()
    elif args.command == 'status':
        deployer.status()
    elif args.command == 'logs':
        deployer.logs(follow=args.follow)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

