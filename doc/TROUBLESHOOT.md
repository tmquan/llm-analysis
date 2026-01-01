# GPU & CUDA Troubleshooting Guide

## Server Configuration
- **GPUs**: 8x NVIDIA B300 SXM6 AC (275GB VRAM each)
- **Architecture**: Blackwell
- **Driver**: 590.44.01
- **CUDA**: 13.1
- **NVSwitch**: Yes (requires Fabric Manager)

---

## Common Issues & Solutions

### 1. Error 802: System Not Yet Initialized

**Symptoms:**
```
CUDA initialization: Unexpected error from cudaGetDeviceCount()
Error 802: system not yet initialized
torch.cuda.is_available() = False
torch.cuda.device_count() = 0 (or 8 but still fails)
```

**Cause:** NVIDIA Fabric Manager is not running. Required for NVSwitch multi-GPU systems.

**Solution:**
```bash
# Check if Fabric Manager is running
sudo systemctl status nvidia-fabricmanager

# If not installed, install it
sudo apt-get update
sudo apt-get install -y nvidia-fabricmanager-570

# Start the service
sudo systemctl start nvidia-fabricmanager
sudo systemctl enable nvidia-fabricmanager

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

### 2. GPU Reset Failed: "In use by another client"

**Symptoms:**
```
sudo nvidia-smi --gpu-reset -i 0
GPU 00000000:XX:00.0: In use by another client
```

**Cause:** System management processes are holding the GPUs.

**Solution:**
```bash
# Check what's using the GPUs
sudo fuser -v /dev/nvidia*

# Common culprits: nv-hostengine, nvsm_core, dcgm

# Stop the services
sudo systemctl stop nvsm dcgm 2>/dev/null
sudo pkill -9 nv-hostengine
sudo pkill -9 nvsm_core
sleep 2

# Now reset GPUs
sudo nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7

# Re-enable persistence mode
sudo nvidia-smi -pm 1

# Restart services
sudo systemctl start nvsm
sudo systemctl start nvidia-fabricmanager
```

---

### 3. Cannot Unload NVIDIA Kernel Modules

**Symptoms:**
```
rmmod: ERROR: Module nvidia_uvm is in use
rmmod: ERROR: Module nvidia is in use by: nvidia_uvm
```

**Solution:**
```bash
# Step 1: Stop all NVIDIA services
sudo systemctl stop nvidia-fabricmanager nvsm dcgm 2>/dev/null

# Step 2: Kill all processes using GPUs
sudo fuser -k /dev/nvidia*
sleep 2

# Step 3: Try unloading modules
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset  
sudo rmmod nvidia

# Step 4: Reload
sudo modprobe nvidia
sudo modprobe nvidia_uvm

# Step 5: Re-enable persistence and restart services
sudo nvidia-smi -pm 1
sudo systemctl start nvidia-fabricmanager
```

**If modules still won't unload:** Reboot is required.
```bash
sudo reboot
```

---

### 4. After Reboot: CUDA Still Not Working

**Full initialization sequence after reboot:**
```bash
# 1. Enable persistence mode
sudo nvidia-smi -pm 1

# 2. Load UVM module
sudo modprobe nvidia_uvm

# 3. Start Fabric Manager (CRITICAL for B300/NVSwitch systems)
sudo systemctl start nvidia-fabricmanager

# 4. Verify
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

---

### 5. Specific GPU Unavailable (e.g., GPU 2)

**Symptoms:**
```
CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

**Solution:**
```bash
# Check which processes are using that specific GPU
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv
sudo fuser -v /dev/nvidia2

# Kill the process
sudo kill -9 <PID>

# Or run your job on different GPUs
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 python your_script.py
```

---

### 6. HuggingFace Authentication Issues

**Symptoms:**
```
Dataset 'nvidia/...' is a gated dataset. You must be authenticated.
```

**Solution:**
```bash
# Option 1: Set environment variable
export HF_TOKEN=hf_your_token_here

# Option 2: Login via CLI
huggingface-cli login

# Verify token location
cat ~/.cache/huggingface/token
```

In Python scripts, use:
```python
from datasets import load_dataset
HF_TOKEN = os.environ.get('HF_TOKEN', True)
dataset = load_dataset("nvidia/dataset-name", token=HF_TOKEN)
```

---

## Quick Health Check Script

Save as `gpu_check.sh`:
```bash
#!/bin/bash
echo "=== NVIDIA Driver ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1

echo -e "\n=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo -e "\n=== Fabric Manager ==="
systemctl is-active nvidia-fabricmanager

echo -e "\n=== PyTorch CUDA ==="
python -c "import torch; print(f'Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not available in current env"

echo -e "\n=== Processes on GPUs ==="
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv 2>/dev/null || echo "None"
```

---

## Service Dependencies (Startup Order)

1. `nvidia` kernel module (auto-loaded)
2. `nvidia_uvm` module
3. `nvidia-fabricmanager` (REQUIRED for NVSwitch)
4. `nvsm` (optional, for monitoring)
5. `nvidia-dcgm` (optional, for metrics)

---

## Useful Commands Reference

```bash
# Check driver version
cat /proc/driver/nvidia/version

# List loaded NVIDIA modules
lsmod | grep nvidia

# Check GPU topology (NVLink connections)
nvidia-smi topo -m

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check NVLink status
nvidia-smi nvlink -s

# Reset all GPUs (after stopping services)
sudo nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7

# Enable persistence mode (survives process exit)
sudo nvidia-smi -pm 1
```

---

## When All Else Fails

```bash
# Full reset sequence
sudo systemctl stop nvidia-fabricmanager nvsm dcgm 2>/dev/null
sudo pkill -9 nv-hostengine nvsm_core
sudo fuser -k /dev/nvidia*
sleep 3
sudo nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7
sudo nvidia-smi -pm 1
sudo modprobe nvidia_uvm
sudo systemctl start nvidia-fabricmanager

# If that doesn't work: REBOOT
sudo reboot
```

After reboot, always start Fabric Manager:
```bash
sudo systemctl start nvidia-fabricmanager
```

