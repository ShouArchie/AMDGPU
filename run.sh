#!/bin/bash
# Print divider line
divider() {
  echo "======================================================================"
}

# Check system information
divider
echo "SYSTEM INFORMATION:"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
uname -a
lscpu | grep "Model name" || echo "CPU information not available"

# Check for NVIDIA GPU
divider
echo "CHECKING FOR NVIDIA GPU:"
if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected"
  nvidia-smi
  PLATFORM="nvidia"
else
  echo "NVIDIA GPU not detected"
fi

# Check for AMD GPU
divider
echo "CHECKING FOR AMD GPU:"
if command -v rocminfo &> /dev/null; then
  echo "AMD GPU detected (ROCm available)"
  rocminfo | grep -E "Name:|Marketing"
  PLATFORM="amd"
elif [ "$PLATFORM" != "nvidia" ]; then
  echo "AMD GPU not detected or ROCm not installed"
  PLATFORM="cpu"
fi

# Create output directory
divider
echo "SETTING UP ENVIRONMENT:"
mkdir -p outputs
echo "Created outputs directory"

# Create model cache directory
mkdir -p shap_e_model_cache
echo "Created model cache directory"

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Warning: Some dependencies may have failed to install"
fi

# Install Shap-E directly from GitHub
echo "Ensuring Shap-E is installed correctly..."
pip install git+https://github.com/openai/shap-e.git
if [ $? -ne 0 ]; then
  echo "Warning: Shap-E installation may have issues"
fi

# Define Python script based on platform - use platform_ready.py for AMD GPUs
divider
echo "DETECTED PLATFORM: $PLATFORM"
SCRIPT="main.py"

if [ "$PLATFORM" = "amd" ]; then
  echo "Using platform_ready.py for AMD GPU support"
  SCRIPT="platform_ready.py"
elif [ "$PLATFORM" = "nvidia" ]; then
  echo "Using main.py for NVIDIA GPU support"
else
  echo "Using platform_ready.py for CPU fallback"
  SCRIPT="platform_ready.py"
fi

# Run the Flask app
divider
echo "STARTING FLASK APPLICATION WITH: python $SCRIPT"
echo "The application will be available at http://localhost:8000"
divider

# Export PYTORCH_HF_CACHE_HOME to use our local cache directory
export PYTORCH_HF_CACHE_HOME="$(pwd)/shap_e_model_cache"
export HF_HOME="$(pwd)/shap_e_model_cache"
export TRANSFORMERS_CACHE="$(pwd)/shap_e_model_cache"

# Run the app
python $SCRIPT 