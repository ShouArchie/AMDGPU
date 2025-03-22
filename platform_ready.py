import os
import torch
import base64
import tempfile
from flask import Flask, request, jsonify, send_file
import numpy as np
import time
import traceback
import sys
import importlib.util

app = Flask(__name__)

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Create cache directory if it doesn't exist
os.makedirs('shap_e_model_cache', exist_ok=True)

# Set environment variables for HuggingFace cache
os.environ['PYTORCH_HF_CACHE_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'shap_e_model_cache')

# Try to import required modules - we'll implement the bare essentials directly if needed
try:
    # Try to import standard libraries first
    from transformers import CLIPTextModel, CLIPTokenizer
    print("Successfully imported transformers library")
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Please install transformers with 'pip install transformers'")

# Implement needed parts directly in case shap-e isn't available
class MinimalLatentDecoder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

# Versatile GPU detection for UofT platform
device = torch.device('cpu')  # Default fallback

def check_gpu_compatibility():
    global device
    
    # Try to detect platform type
    platform_gpu_type = "unknown"
    try:
        # Check for AMD GPUs via ROCm (if NVIDIA not found)
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            platform_gpu_type = "amd"
            print("AMD ROCm platform detected")
            device = torch.device('cuda:0')  # ROCm uses CUDA device naming
            print("Selected AMD GPU for processing")
        # Check for NVIDIA GPUs second
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            platform_gpu_type = "nvidia"
            print(f"Found {torch.cuda.device_count()} NVIDIA CUDA device(s)")
            
            # List available NVIDIA GPUs
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  • GPU {i}: {name} ({mem:.1f} GB memory)")
            
            # Select first NVIDIA GPU
            device = torch.device('cuda:0')
            print(f"Selected {torch.cuda.get_device_name(0)} for processing")
        else:
            print("No compatible GPUs found, using CPU")
    except Exception as e:
        print(f"Error during GPU detection: {str(e)}")
        print("Falling back to CPU")
    
    print(f"Selected device: {device}")
    return platform_gpu_type

# Detect and configure GPU
platform_type = check_gpu_compatibility()

# Simple mock implementation for testing without Shap-E installed
class MockShapEInterface:
    def __init__(self, device):
        self.device = device
        print(f"Initializing mock Shap-E interface on {device}")
        
    def generate_mesh(self, prompt, guidance_scale=15.0):
        print(f"Generating 3D model for prompt: '{prompt}' with guidance_scale={guidance_scale}")
        time.sleep(2)  # Simulate processing time
        
        # Create a simple mesh (just a cube in this mock implementation)
        import trimesh
        mesh = trimesh.creation.box([1, 1, 1])
        return mesh

# Initialize models
model = None

def initialize_model():
    global model
    if model is None:
        print("Initializing models...")
        try:
            model = MockShapEInterface(device)
            print("Models initialized successfully")
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            traceback.print_exc()
            return False
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    device_info = {
        "status": "ok",
        "device": str(device),
        "device_type": platform_type,
        "torch_version": torch.__version__
    }
    if device.type == 'cuda' and torch.cuda.is_available():
        device_info["device_name"] = torch.cuda.get_device_name(0)
    return jsonify(device_info)

@app.route('/generate', methods=['POST'])
def generate_3d():
    """
    Generate a 3D model from a text prompt and return the STL file.
    
    Expected input JSON:
    {
        "prompt": "A detailed unicorn", 
        "guidance_scale": 15.0,       // optional
    }
    """
    try:
        # Initialize the model
        if not initialize_model():
            return jsonify({"error": "Failed to initialize models"}), 500
        
        # Parse request
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing prompt in request"}), 400
        
        prompt = data.get('prompt')
        guidance_scale = float(data.get('guidance_scale', 15.0))
        
        print(f"Generating 3D model for prompt: '{prompt}'")
        
        # Start timing
        start_time = time.time()
        
        # Generate mesh using mock implementation
        mesh = model.generate_mesh(prompt, guidance_scale)
        
        # Create a unique filename
        timestamp = int(time.time())
        file_prefix = f"{timestamp}_{prompt.replace(' ', '_')[:20]}"
        stl_path = os.path.join('outputs', f"{file_prefix}.stl")
        
        # Export to STL file
        mesh.export(stl_path)
        
        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        # Return the STL file to the client
        return send_file(
            stl_path,
            as_attachment=True,
            download_name=f"{file_prefix}.stl",
            mimetype='application/octet-stream'
        )
    
    except Exception as e:
        print(f"Error generating 3D model: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Simple info page."""
    gpu_info = ""
    if device.type == 'cuda':
        if platform_type == "nvidia":
            gpu_info = f"NVIDIA {torch.cuda.get_device_name(0)}"
        elif platform_type == "amd":
            gpu_info = "AMD GPU (via ROCm)"
        else:
            gpu_info = "Unknown GPU"
    else:
        gpu_info = "CPU (No GPU detected)"
    
    return f"""
    <html>
        <head>
            <title>Shap-E Text-to-3D API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .gpu-info {{ background: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Shap-E Text-to-3D API</h1>
            <p>This API generates 3D models from text prompts.</p>
            
            <div class="gpu-info">
                <strong>Hardware:</strong> {gpu_info}<br>
                <strong>PyTorch:</strong> {torch.__version__}<br>
                <strong>Device:</strong> {device}
            </div>
            
            <h2>API Usage:</h2>
            <pre>
curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{{"prompt": "A detailed unicorn"}}'
            </pre>
            
            <h2>Optional Parameters:</h2>
            <ul>
                <li><code>guidance_scale</code>: Controls how closely the model follows your text (default: 15.0)</li>
            </ul>
            
            <p>The API will return an STL file that can be used for 3D printing or visualization.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 