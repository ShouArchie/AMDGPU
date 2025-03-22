import os
import torch
import base64
import tempfile
from flask import Flask, request, jsonify, send_file
import numpy as np
import time
import traceback
import subprocess
import sys

app = Flask(__name__)

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Check if shap-e is installed, if not install it
try:
    import shap_e
    print("Shap-E is already installed.")
except ImportError:
    print("Installing Shap-E from GitHub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/shap-e.git"])
    print("Shap-E installed successfully.")

# Now we can import the required modules from shap-e
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

# Versatile GPU detection for UofT platform
device = torch.device('cpu')  # Default fallback

def check_gpu_compatibility():
    global device
    
    # Try to detect platform type
    platform_gpu_type = "unknown"
    try:
        # Check for NVIDIA GPUs first
        if torch.cuda.is_available():
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
            
        # Check for AMD GPUs via ROCm (if NVIDIA not found)
        elif hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            platform_gpu_type = "amd"
            print("AMD ROCm platform detected")
            device = torch.device('cuda:0')  # ROCm uses CUDA device naming
            print("Selected AMD GPU for processing")
            
        else:
            print("No compatible GPUs found, using CPU")
    except Exception as e:
        print(f"Error during GPU detection: {str(e)}")
        print("Falling back to CPU")
    
    print(f"Selected device: {device}")
    return platform_gpu_type

# Detect and configure GPU
platform_type = check_gpu_compatibility()

# Initialize models to None, will load on first request to avoid startup failures
xm = None
model = None
diffusion = None

def load_models_if_needed():
    global xm, model, diffusion
    if xm is None or model is None or diffusion is None:
        print("Loading Shap-E models...")
        try:
            xm = load_model('transmitter', device=device)
            model = load_model('text300M', device=device)
            diffusion = diffusion_from_config(load_config('diffusion'))
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            raise

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
        "batch_size": 1,             // optional
    }
    """
    try:
        # Ensure models are loaded
        load_models_if_needed()
        
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing prompt in request"}), 400
        
        prompt = data.get('prompt')
        guidance_scale = data.get('guidance_scale', 15.0)
        batch_size = data.get('batch_size', 1)
        
        print(f"Generating 3D model for prompt: '{prompt}'")
        
        # Start timing
        start_time = time.time()
        
        # Generate latents with the text model
        # Providing all required arguments for sample_latents
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            clip_denoised=True,
            use_fp16=True,
            use_karras=False,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
            device=device
        )
        
        # Create a unique filename
        timestamp = int(time.time())
        file_prefix = f"{timestamp}_{prompt.replace(' ', '_')[:20]}"
        stl_path = os.path.join('outputs', f"{file_prefix}.stl")
        
        # We take the first (and only) latent in the batch and decode it
        latent = latents[0]
        mesh = decode_latent_mesh(xm, latent).tri_mesh()
        
        # Export to STL file
        mesh.write_stl(stl_path)
        
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
            <p>This API generates 3D models from text prompts using Shap-E.</p>
            
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
                <li><code>batch_size</code>: Number of samples to generate (default: 1)</li>
            </ul>
            
            <p>The API will return an STL file that can be used for 3D printing or visualization.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 