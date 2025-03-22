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
import gc

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

# Setup for NVIDIA GPU (targeting 3060)
device = torch.device('cpu')  # Default fallback
try:
    if torch.cuda.is_available():
        # Clear CUDA cache to optimize memory usage
        torch.cuda.empty_cache()
        gc.collect()
        
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s)")
        
        # List all available GPUs
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            free_memory = total_memory - torch.cuda.memory_allocated(i) / (1024**3)
            print(f"GPU {i}: {device_name} with {total_memory:.2f} GB memory ({free_memory:.2f} GB free)")
            
            # Check if this is a 3060
            if "3060" in device_name:
                device = torch.device(f'cuda:{i}')
                print(f"Selected NVIDIA 3060 GPU at index {i}")
                # Set some optimizations for RTX 3060
                torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
                
                # Apply 3060-specific memory optimizations
                # The 3060 has 12GB of VRAM, but we'll try to be conservative
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
                print("Applied RTX 3060 specific optimizations")
            elif i == 0 and device.type == 'cpu':  # Select the first GPU if 3060 isn't found
                device = torch.device(f'cuda:{i}')
                print(f"Selected {device_name} as default GPU")
        
        # If a CUDA device was selected, print additional details
        if device.type == 'cuda':
            cuda_index = device.index if device.index is not None else 0
            print(f"Using CUDA device: {torch.cuda.get_device_name(cuda_index)}")
            print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(cuda_index)}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(cuda_index).total_memory / (1024**3):.2f} GB")
            
            # Configure half-precision if using GPU to improve performance
            torch.set_float32_matmul_precision('high')
    else:
        print("No CUDA devices available.")
except Exception as e:
    print(f"Error detecting CUDA devices: {e}")
    print("Falling back to CPU.")

print(f"Using device: {device}")

# Initialize models to None, will load on first request to avoid startup failures
xm = None
model = None
diffusion = None

def load_models_if_needed():
    global xm, model, diffusion
    if xm is None or model is None or diffusion is None:
        print("Loading Shap-E models...")
        try:
            # Clear memory before loading models
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            xm = load_model('transmitter', device=device)
            model = load_model('text300M', device=device)
            diffusion = diffusion_from_config(load_config('diffusion'))
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            raise

def release_gpu_memory():
    """Helper function to release GPU memory after a generation"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        free_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3) - torch.cuda.memory_allocated(device) / (1024**3)
        print(f"Released GPU memory, {free_memory:.2f} GB free")

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    device_info = {
        "status": "ok",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == 'cuda' else "CPU"
    }
    
    # Add GPU memory info if available
    if device.type == 'cuda':
        try:
            total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(device) / (1024**3)
            free_mem = total_mem - allocated_mem
            device_info.update({
                "cuda_version": torch.version.cuda,
                "total_memory_gb": f"{total_mem:.2f}",
                "free_memory_gb": f"{free_mem:.2f}",
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(device)))
            })
        except Exception as e:
            device_info["memory_error"] = str(e)
    
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
    start_time = time.time()
    generation_start = None
    
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
        
        # Start timing for generation
        generation_start = time.time()
        
        # Clear memory before generation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
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
        
        # Release memory after generation
        release_gpu_memory()
        
        end_time = time.time()
        generation_time = end_time - generation_start if generation_start else "unknown"
        total_time = end_time - start_time
        
        print(f"Generation completed in {generation_time:.2f} seconds")
        print(f"Total request time: {total_time:.2f} seconds")
        
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
        
        # Release memory even in case of error
        release_gpu_memory()
        
        # Calculate timing for error reporting
        if generation_start:
            generation_time = time.time() - generation_start
            print(f"Error occurred after {generation_time:.2f} seconds of generation")
        
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Simple info page."""
    gpu_info = ""
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(device)
        # Special styling if it's a 3060
        if "3060" in device_name:
            gpu_info = f"<strong style='color:#76b900'>NVIDIA {device_name}</strong>"
        else:
            gpu_info = f"NVIDIA {device_name}"
        
        # Add memory info
        try:
            total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(device) / (1024**3)
            free_mem = total_mem - allocated_mem
            gpu_info += f" ({free_mem:.1f} GB free / {total_mem:.1f} GB total)"
        except:
            pass
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
                .nvidia {{ color: #76b900; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Shap-E Text-to-3D API</h1>
            <p>This API generates 3D models from text prompts using Shap-E.</p>
            
            <div class="gpu-info">
                <strong>Hardware:</strong> {gpu_info}<br>
                <strong>PyTorch:</strong> {torch.__version__}<br>
                <strong>CUDA:</strong> {torch.version.cuda if hasattr(torch.version, 'cuda') else "Not available"}<br>
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
