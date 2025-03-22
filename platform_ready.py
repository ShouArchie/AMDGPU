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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the local shap-e directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'shap-e'))
logger.info(f"Added {os.path.join(os.getcwd(), 'shap-e')} to Python path")

app = Flask(__name__)

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Create cache directory if it doesn't exist
os.makedirs('shap_e_model_cache', exist_ok=True)

# Set environment variables for HuggingFace cache
os.environ['PYTORCH_HF_CACHE_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'shap_e_model_cache')

# Try to import Shap-E modules from the local directory
try:
    logger.info("Attempting to import Shap-E modules from local directory")
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import decode_latent_mesh
    logger.info("Successfully imported Shap-E modules")
    HAS_SHAPE = True
except ImportError as e:
    logger.error(f"Error importing Shap-E modules: {e}")
    logger.info("Falling back to mock implementation")
    HAS_SHAPE = False

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
            logger.info("AMD ROCm platform detected")
            device = torch.device('cuda:0')  # ROCm uses CUDA device naming
            logger.info("Selected AMD GPU for processing")
        # Check for NVIDIA GPUs second
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            platform_gpu_type = "nvidia"
            logger.info(f"Found {torch.cuda.device_count()} NVIDIA CUDA device(s)")
            
            # List available NVIDIA GPUs
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  • GPU {i}: {name} ({mem:.1f} GB memory)")
            
            # Select first NVIDIA GPU
            device = torch.device('cuda:0')
            logger.info(f"Selected {torch.cuda.get_device_name(0)} for processing")
        else:
            logger.info("No compatible GPUs found, using CPU")
    except Exception as e:
        logger.error(f"Error during GPU detection: {str(e)}")
        logger.info("Falling back to CPU")
    
    logger.info(f"Selected device: {device}")
    return platform_gpu_type

# Detect and configure GPU
platform_type = check_gpu_compatibility()

# Simple mock implementation for testing without Shap-E installed
class MockShapEInterface:
    def __init__(self, device):
        self.device = device
        logger.info(f"Initializing mock Shap-E interface on {device}")
        
    def generate_mesh(self, prompt, guidance_scale=15.0):
        logger.info(f"Generating 3D model for prompt: '{prompt}' with guidance_scale={guidance_scale}")
        time.sleep(2)  # Simulate processing time
        
        # Create a simple mesh based on keywords in the prompt
        import trimesh
        
        # Default shape is a cube
        mesh = trimesh.creation.box([1, 1, 1])
        
        # Check for common shapes in the prompt
        prompt_lower = prompt.lower()
        if "sphere" in prompt_lower or "ball" in prompt_lower or "round" in prompt_lower:
            mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
        elif "cylinder" in prompt_lower or "tube" in prompt_lower or "pipe" in prompt_lower:
            mesh = trimesh.creation.cylinder(radius=0.5, height=1.0)
        elif "cone" in prompt_lower:
            mesh = trimesh.creation.cone(radius=0.5, height=1.0)
        elif "torus" in prompt_lower or "donut" in prompt_lower or "ring" in prompt_lower:
            # Create a simple torus
            r_torus = 0.5  # radius from center of tube to center of torus
            r_tube = 0.2   # radius of the tube
            n_major = 40    # number of points along the major axis
            n_minor = 20    # number of points along the minor axis
            
            u = np.linspace(0, 2*np.pi, n_major)
            v = np.linspace(0, 2*np.pi, n_minor)
            u, v = np.meshgrid(u, v)
            u = u.flatten()
            v = v.flatten()
            
            x = (r_torus + r_tube * np.cos(v)) * np.cos(u)
            y = (r_torus + r_tube * np.cos(v)) * np.sin(u)
            z = r_tube * np.sin(v)
            
            vertices = np.vstack([x, y, z]).T
            
            # Create faces for the mesh
            faces = []
            for i in range(n_major):
                for j in range(n_minor):
                    # Get indices of the 4 points for this face
                    p1 = j * n_major + i
                    p2 = j * n_major + (i + 1) % n_major
                    p3 = ((j + 1) % n_minor) * n_major + (i + 1) % n_major
                    p4 = ((j + 1) % n_minor) * n_major + i
                    
                    # Create two triangular faces
                    faces.append([p1, p2, p3])
                    faces.append([p1, p3, p4])
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return mesh

# Initialize models
xm = None
model = None
diffusion = None

def initialize_model():
    global xm, model, diffusion
    
    # If we have Shap-E, use that, otherwise use the mock
    if HAS_SHAPE:
        if xm is None or model is None or diffusion is None:
            logger.info("Loading Shap-E models...")
            try:
                xm = load_model('transmitter', device=device)
                model = load_model('text300M', device=device)
                diffusion = diffusion_from_config(load_config('diffusion'))
                logger.info("Shap-E models loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading Shap-E models: {str(e)}")
                traceback.print_exc()
                logger.info("Falling back to mock implementation")
                return False
        return True
    else:
        # Use mock implementation
        global mock_model
        if 'mock_model' not in globals() or mock_model is None:
            logger.info("Initializing mock model...")
            try:
                mock_model = MockShapEInterface(device)
                logger.info("Mock model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing mock model: {str(e)}")
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
        "torch_version": torch.__version__,
        "shap_e_available": HAS_SHAPE
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
        
        logger.info(f"Generating 3D model for prompt: '{prompt}'")
        
        # Start timing
        start_time = time.time()
        
        # Create a unique filename
        timestamp = int(time.time())
        file_prefix = f"{timestamp}_{prompt.replace(' ', '_')[:20]}"
        stl_path = os.path.join('outputs', f"{file_prefix}.stl")
        
        if HAS_SHAPE:
            # Generate with Shap-E
            logger.info("Using Shap-E for generation")
            # Generate latents with the text model
            latents = sample_latents(
                batch_size=1,
                model=model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[prompt]),
                clip_denoised=True,
                use_fp16=True,
                use_karras=False,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
                device=device
            )
            
            # We take the first (and only) latent in the batch and decode it
            latent = latents[0]
            mesh = decode_latent_mesh(xm, latent).tri_mesh()
            
            # Export to STL file
            mesh.write_stl(stl_path)
        else:
            # Generate with mock implementation
            logger.info("Using mock implementation for generation")
            global mock_model
            mesh = mock_model.generate_mesh(prompt, guidance_scale)
            
            # Export to STL file
            mesh.export(stl_path)
        
        end_time = time.time()
        logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        # Return the STL file to the client
        return send_file(
            stl_path,
            as_attachment=True,
            download_name=f"{file_prefix}.stl",
            mimetype='application/octet-stream'
        )
    
    except Exception as e:
        logger.error(f"Error generating 3D model: {e}")
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
    
    shape_status = "Available ✅" if HAS_SHAPE else "Not available ❌ (using mock implementation)"
    
    return f"""
    <html>
        <head>
            <title>Shap-E Text-to-3D API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .gpu-info {{ background: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
                .model-error {{ background: #ffebee; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
                .success {{ color: #2e7d32; }}
                .error {{ color: #c62828; }}
            </style>
        </head>
        <body>
            <h1>Shap-E Text-to-3D API</h1>
            <p>This API generates 3D models from text prompts.</p>
            
            <div class="gpu-info">
                <strong>Hardware:</strong> {gpu_info}<br>
                <strong>PyTorch:</strong> {torch.__version__}<br>
                <strong>Device:</strong> {device}<br>
                <strong>Shap-E:</strong> {shape_status}
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
            
            {'' if HAS_SHAPE else '''
            <div class="model-error">
                <strong class="error">⚠️ Notice:</strong> The actual Shap-E model is not available. 
                The API is running with a mock implementation that provides basic shapes based on keywords.
                <p>Supported keywords include: sphere, cylinder, cone, torus, etc.</p>
            </div>
            '''}
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 