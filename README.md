# Shap-E Text-to-3D API

This API uses OpenAI's Shap-E to generate 3D models from text prompts. It's designed to run on multiple platforms including:

- ✅ UofT GenAIGenesis AI Compute Platform (AMD GPU)
- ✅ Local machines with NVIDIA GPUs (including GeForce RTX 3060)
- ✅ Any machine with CPU (slower, but works as fallback)

## 🌟 Features

- Generate 3D models from text descriptions
- Automatic GPU detection (NVIDIA or AMD)
- Optimized for the UofT compute platform
- Returns STL files for 3D printing or visualization
- RESTful API with simple JSON interface

## 🖥️ Hardware Support

| Device Type | Support | Performance | Notes |
|-------------|---------|-------------|-------|
| NVIDIA RTX 3060 | ✅ Full | Fast | Optimized for local testing |
| AMD GPUs (ROCm) | ✅ Full | Fast | Optimized for UofT platform |
| CPU | ✅ Basic | Slow | Fallback mode |

## 🚀 Quickstart

### Running Locally (with NVIDIA 3060)

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `./run.sh`
4. The API will be available at: http://localhost:8000

### Running on UofT Platform (with AMD GPU)

1. Submit this repository to the UofT platform
2. The job will be queued and executed using run.sh
3. Access the hosted API at: `/site/<job_id>`

## 📡 API Endpoints

### `GET /health`

Health check endpoint that reports GPU detection status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "device": "cuda:0",
  "device_type": "nvidia",
  "device_name": "NVIDIA GeForce RTX 3060",
  "torch_version": "2.0.1"
}
```

### `POST /generate`

Generates a 3D model from a text prompt.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A detailed unicorn"}' \
  --output unicorn.stl
```

Parameters:
- `prompt`: Text description of the 3D model (required)
- `guidance_scale`: Controls how closely the model follows your text (default: 15.0)
- `batch_size`: Number of samples to generate (default: 1)

## 🔍 Implementation Details

This project includes several key files:

- `main.py`: Optimized for NVIDIA GPUs (especially 3060)
- `platform_ready.py`: Multi-platform version that works with AMD GPUs
- `run.sh`: Smart script that detects hardware and runs the appropriate version
- `test_basic.py`: Simple test script to verify API functionality

The system automatically detects the available hardware:
1. First checks for NVIDIA GPUs (preferred for local testing)
2. Then checks for AMD GPUs (preferred on UofT platform)
3. Falls back to CPU if no GPU is available

## 📊 Performance Comparison

| Hardware | Generation Time (approx.) |
|----------|---------------------------|
| NVIDIA RTX 3060 | 10-20 seconds |
| AMD MI100 (UofT) | 10-20 seconds |
| CPU | 2-5 minutes |

## 📝 Testing Your Setup

After starting the server, run the test script:

```bash
python test_basic.py
```

This will verify that:
1. The health endpoint is accessible
2. GPU detection is working
3. Generation endpoint can produce a valid STL file

## 🔧 Troubleshooting

### NVIDIA GPU Issues
- Ensure you have the latest NVIDIA drivers installed
- Check that CUDA is properly installed for your PyTorch version
- Try running `nvidia-smi` to verify your GPU is detected

### AMD GPU Issues
- The UofT platform should have ROCm pre-installed
- If testing locally with AMD, ensure ROCm is properly installed
- PyTorch must be compiled with ROCm support

### General Issues
- First request might take longer due to model loading
- The server requires at least 4GB of GPU memory
- For large models, reduce `batch_size` to 1

## 🛠️ Advanced Configuration

For advanced users, you can modify `platform_ready.py` to fine-tune GPU parameters:
- Adjust precision with `use_fp16` parameter
- Change diffusion parameters like `karras_steps` for quality/speed tradeoffs

## 📚 Resources

- [Shap-E GitHub Repository](https://github.com/openai/shap-e)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [UofT GenAIGenesis Platform Documentation](http://100.66.69.43:5000/) 