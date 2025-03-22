import requests
import time
import json
import os

# Settings
API_URL = "http://localhost:8000"
OUTPUT_FILE = "gpu_test_output.stl"
REQUEST_TIMEOUT = 600  # 10 minutes - models might take longer on first load

def print_separator():
    print("\n" + "="*50 + "\n")

# Test the health endpoint to check GPU detection
print("Testing health endpoint (GPU detection)...")
try:
    response = requests.get(f"{API_URL}/health", timeout=10)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("GPU Detection Result:")
        print(f"  • Device: {data.get('device', 'Unknown')}")
        print(f"  • Device Name: {data.get('device_name', 'Unknown')}")
        print(f"  • Status: {data.get('status', 'Unknown')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error reaching health endpoint: {e}")

print_separator()

# Test the root endpoint
print("Testing root endpoint (homepage)...")
try:
    response = requests.get(API_URL, timeout=10)
    print(f"Status code: {response.status_code}")
    print(f"Content length: {len(response.text)} characters")
    if response.status_code != 200:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error reaching root endpoint: {e}")

print_separator()

# Test a simple generation request
print("Testing generation endpoint with a simple request...")
try:
    # A simple prompt that should work well with Shap-E
    payload = {"prompt": "A simple cat", "guidance_scale": 12.0}
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    print("This may take a while, especially on first run...")
    
    start_time = time.time()
    
    # Make the request with streaming enabled
    print("Sending request and waiting for response...")
    response = requests.post(
        f"{API_URL}/generate",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=REQUEST_TIMEOUT
    )
    
    processing_time = time.time() - start_time
    print(f"Received response in {processing_time:.2f} seconds")
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        # Save the STL file
        print("Downloading STL file...")
        download_start = time.time()
        with open(OUTPUT_FILE, "wb") as f:
            byte_count = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                byte_count += len(chunk)
                # Print progress every 100KB
                if byte_count % (100 * 1024) == 0:
                    print(f"Downloaded {byte_count / 1024:.1f}KB...")
        
        download_time = time.time() - download_start
        total_time = time.time() - start_time
        file_size = os.path.getsize(OUTPUT_FILE) / 1024  # KB
        
        print("\n✅ Success!")
        print(f"Downloaded {file_size:.1f}KB STL file to {OUTPUT_FILE}")
        print(f"Download speed: {file_size / download_time:.1f}KB/s")
        print(f"Total processing time: {total_time:.2f} seconds")
        
        # Check if file seems valid (very basic check)
        if file_size < 1:
            print("⚠️ Warning: File is smaller than 1KB, might be invalid")
    else:
        print("❌ Error generating 3D model")
        try:
            error_data = response.json()
            print(f"Error details: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Error content: {response.text[:500]}...")
except requests.exceptions.Timeout:
    elapsed = time.time() - start_time
    print(f"❌ Request timed out after {elapsed:.1f} seconds")
    print("The model may still be loading or processing. Try again in a few minutes.")
except Exception as e:
    print(f"❌ Error during testing: {e}")

print_separator()
print("Test script completed.") 