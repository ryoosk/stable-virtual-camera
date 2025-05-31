#!/usr/bin/env python3
"""
Example usage of the Stable Virtual Camera API.
"""

import requests
import base64
from PIL import Image
import io

def test_api_server():
    """Test the API server with a sample image."""
    
    url = "http://localhost:8000/render/single-image"
    
    test_image_path = "test_image.jpg"
    
    with open(test_image_path, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        data = {
            "trajectory_type": "orbit",
            "num_frames": 21,
            "cfg_scale": 3.0,
            "seed": 42
        }
        
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Generated {len(result['images'])} frames")
        
        first_image_b64 = result['images'][0]
        image_data = base64.b64decode(first_image_b64)
        image = Image.open(io.BytesIO(image_data))
        image.save("output_frame_0.jpg")
        print("Saved first frame as output_frame_0.jpg")
        
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_direct_api():
    """Test the API directly without the server."""
    from seva.api import SevaAPI
    
    api = SevaAPI(device="cuda")
    
    preprocessed = api.preprocess_single_image("test_image.jpg")
    
    results = api.render_novel_views(
        preprocessed_data=preprocessed,
        trajectory_type="orbit",
        num_frames=21,
        cfg_scale=3.0,
        seed=42
    )
    
    print(f"Generated {len(results['rendered_images'])} frames")

if __name__ == "__main__":
    test_direct_api()
