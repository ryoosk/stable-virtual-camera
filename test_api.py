#!/usr/bin/env python3
"""
Test script for the Stable Virtual Camera API.
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image

def create_test_image(path: str, size=(512, 512)):
    """Create a simple test image."""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)
    return path

def test_direct_api():
    """Test the API directly without the server."""
    print("Testing direct API usage...")
    
    try:
        from seva.api import SevaAPI
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            test_image_path = create_test_image(tmp_file.name)
        
        try:
            api = SevaAPI(device="cuda")
            
            preprocessed = api.preprocess_single_image(test_image_path)
            print(f"Preprocessed data keys: {preprocessed.keys()}")
            
            results = api.render_novel_views(
                preprocessed_data=preprocessed,
                trajectory_type="orbit",
                num_frames=5,
                cfg_scale=3.0,
                seed=42
            )
            
            print(f"Generated {len(results['rendered_images'])} frames")
            print("Direct API test passed!")
            
        finally:
            os.unlink(test_image_path)
            
    except Exception as e:
        print(f"Direct API test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_api()
