#!/usr/bin/env python3
"""
Test script for the Stable Virtual Camera API with real images.
"""

import os
import sys

def test_direct_api_with_real_image():
    """Test the API directly with a real image from assets."""
    print("Testing direct API usage with real image...")
    
    test_image_path = "/home/ubuntu/repos/stable-virtual-camera/assets/basic/vasedeck.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    print(f"Using test image: {test_image_path}")
    
    try:
        from seva.api import SevaAPI
        
        print("Initializing SevaAPI...")
        api = SevaAPI(device="cpu")
        
        print("Preprocessing image...")
        preprocessed = api.preprocess_single_image(test_image_path)
        print(f"Preprocessed data keys: {list(preprocessed.keys())}")
        
        print("Rendering novel views...")
        results = api.render_novel_views(
            preprocessed_data=preprocessed,
            trajectory_type="orbit",
            num_frames=5,
            cfg_scale=3.0,
            seed=42
        )
        
        rendered_images = results['rendered_images']
        if hasattr(rendered_images, '__iter__') and not isinstance(rendered_images, (list, tuple)):
            rendered_images = list(rendered_images)
            results['rendered_images'] = rendered_images
        
        print(f"Generated {len(rendered_images)} frames")
        print(f"Result keys: {list(results.keys())}")
        print("Direct API test with real image passed!")
        
    except Exception as e:
        print(f"Direct API test failed: {e}")
        import traceback
        traceback.print_exc()

def test_image_preprocessing_only():
    """Test just the image preprocessing step."""
    print("\nTesting image preprocessing only...")
    
    test_image_path = "/home/ubuntu/repos/stable-virtual-camera/assets/basic/mountain-lake.jpg"
    
    try:
        from seva.api import SevaAPI
        
        print("Initializing SevaAPI...")
        api = SevaAPI(device="cpu")
        
        print("Testing preprocessing...")
        preprocessed = api.preprocess_single_image(test_image_path)
        
        print(f"Preprocessing successful!")
        print(f"Input image shape: {preprocessed['input_imgs'].shape}")
        print(f"Input K shape: {preprocessed['input_Ks'].shape}")
        print(f"Input c2w shape: {preprocessed['input_c2ws'].shape}")
        print(f"Image dimensions: {preprocessed['input_wh']}")
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_preprocessing_only()
    
    test_direct_api_with_real_image()
