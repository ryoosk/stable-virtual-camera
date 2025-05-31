import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Stable Virtual Camera API - Colab Demo

This notebook demonstrates the Stable Virtual Camera API for novel view synthesis. The API extracts core model functionality from the original Gradio demo and provides clean, modular endpoints for programmatic access.

- **Direct Python API**: Clean SevaAPI class for programmatic access
- **REST API Server**: FastAPI endpoints for web integration
- **Multiple Trajectories**: Support for orbit, spiral, zoom, and linear camera movements
- **GPU Acceleration**: Optimized for CUDA execution

- GPU runtime (T4, V100, or A100 recommended)
- ~8GB GPU memory for default settings
- Hugging Face token for model access

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryoosk/stable-virtual-camera/blob/devin/1748692706-api-implementation/stable_virtual_camera_api_colab.ipynb)"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Installation"))

cells.append(nbf.v4.new_code_cell("""# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ GPU not available. Please enable GPU runtime: Runtime > Change runtime type > Hardware accelerator > GPU")"""))

cells.append(nbf.v4.new_code_cell("""# Clone the repository with API implementation
!git clone https://github.com/ryoosk/stable-virtual-camera.git
%cd stable-virtual-camera
!git checkout devin/1748692706-api-implementation

!ls -la
print("✅ Repository cloned and checked out successfully!")"""))

cells.append(nbf.v4.new_code_cell("""# Aggressive dependency fix bypassing pip resolver conflicts

print("🔧 Implementing aggressive dependency fix...")
print("🚨 This approach bypasses pip's dependency resolver to avoid conflicts!")

print("💥 Nuclear uninstall of all conflicting packages...")
!pip uninstall -y torch torchvision torchaudio transformers diffusers accelerate xformers sentence-transformers peft tensorflow jax jaxlib fastai -q

print("🔥 Installing torch ecosystem with dependency bypass...")
!pip install torch==2.6.0 torchvision==0.19.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 --no-deps
!pip install numpy==1.26.4 --no-deps  # Compatible numpy version
!pip install pillow typing-extensions sympy networkx jinja2 fsspec --no-deps

print("🤗 Installing transformers/diffusers with manual dependency control...")
!pip install huggingface_hub==0.20.2 --no-deps  # Use version that works with both
!pip install tokenizers==0.15.2 safetensors==0.4.5 --no-deps
!pip install transformers==4.37.2 --no-deps
!pip install diffusers==0.25.1 --no-deps
!pip install accelerate==0.26.1 --no-deps

# Step 4: Install remaining essential dependencies
print("📦 Installing remaining dependencies...")
!pip install requests tqdm regex pyyaml packaging filelock --no-deps

print("✅ Aggressive installation completed!")
print("⚠️  Runtime restart required. Please restart and run next cell.")

import os
os.kill(os.getpid(), 9)"""))

cells.append(nbf.v4.new_code_cell("""# Post-restart: Verify and complete setup with manual dependency management
print("🔍 Post-restart verification with manual dependency management...")

print("🔥 Verifying torch installation...")
try:
    import torch
    import torchvision
    print(f"✅ Torch {torch.__version__}, Torchvision {torchvision.__version__}")
    if not torch.__version__.startswith("2.6.0"):
        print(f"⚠️  Expected torch 2.6.0+cu118, got {torch.__version__}")
        print("🔄 Reinstalling correct torch version...")
        !pip uninstall torch torchvision torchaudio -y -q
        !pip install torch==2.6.0 torchvision==0.19.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 --no-deps
        import torch
        import torchvision
        print(f"✅ Fixed: Torch {torch.__version__}, Torchvision {torchvision.__version__}")
except Exception as e:
    print(f"❌ Torch verification failed: {e}")
    !pip install torch==2.6.0 torchvision==0.19.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 --no-deps

print("🤗 Verifying transformers...")
try:
    from transformers import AutoImageProcessor
    print("✅ AutoImageProcessor import successful!")
except Exception as e:
    print(f"❌ AutoImageProcessor failed: {e}")
    print("🔄 Installing transformers dependencies...")
    !pip install transformers==4.37.2 tokenizers==0.15.2 --no-deps

print("🎨 Verifying diffusers...")
try:
    from diffusers.models import AutoencoderKL
    print("✅ AutoencoderKL import successful!")
except Exception as e:
    print(f"❌ AutoencoderKL failed: {e}")
    print("🔄 Installing diffusers with compatible hub version...")
    !pip install huggingface_hub==0.20.2 diffusers==0.25.1 --no-deps

print("📦 Installing seva package...")
%cd /content/stable-virtual-camera
!pip install -e . --no-deps
!pip install fastapi uvicorn python-multipart imageio --no-deps

print("🎯 Final verification...")
try:
    from seva.api import SevaAPI
    print("✅ SevaAPI import successful!")
except Exception as e:
    print(f"❌ SevaAPI failed: {e}")
    print("🔄 Installing missing dependencies...")
    !pip install opencv-python matplotlib --no-deps
    from seva.api import SevaAPI
    print("✅ SevaAPI import fixed!")

print("✅ Setup completed with manual dependency management!")"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Authentication and Model Download

You need a Hugging Face token to access the gated model. Get one at: https://huggingface.co/settings/tokens"""))

cells.append(nbf.v4.new_code_cell("""import os
from getpass import getpass

hf_token = getpass("Enter your Hugging Face token: ")
os.environ['HUGGINGFACE_TOKEN'] = hf_token
print("✅ Token set successfully!")"""))

cells.append(nbf.v4.new_code_cell("""# Download the model (5.06GB)
from huggingface_hub import hf_hub_download

print("Downloading Stable Virtual Camera model...")
try:
    hf_hub_download(
        repo_id='stabilityai/stable-virtual-camera',
        filename='model.safetensors',
        local_dir='.',
        token=os.environ['HUGGINGFACE_TOKEN']
    )
    print("✅ Model downloaded successfully!")
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("Please check your token and try again.")"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Direct API Usage Demo"))

cells.append(nbf.v4.new_code_cell("""# Initialize the API with robust error handling
print("🚀 Initializing Stable Virtual Camera API...")
print("⚠️  This will download ~5GB of model weights on first run")
print("🚨 Any remaining dependency warnings can be safely ignored!")

try:
    from seva.api import SevaAPI
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("🔄 Attempting import fix...")
    !pip install -e . --force-reinstall --quiet
    from seva.api import SevaAPI
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch
    print("✅ Imports fixed!")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

try:
    api = SevaAPI(device=device, compile_model=False)
    print("✅ API initialized successfully!")
except Exception as e:
    print(f"❌ API initialization failed: {e}")
    if device == "cuda":
        print("🔄 Attempting CPU fallback...")
        api = SevaAPI(device="cpu", compile_model=False)
        print("✅ API initialized with CPU fallback!")
    else:
        raise e

print(f"🎯 Ready for novel view synthesis on {api.device}!")"""))

cells.append(nbf.v4.new_code_cell("""# Test image preprocessing
test_image_path = "assets/basic/vasedeck.jpg"

print(f"Testing preprocessing with: {test_image_path}")
preprocessed = api.preprocess_single_image(test_image_path)

print(f"✅ Preprocessing successful!")
print(f"Input image shape: {preprocessed['input_imgs'].shape}")
print(f"Camera intrinsics shape: {preprocessed['input_Ks'].shape}")
print(f"Camera pose shape: {preprocessed['input_c2ws'].shape}")
print(f"Image dimensions: {preprocessed['input_wh']}")

input_img = preprocessed['input_imgs'][0].numpy()
plt.figure(figsize=(8, 6))
plt.imshow(input_img)
plt.title("Input Image")
plt.axis('off')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Generate novel views with orbit trajectory
print("Generating novel views with orbit trajectory...")
print("This may take 2-5 minutes depending on GPU...")

results = api.render_novel_views(
    preprocessed_data=preprocessed,
    trajectory_type="orbit",
    num_frames=8,  # Reduced for faster demo
    cfg_scale=3.0,
    seed=42
)

rendered_images = results["rendered_images"]
print(f"✅ Generated {len(rendered_images)} frames!")"""))

cells.append(nbf.v4.new_code_cell("""# Display results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, img_tensor in enumerate(rendered_images):
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.cpu().numpy()
    else:
        img_np = img_tensor
    
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    axes[i].imshow(img_np)
    axes[i].set_title(f"Frame {i+1}")
    axes[i].axis('off')

plt.suptitle("Novel View Synthesis - Orbit Trajectory", fontsize=16)
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Different Trajectory Types Demo"))

cells.append(nbf.v4.new_code_cell("""# Test different trajectory types
trajectory_types = ["spiral", "zoom-in", "move-forward"]
trajectory_results = {}

for traj_type in trajectory_types:
    print(f"Generating {traj_type} trajectory...")
    
    results = api.render_novel_views(
        preprocessed_data=preprocessed,
        trajectory_type=traj_type,
        num_frames=4,  # Fewer frames for demo
        cfg_scale=3.0,
        seed=42
    )
    
    trajectory_results[traj_type] = results["rendered_images"]
    print(f"✅ {traj_type} completed!")

print("All trajectories generated!")"""))

cells.append(nbf.v4.new_code_cell("""# Display trajectory comparison
fig, axes = plt.subplots(len(trajectory_types), 4, figsize=(16, 12))

for row, traj_type in enumerate(trajectory_types):
    for col, img_tensor in enumerate(trajectory_results[traj_type]):
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.cpu().numpy()
        else:
            img_np = img_tensor
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        axes[row, col].imshow(img_np)
        if col == 0:
            axes[row, col].set_ylabel(traj_type, fontsize=12)
        axes[row, col].set_title(f"Frame {col+1}")
        axes[row, col].axis('off')

plt.suptitle("Trajectory Comparison", fontsize=16)
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Upload Your Own Image"))

cells.append(nbf.v4.new_code_cell("""# Upload your own image
from google.colab import files
import os

print("Upload an image file:")
uploaded = files.upload()

if uploaded:
    uploaded_filename = list(uploaded.keys())[0]
    print(f"Uploaded: {uploaded_filename}")
    
    print("Processing uploaded image...")
    custom_preprocessed = api.preprocess_single_image(uploaded_filename)
    
    custom_results = api.render_novel_views(
        preprocessed_data=custom_preprocessed,
        trajectory_type="orbit",
        num_frames=6,
        cfg_scale=3.0,
        seed=42
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_tensor in enumerate(custom_results["rendered_images"]):
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.cpu().numpy()
        else:
            img_np = img_tensor
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(f"Custom Image Results: {uploaded_filename}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("✅ Custom image processing completed!")
else:
    print("No file uploaded.")"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Summary and Next Steps

🎉 **Congratulations!** You've successfully tested the Stable Virtual Camera API!

- ✅ Set up the API in a GPU environment
- ✅ Tested image preprocessing and novel view synthesis
- ✅ Explored different camera trajectories
- ✅ Processed custom images

- **Trajectory Types**: orbit, spiral, lemniscate, zoom-in/out, dolly zoom, linear movements
- **Flexible Interface**: Direct Python API and REST endpoints
- **GPU Optimized**: CUDA acceleration for fast inference
- **Modular Design**: Clean separation from UI dependencies

1. **Integration**: Use the API in your own applications
2. **Customization**: Modify trajectory parameters and camera settings
3. **Scaling**: Deploy the REST API server for production use
4. **Enhancement**: Contribute to the project with new features

- **GitHub Repository**: https://github.com/ryoosk/stable-virtual-camera
- **API Documentation**: See `API_README.md` in the repository
- **Original Paper**: Stable Virtual Camera for Novel View Synthesis

Happy coding! 🚀"""))

nb.cells = cells

with open('stable_virtual_camera_api_colab.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Colab notebook created successfully!")
