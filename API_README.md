# Stable Virtual Camera API

This API provides programmatic access to the Stable Virtual Camera model for novel view synthesis.

## Installation

```bash
# Install the package
pip install -e .

# Install API dependencies
pip install -r requirements-api.txt
# or
pip install -e ".[api]"
```

## Quick Start

### Direct API Usage

```python
from seva.api import SevaAPI

# Initialize API
api = SevaAPI(device="cuda")

# Preprocess input image
preprocessed = api.preprocess_single_image("input.jpg")

# Generate novel views with orbit trajectory
results = api.render_novel_views(
    preprocessed_data=preprocessed,
    trajectory_type="orbit",
    num_frames=21,
    cfg_scale=3.0,
    seed=42
)

# Access rendered images
rendered_images = results["rendered_images"]  # List of torch tensors
```

### REST API Server

Start the server:
```bash
python api_server.py
```

Make requests:
```bash
curl -X POST "http://localhost:8000/render/single-image" \
     -F "file=@input.jpg" \
     -F "trajectory_type=orbit" \
     -F "num_frames=21" \
     -F "cfg_scale=3.0"
```

## API Reference

### SevaAPI Class

#### Methods

- `preprocess_single_image(image_path: str) -> Dict`: Preprocess single input image
- `render_novel_views(preprocessed_data: Dict, **kwargs) -> Dict`: Generate novel views
- `generate_trajectory_cameras(...)`: Generate camera trajectory

#### Supported Trajectory Types

- `orbit`: Circular orbit around the scene
- `spiral`: Spiral trajectory  
- `lemniscate`: Figure-8 trajectory
- `zoom-in`/`zoom-out`: Zoom trajectories
- `dolly zoom-in`/`dolly zoom-out`: Dolly zoom effects
- `move-forward`/`move-backward`: Linear forward/backward motion
- `move-up`/`move-down`: Vertical motion
- `move-left`/`move-right`: Horizontal motion

### REST API Endpoints

#### POST /render/single-image

Generate novel views from a single input image.

**Parameters:**
- `file`: Input image file (multipart/form-data)
- `trajectory_type`: Camera trajectory type
- `num_frames`: Number of frames to generate (default: 21)
- `cfg_scale`: Classifier-free guidance scale (default: 3.0)
- `seed`: Random seed (default: 42)

**Response:**
```json
{
  "success": true,
  "images": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
  "metadata": {"num_frames": 21, "seed": 42}
}
```

#### GET /health

Check API health status.

## Examples

### Python API Usage

```python
from seva.api import SevaAPI

api = SevaAPI(device="cuda")

# Single image processing
preprocessed = api.preprocess_single_image("input.jpg")
results = api.render_novel_views(
    preprocessed_data=preprocessed,
    trajectory_type="spiral",
    num_frames=30,
    cfg_scale=2.5,
    seed=123
)

# Access results
images = results["rendered_images"]
cameras = results["target_cameras"]
metadata = results["metadata"]
```

### REST API Usage

```python
import requests

url = "http://localhost:8000/render/single-image"
files = {"file": open("input.jpg", "rb")}
data = {
    "trajectory_type": "orbit",
    "num_frames": 21,
    "cfg_scale": 3.0,
    "seed": 42
}

response = requests.post(url, files=files, data=data)
result = response.json()
```

## Error Handling

The API provides detailed error messages for common issues:

- Invalid trajectory types
- Missing required parameters
- File format errors
- Model loading failures

## Performance Notes

- Model loading takes ~30 seconds on first initialization
- GPU memory usage: ~8GB for default settings
- Processing time: ~2-5 minutes per trajectory depending on num_frames
- Compilation can improve performance but increases startup time

## Testing

Run the test script to verify the API works:

```bash
python test_api.py
```

Or test the REST API server:

```bash
# Start the server
python api_server.py

# In another terminal, test with curl
curl -X POST "http://localhost:8000/render/single-image" \
     -F "file=@test_image.jpg" \
     -F "trajectory_type=orbit" \
     -F "num_frames=5"
```
