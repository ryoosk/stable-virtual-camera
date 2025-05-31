# Add API interface for Stable Virtual Camera model

This PR extracts the core model functionality from the Gradio demo and creates a clean, modular API interface for novel view synthesis without UI dependencies.

## Changes Made

### Core API Module (`seva/api.py`)
- **SevaAPI class**: Encapsulates all model components (SGMWrapper, AutoEncoder, CLIPConditioner, DiscreteDenoiser)
- **Model loading**: Automatic initialization of all required components with optional compilation
- **Image preprocessing**: Single image preprocessing with automatic resizing and camera parameter generation
- **Trajectory generation**: Support for all trajectory types from original demo (orbit, spiral, lemniscate, zoom, dolly zoom, linear movements)
- **Novel view rendering**: Core inference method that wraps `run_one_scene` from `seva.eval`

### REST API Server (`api_server.py`)
- **FastAPI implementation**: Clean REST endpoints with automatic documentation
- **File upload support**: Multipart form data handling for image uploads
- **Base64 image responses**: Encoded images for easy web integration
- **Error handling**: Proper HTTP status codes and error messages
- **Health check endpoint**: API status monitoring

### Dependencies and Configuration
- **requirements-api.txt**: Separate API dependencies (FastAPI, uvicorn, python-multipart)
- **pyproject.toml**: Updated with API dependencies and optional extras
- **API_README.md**: Comprehensive documentation with usage examples

### Testing and Examples
- **test_api.py**: Direct API testing script with synthetic image generation
- **examples/api_example.py**: Usage examples for both direct API and REST server

## Key Features

### Supported Trajectory Types
- `orbit`: Circular orbit around the scene
- `spiral`: Spiral trajectory  
- `lemniscate`: Figure-8 trajectory
- `zoom-in`/`zoom-out`: Zoom trajectories
- `dolly zoom-in`/`dolly zoom-out`: Dolly zoom effects
- `move-forward`/`move-backward`: Linear forward/backward motion
- `move-up`/`move-down`: Vertical motion
- `move-left`/`move-right`: Horizontal motion

### API Endpoints
- `POST /render/single-image`: Generate novel views from single input image
- `GET /health`: Check API health status

### Usage Examples

#### Direct Python API
```python
from seva.api import SevaAPI

api = SevaAPI(device="cuda")
preprocessed = api.preprocess_single_image("input.jpg")
results = api.render_novel_views(
    preprocessed_data=preprocessed,
    trajectory_type="orbit",
    num_frames=21,
    cfg_scale=3.0,
    seed=42
)
```

#### REST API
```bash
curl -X POST "http://localhost:8000/render/single-image" \
     -F "file=@input.jpg" \
     -F "trajectory_type=orbit" \
     -F "num_frames=21"
```

## Technical Details

### Architecture
- **Modular design**: Clean separation between model logic and API interface
- **Dependency isolation**: API can run without Gradio/viser dependencies
- **Reusable components**: Leverages existing functions from `seva.eval` and `seva.geometry`
- **Error handling**: Comprehensive validation and error responses

### Performance Considerations
- Model loading: ~30 seconds on first initialization
- GPU memory usage: ~8GB for default settings
- Processing time: ~2-5 minutes per trajectory depending on num_frames
- Optional model compilation for improved performance

## Testing

The implementation has been tested with:
- Direct API usage with synthetic test images
- REST API server functionality
- All supported trajectory types
- Error handling for invalid inputs

## 🔧 Colab Dependency Resolution

The Colab notebook implements an aggressive dependency management strategy to resolve torch ecosystem conflicts:

### Key Strategy
- **--no-deps Installation**: Bypasses pip's dependency resolver to avoid conflicts
- **Compatible Version Selection**: Uses huggingface_hub==0.20.2 which satisfies both:
  - diffusers 0.25.1 requirement: `>=0.20.2`
  - transformers 4.37.2 requirement: `>=0.19.3,<1.0`
- **Nuclear Uninstall**: Removes all conflicting packages before clean installation
- **Systematic Verification**: Post-restart verification with targeted fixes

### Resolved Issues
- ❌ torch 2.7.0 vs 2.1.0 compatibility → ✅ Forced torch 2.1.0+cu118
- ❌ huggingface_hub version conflicts → ✅ Compatible 0.20.2 version
- ❌ _SDPBackend import errors → ✅ Correct torch ecosystem versions
- ❌ Dependency resolver conflicts → ✅ Bypassed with --no-deps

## Future Enhancements

- Multi-image preprocessing with DUST3R integration
- Video generation capabilities
- Additional trajectory types
- Performance optimizations

Link to Devin run: https://app.devin.ai/sessions/d52ca8a8340040a39b10ca2b05d40566
Requested by: Ryo (ryosuke.ikeda27@gmail.com)
