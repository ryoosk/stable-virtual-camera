from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
import tempfile
import os
import base64
import io
from PIL import Image
import numpy as np
import torch

from seva.api import SevaAPI

app = FastAPI(title="Stable Virtual Camera API", version="1.0.0")

seva_api = None

class TrajectoryRequest(BaseModel):
    trajectory_type: Literal[
        "orbit", "spiral", "lemniscate", "zoom-in", "zoom-out",
        "dolly zoom-in", "dolly zoom-out", "move-forward", "move-backward", 
        "move-up", "move-down", "move-left", "move-right"
    ]
    num_frames: int = 21
    zoom_factor: Optional[float] = None
    cfg_scale: float = 3.0
    seed: int = 42
    chunk_strategy: str = "nearest"

class RenderResponse(BaseModel):
    success: bool
    images: List[str]
    metadata: dict

@app.on_event("startup")
async def startup_event():
    global seva_api
    seva_api = SevaAPI(device="cuda", compile_model=False)

@app.post("/render/single-image", response_model=RenderResponse)
async def render_single_image(
    file: UploadFile = File(...),
    trajectory_type: str = "orbit",
    num_frames: int = 21,
    zoom_factor: Optional[float] = None,
    cfg_scale: float = 3.0,
    seed: int = 42,
    chunk_strategy: str = "nearest"
):
    """
    Render novel views from a single input image.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            preprocessed = seva_api.preprocess_single_image(tmp_file_path)
            
            results = seva_api.render_novel_views(
                preprocessed_data=preprocessed,
                trajectory_type=trajectory_type,
                num_frames=num_frames,
                zoom_factor=zoom_factor,
                cfg_scale=cfg_scale,
                seed=seed,
                chunk_strategy=chunk_strategy
            )
            
            encoded_images = []
            if isinstance(results["rendered_images"], torch.Tensor):
                rendered_imgs = results["rendered_images"]
            else:
                rendered_imgs = torch.stack(results["rendered_images"])
            
            for i in range(rendered_imgs.shape[0]):
                img_tensor = rendered_imgs[i]
                if img_tensor.dim() == 4:
                    img_tensor = img_tensor[0]
                img_np = (img_tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                buffer = io.BytesIO()
                img_pil.save(buffer, format="JPEG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                encoded_images.append(img_base64)
            
            return RenderResponse(
                success=True,
                images=encoded_images,
                metadata=results["metadata"]
            )
            
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": seva_api is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
