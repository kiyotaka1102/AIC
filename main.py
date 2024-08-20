from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from typing import List
from src.service.faiss import MyFaiss
import os
app = FastAPI()

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_path = r'D:/AIC-24/AIC2024_UTE_AI_Unknown/data/'
# Initialize MyFaiss with initial parameters (no .bin files loaded yet)
cosine_faiss = MyFaiss(
    bin_files=[],
    dict_json='./data/dicts/keyframes_id_search.json',
    device=device,
    modes=[]
)

templates = Jinja2Templates(directory="./src/template")

class LoadIndexRequest(BaseModel):
    bin_files: List[str]

class SearchQuery(BaseModel):
    text: str
    k: int

@app.post("/load_index/")
async def load_index(request: LoadIndexRequest):
    try:
        bin_files = request.bin_files
        modes = [file.split('/')[-1].split('_')[0] for file in bin_files]  # Extract modes from file names

        global cosine_faiss
        cosine_faiss = MyFaiss(
            bin_files=bin_files,
            dict_json='./data/dicts/keyframes_id_search.json',
            device=device,
            modes=modes,
            rerank_bin_file='./src/working/dicts/bin_clip/faiss_CLIP_cosine.bin'  # Optional: set if using a re-ranking index
        )
        return JSONResponse(content={"status": "Index loaded from selected .bin files"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
async def search_images(query: SearchQuery):
    try:
        if not cosine_faiss.indexes:
            raise HTTPException(status_code=400, detail="No index loaded")
   
        image_paths = cosine_faiss.text_search(query.text, query.k)
        resolved_image_paths = [os.path.join(base_path, image_path) for image_path in image_paths]
        return {"image_paths": resolved_image_paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
