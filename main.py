from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from typing import List, Optional
from src.service.faiss import MyFaiss

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cosine_faiss = MyFaiss(
    bin_files=[],
    dict_json='./data/dicts/keyframes_id_search.json',
    device=device,
    modes=[],
    rerank_bin_file=None
)

app.mount("/data", StaticFiles(directory="./data"), name="data")
templates = Jinja2Templates(directory="./src/template")

class LoadIndexRequest(BaseModel):
    bin_files: List[str]
    rerank_file: Optional[str] = None

class SearchQuery(BaseModel):
    text: str
    k: int

class ImageClickRequest(BaseModel):
    index: int
    k: int

@app.post("/load_index/")
async def load_index(request: LoadIndexRequest):
    try:
        bin_files = request.bin_files
        rerank_file = request.rerank_file
        modes = [file.split('/')[-1].split('_')[0] for file in bin_files]

        global cosine_faiss
        cosine_faiss = MyFaiss(
            bin_files=bin_files,
            dict_json='./data/dicts/keyframes_id_search.json',
            device=device,
            modes=modes,
            rerank_bin_file=rerank_file
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
        resolved_image_paths = [f"/data/{image_path}" for image_path in image_paths]
        return {"image_paths": resolved_image_paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image_click/")
async def image_click(request_Image: ImageClickRequest):
    try:
        if not cosine_faiss.indexes:
            raise HTTPException(status_code=400, detail="No index loaded")

        image_idx = request_Image.index
        k = request_Image.k

        # Debugging: Print the received values

        _, _, image_paths = cosine_faiss.image_search(image_idx, k)

        # Debugging: Print the search results

        resolved_image_paths = [f"/data/{image_path}" for image_path in image_paths]
        return {"image_paths": resolved_image_paths}
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
