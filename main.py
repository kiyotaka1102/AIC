from flask import Flask, request, jsonify, send_from_directory
import os
import json
import numpy as np
import torch
import faiss
from your_faiss_module import MyFaiss  # Update this import based on your file structure

app = Flask(__name__)

# Define your working directory and paths
WORK_DIR = "/path/to/your/work_dir"  # Change this to your actual working directory

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MyFaiss
bin_files = [
    f"{WORK_DIR}/data/dicts/bin_ocr/faiss_OCR_cosine.bin",
    f"{WORK_DIR}/data/dicts/bin_clip/faiss_CLIP_cosine.bin",
    f"{WORK_DIR}/data/dicts/bin_nomic/faiss_nomic_cosine.bin",
    f"{WORK_DIR}/data/dicts/bin_blip/faiss_BLIP_cosine.bin"
]
modes = ["ocr", "clip", "nomic", "blip"]
rerank_bin_file = f"{WORK_DIR}/data/dicts/bin_vlm/faiss_VLM_cosine.bin"
json_path = f"{WORK_DIR}/data/dicts/keyframes_id_search.json"

faiss_instance = MyFaiss(bin_files, json_path, device, modes, rerank_bin_file)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('text')
    
    if not query_text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Perform text search using MyFaiss
        result_strings = faiss_instance.text_search(query_text, k=5)
        
        # Create a list of image paths
        base_path = f"{WORK_DIR}/data/"
        image_paths = [os.path.join(base_path, path) for path in result_strings if path]
        
        return jsonify({'images': image_paths})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
