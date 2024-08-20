import faiss
import json
import numpy as np
import torch
import clip
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import math
from langdetect import detect
from sklearn.decomposition import PCA

class MyFaiss:
    def __init__(self, bin_files: list, dict_json: str, device, modes: list, rerank_bin_file: str = None):
        # Ensure that bin_files and modes lists have the same length
      assert len(bin_files) == len(modes), "The number of bin_files must match the number of modes"
      self.indexes = [self.load_bin_file(f) for f in bin_files]

    # Initialize re-ranking index if provided
      self.rerank_index = self.load_bin_file(rerank_bin_file) if rerank_bin_file else None

      self.translate = Translation()
      self.dict_json = self._read_json(dict_json)
      self.modes = modes
      self.device = device
      # Initialize models based on modes
      #self.models =[]
      self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    #   for mode in modes:
    #       if mode == "clip":
    #           model, preprocess = clip.load("ViT-B/32", device=self.device)
    #           self.models.append(model)
    #       else:
    #           model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    #           self.models.append(model)

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def _read_json(self, file_json):
        with open(file_json, "r") as file:
            data = json.load(file)
        return data

    # def _initialize_models(self):
    #     for idx, mode in enumerate(self.modes):
    #         if mode == "clip":
    #             self.models[idx] = clip.load("ViT-B/32", device=self.device)
    #         else:
    #             self.models[idx] = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    def image_search(self, id_query, k):
        # Gather features from all indices
        query_feats = []
        for index in self.indexes:
            features = index.reconstruct(id_query).reshape(1, -1)
            query_feats.append(features)

        # Stack the features from all indices
        initial_image_features = np.vstack(query_feats)

        # Perform the search using the first index (primary search)
        scores, idx_image = self.indexes[0].search(initial_image_features, k=k)
        idx_image = idx_image.flatten()

        # Map indices to result strings
        image_paths = [self.dict_json.get(idx) for idx in idx_image]

        return scores, idx_image, image_paths

    def show_images(self, image_paths):
      num_images = len(image_paths)

      if num_images == 0:
          print("No images to display.")
          return

      # Calculate grid size
      columns = int(math.sqrt(num_images))
      rows = int(np.ceil(num_images / columns))

      # Adjust figure size for 50 images
      fig = plt.figure(figsize=(15, 15))

      for i in range(num_images):
          img = plt.imread(image_paths[i])
          ax = fig.add_subplot(rows, columns, i + 1)
          ax.set_title('/'.join(image_paths[i].split('/')[-3:]), fontsize=8)
          ax.imshow(img)
          ax.axis("off")

      # Turn off the axes for any remaining subplot areas that do not have images
      for j in range(num_images, rows * columns):
          fig.add_subplot(rows, columns, j + 1).axis('off')

      plt.tight_layout(pad=1.0)  # Adjust padding for better spacing
      plt.show()

    def text_search(self, text, k):
      text = self.translate(text)
      print("Text translation:", text)

      all_results = []

      text_features = self.model.encode(text)
      text_features = text_features.reshape(1, -1)
      text_features_rr = text_features
      for index in self.indexes:
          if text_features.shape[1] != index.d:
              if text_features.shape[1] < index.d:
                  text_features = np.pad(text_features, ((0, 0), (0, index.d - text_features.shape[1])), 'constant')
              else:
                  text_features = text_features[:, :index.d]

          scores, idx_image = index.search(text_features, k=k)
          all_results.append((scores, idx_image))

      if self.rerank_index is not None:
          rerank_features_list = []
          rerank_indices = []

          for _, idx_image in all_results:
              rerank_features = np.array([self.rerank_index.reconstruct(int(idx)) for idx in idx_image[0]])
              rerank_features_list.append(rerank_features)
              rerank_indices.extend(idx_image[0])

          # Check if rerank_features_list is empty
          if len(rerank_features_list) == 0:
              print("No rerank features to process.")
              return []

          rerank_features_combined = np.vstack(rerank_features_list)
          k_rerank = len(rerank_features_combined)

          if rerank_features_combined.shape[1] != self.rerank_index.d:
              if rerank_features_combined.shape[1] < self.rerank_index.d:
                  rerank_features_combined = np.pad(rerank_features_combined, ((0, 0), (0, self.rerank_index.d - rerank_features_combined.shape[1])), 'constant')
              else:
                  rerank_features_combined = rerank_features_combined[:, :self.rerank_index.d]

 
          
          # Create a new FAISS index for reranking
          search_rr_index = faiss.IndexFlatIP(self.rerank_index.d)  # Using inner product (cosine similarity)          
          search_rr_index.add(rerank_features_combined)
          #resize for text_search_rr
          if text_features_rr.shape[1] != self.rerank_index.d:
              if text_features_rr.shape[1] < self.rerank_index.d:
                  text_features_rr = np.pad(text_features_rr, ((0, 0), (0, self.rerank_index.d - text_features_rr.shape[1])), 'constant')
              else:
                  text_features_rr = text_features_rr[:, :self.rerank_index.d]

          # Perform FAISS search on the combined rerank features
          rerank_scores, rerank_idx_image = search_rr_index.search(text_features_rr, k=k_rerank)

          # Map rerank scores back to the original indices
          rerank_score_map = {}
          for i, idx in enumerate(rerank_idx_image[0]):
              original_idx = rerank_indices[idx]
              rerank_score_map[original_idx] = rerank_scores[0][i]
          

          # Sort all_results based on reranking scores
          sorted_results = sorted(all_results, key=lambda result: -np.mean([rerank_score_map.get(idx, 0) for idx in result[1][0]]))

          # Prepare final result strings using list indexing
          result_strings = []
          for scores, idx_image in sorted_results:
              result_strings.extend([self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None for idx in idx_image[0]])
      else:
          combined_results = []
          for scores, idx_image in all_results:
              combined_results.extend([(score, self.dict_json[idx_image[0][i]]) for i, score in enumerate(scores[0])])

          combined_results.sort(key=lambda x: -x[0])
          result_strings = [image_path for _, image_path in combined_results[:k]]

      return result_strings



    
def main():
    ##### TESTING #####
    # Define your working directory
    #WORK_DIR = "/path/to/your/work_dir"  # Change this to your actual working directory

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths to the primary Faiss indices for initial feature extraction
    bin_files = [
        f"{WORK_DIR}/data/dicts/bin_ocr/faiss_OCR_cosine.bin",
        f"{WORK_DIR}/data/dicts/bin_clip/faiss_CLIP_cosine.bin",
        f"{WORK_DIR}/data/dicts/bin_nomic/faiss_nomic_cosine.bin"
        f"{WORK_DIR}/data/dicts/bin_blip/faiss_BLIP_cosine.bin"  # Ensure this path is correct
    ]

    # Modes corresponding to each bin file
    modes = ["ocr", "clip","nomic", "blip"]  # Adjust modes according to your bin files
    #modes = ["nomic"]
    # Path to the re-ranking Faiss index
    rerank_bin_file = f"{WORK_DIR}/data/dicts/bin_vlm/faiss_VLM_cosine.bin"

    # Path to the JSON file
    json_path = f"{WORK_DIR}/data/dicts/keyframes_id_search.json"

    # Initialize MyFaiss with multiple initial indices and one re-ranking index
    cosine_faiss = MyFaiss(bin_files, json_path, device, modes, rerank_bin_file)

    ##### TEXT SEARCH #####
    text = "lũ lụt, mực nước dân cao"

    # Perform text search
    result_strings = cosine_faiss.text_search(text, k=12)

    # Extract image paths from the result strings
    image_paths = [result_string for result_string in result_strings if result_string is not None]

    # Base path for images
    base_path = f"{WORK_DIR}/data/"

    # Create absolute paths for each image
    img_paths = [os.path.join(base_path, image_path) for image_path in image_paths]
    print (len(img_paths))
    # Show images
    cosine_faiss.show_images(img_paths)

if __name__ == "__main__":
    main()

