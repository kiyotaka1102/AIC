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
      self.models = []
      for mode in modes:
          if mode == "clip":
              model, preprocess = clip.load("ViT-B/32", device=self.device)
              self.models.append(model)
          else:
              model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
              self.models.append(model)

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
        #if detect(text) == 'vi':
        text = self.translate(text)
        print("Text translation: ", text)

        # Collect results from all modes
        all_results = []
        rerank_results = None
        index_count_map = {}  # Dictionary to store counts for indices
        index_score_map = {}  # Dictionary to store the aggregated scores for each index

        for idx, (mode, index) in enumerate(zip(self.modes, self.indexes)):
            if mode == "clip":
                with torch.no_grad():
                    text_input = clip.tokenize([text]).to(self.device)
                    text_features = self.models[idx].encode_text(text_input).cpu().numpy().astype(np.float32)
            else:
                with torch.no_grad():
                    text_features = self.models[idx].encode(text)

            # Resize or pad text_features to match the dimensionality of self.d
            text_features = text_features.reshape(1, -1)
            if text_features.shape[1] != index.d:
                if text_features.shape[1] < index.d:
                    text_features = np.pad(text_features, ((0, 0), (0, index.d - text_features.shape[1])), 'constant')
                else:
                    text_features = text_features[:, :index.d]

            # Perform search with each index
            scores, idx_image = index.search(text_features, k=k)
            for i, idx in enumerate(idx_image[0]):
                if idx not in index_count_map:
                    index_count_map[idx] = 0
                index_count_map[idx] += 1
                if idx not in index_score_map:
                    index_score_map[idx] = 0
                index_score_map[idx] += scores[0][i]
            all_results.append((scores, idx_image, mode, index, idx))
            
            if text_features.shape[1] != self.rerank_index.d:
                if text_features.shape[1] < self.rerank_index.d:
                    print("Padding rerank_features to match re-ranking index dimensionality.")
                    text_features = np.pad(text_features, ((0, 0), (0, self.rerank_index.d - text_features.shape[1])), 'constant')
                else:
                    print("Trimming rerank_features to match re-ranking index dimensionality.")
                    text_features = text_features[:, :self.rerank_index.d]
            assert text_features.shape[1] == self.rerank_index.d, "Dimensionality mismatch for re-ranking features"
            
            if mode != "clip" and rerank_results is None:
                rerank_scores, rerank_idx_image = self.rerank_index.search(text_features, k=k)
                rerank_results = (rerank_scores, rerank_idx_image)

        if rerank_results:
            rerank_scores, rerank_idx_image = rerank_results

            # Flatten rerank_idx_image for easy indexing
            rerank_idx_image_flat = rerank_idx_image.flatten()

            # Create a dictionary to map index to its re-ranking score
            rerank_score_map = {}
            for i, idx in enumerate(rerank_idx_image_flat):
                rerank_score_map[idx] = rerank_scores[0, i]
            
            # # Adjust rerank scores based on the count
            adjusted_rerank_scores = {}
            for idx, count in index_count_map.items():
                multiplier = 1.2 + 0.6 * (count - 1)  # 1.2 for 1 occurrence, 1.8 for 2 occurrences, etc.
                adjusted_rerank_scores[idx] = rerank_score_map.get(idx, 0) * multiplier
            
            # Sort all_results based on adjusted reranking scores
            sorted_results = sorted(all_results, key=lambda result: -np.mean([adjusted_rerank_scores.get(idx, 0) for idx in result[1][0]]))

            # Prepare final result strings using list indexing
            result_strings = []
            for scores, idx_image, mode, index, idx in sorted_results:
                result_strings.extend([self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None for idx in idx_image[0]])
        else:
            # If no re-ranking is provided, sort by scores
            combined_scores = np.concatenate([r[0] for r in all_results], axis=0)  # Flatten scores from all results
            sorted_indices = np.argsort(-combined_scores.flatten())  # Sorting by scores in descending order
            top_indices = sorted_indices[:k]
            result_strings = [self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None for idx in top_indices]

        return result_strings

