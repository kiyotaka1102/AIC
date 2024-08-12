import clip
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import matplotlib.pyplot as plt
import math
import numpy as np 
import clip
from langdetect import detect


class MyFaiss:
  def __init__(self, bin_file: str, dict_json: str,device , mode):
    self.index = self.load_bin_file(bin_file)
    self.translate = Translation()
    self.dict_json = self._read_json(dict_json)
    self.mode = mode
    self.device = device
    self.d = self.index.d
    if mode == "clip":
      self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    else:
      self.q_encoder = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)


  def load_bin_file(self, bin_file: str):
    return faiss.read_index(bin_file)

  def _read_json(self, file_json):
    with open(file_json, "r") as file:
      data = json.load(file)
    return data

  def image_search(self, id_query, k, bin_file):

    query_feats = self.index.reconstruct(id_query).reshape(1,-1)

    scores, idx_image = self.index.search(query_feats, k=k)
    idx_image = idx_image.flatten()

    infos_query = list(map(self.id2img_fps.get, list(idx_image)))
    image_paths = [info for info in infos_query]


    return scores, idx_image, infos_query, image_paths
  def show_images(self, image_paths):
    fig = plt.figure(figsize=(15, 10))
    columns = int(math.sqrt(len(image_paths)))
    rows = int(np.ceil(len(image_paths)/columns))

    for i in range(1, columns*rows +1):
      img = plt.imread(image_paths[i - 1])
      ax = fig.add_subplot(rows, columns, i)
      ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

      plt.imshow(img)
      plt.axis("off")

    plt.show()

  def text_search(self, text, k):
    if detect(text) == 'vi' :
        text = self.translate(text)
    print("Text translation: ", text)

    if self.mode == "clip":
        with torch.no_grad():
            text = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text).cpu().numpy().astype(np.float32)
    else:
        with torch.no_grad():
            text_features = self.q_encoder.encode(text)

    print("text_features.shape:", text_features.shape)
    text_features = text_features.reshape(1, -1)

  # Resize or pad text_features to match the dimensionality of self.d
    if text_features.shape[1] != self.d:
      if text_features.shape[1] < self.d:
          text_features = np.pad(text_features, ((0, 0), (0, self.d - text_features.shape[1])), 'constant')
      else:
          text_features = text_features[:, :self.d]
          pass

    assert text_features.shape == (1, self.d), f"Query features shape {text_features.shape} do not match expected shape (1, {self.d})"

    # Perform the search using Faiss
    scores, idx_image = self.index.search(text_features, k=k)

    # Map indices to result strings
    result_strings = list(map(lambda idx: self.dict_json[idx] if 0 <= idx < len(self.dict_json) else None, idx_image[-1]))
    return result_strings
