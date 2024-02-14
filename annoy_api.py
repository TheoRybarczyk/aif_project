import argparse
from pathlib import Path
from typing import Any
import pandas as pd
import json
from annoy import AnnoyIndex
from flask import Flask, Response, request, Request

POSTER_PATHS = Path("data/posters_paths.txt")
ANNOYDB_POSTERS_DIM = 576
ANNOYDB_POSTERS_PATH = Path("data/posters.ann")
OVERVIEWS_PATH = Path("data/movies_overviews.csv")
ANNOYDB_BOW_DIM = 1000
ANNOYDB_BOW_PATH = Path("data/rec_overviews_bow1000_10trees.ann")
ANNOYDB_BERT_DIM = 768
ANNOYDB_BERT_PATH = Path("data/rec_overviews_bert.ann")

class AnnoyDB():
    def __init__(self,                
                 embeddings_dim: int,
                 annoydb_path: Path,
                 mapped_objects: list[Any]) -> None:
        
        self.annoy_index = AnnoyIndex(embeddings_dim, 'angular')
        self.annoy_index.load(str(annoydb_path))
        self.mapped_objects = mapped_objects
        
    def knn_obj(self, query: list, k: int = 5) -> list[Any]:        
        return [self.mapped_objects[i] for i in self.knn_idx(query, k)]
  
    def knn_idx(self, query: list, k: int = 5) -> list[int]:        
        return self.annoy_index.get_nns_by_vector(query, k) # type: ignore

# posters
data_folder = POSTER_PATHS.parent
with POSTER_PATHS.open("r") as file:
    posters_paths = [str(Path(data_folder, line.rstrip())) for line in file]

anndb_posters = AnnoyDB(ANNOYDB_POSTERS_DIM,
                        ANNOYDB_POSTERS_PATH,
                        posters_paths)

# overviews
overviews = pd.read_csv(OVERVIEWS_PATH, usecols=["title", "overview"])
overviews = list(zip(overviews["title"], overviews["overview"]))

anndb_bow = AnnoyDB(ANNOYDB_BOW_DIM,
                    ANNOYDB_BOW_PATH,
                    overviews)

anndb_bert = AnnoyDB(ANNOYDB_BERT_DIM,
                     ANNOYDB_BERT_PATH,
                     overviews)

# app
app = Flask(__name__)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    return _predict(request, anndb_posters)

@app.route('/predict_bow', methods=['POST'])
def predict_bow():
    return _predict(request, anndb_bow)

@app.route('/predict_bert', methods=['POST'])
def predict_bert():
    return _predict(request, anndb_bert)

def _predict(req: Request, anndb: AnnoyDB):
    data = req.data
    query_vector = json.loads(data)
    if not isinstance(query_vector, dict):
        print("Incoming request does not have expected format.")
        return Response()
    emb = query_vector.get("emb")
    k = query_vector.get("k")
    if (emb is None) or (k is None):
        print("Could not find embeddings or k argument.")
        return Response()

    # Make prediction
    results = anndb.knn_obj(emb, k=k)
    results_json = json.dumps(results)

    return Response(results_json)
    
if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--docker", help="Docker mode", action="store_true")
    args = argParser.parse_args()
    
    if args.docker:
        host = "0.0.0.0"
        debug = False
    else:
        host = "127.0.0.1"
        debug = True

    app.run(host=host, debug=debug)
    
# def predict_posters():
#     data = request.data
#     query_vector = json.loads(data)

#     # Make prediction
#     results = anndb_posters.knn_obj(query_vector, k=5)
#     results_json = json.dumps(results)

#     return Response(results_json)