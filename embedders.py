from pathlib import Path
from PIL import Image
import numpy as np
import joblib

import re
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

MOBILENETDICT = Path("data/mobilenet_dict")
BOW_TFIDF = Path("data/tfidf_1000.sav") 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nltk.download('punkt')
nltk.download('stopwords')

class ImageEmbedder:
    def __init__(self,
                 device: torch.device = DEVICE,
                 model_dict_path: Path = MOBILENETDICT
                 ) -> None:

        TRANFORM_MEAN = [0.485, 0.456, 0.406]
        TRANSFORM_STD = [0.229, 0.224, 0.225]
        TRANSFORM_SIZE = (224,224)
        
        # initialize model
        mobilenet = models.mobilenet_v3_small()
        model = nn.Sequential(
            mobilenet.features,
            mobilenet.avgpool,
            nn.Flatten()
        ).to(device)     
        model.load_state_dict(torch.load(model_dict_path,
                                         map_location=device))
        model.eval()
        
        # image transformer
        normalize = transforms.Normalize(TRANFORM_MEAN, TRANSFORM_STD)
        transform = transforms.Compose([transforms.Resize(TRANSFORM_SIZE),
                                        transforms.ToTensor(),
                                        normalize])
    
        self.device = device
        self.model = model
        self.transform = transform
        
    def embeddings(self, img_pil: Image.Image) -> list:        
        with torch.no_grad():
            tensor = self.transform(img_pil).to(self.device) 
            tensor = tensor.unsqueeze(0) # add batch dimension
            embeddings = self.model(tensor)[0]

        return embeddings.cpu().tolist()
    

class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', "'"]
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens] 

    
class TextBOWEmbedder:
    def __init__(self, bow_tfidf_path: Path = BOW_TFIDF) -> None:
        
        with open(bow_tfidf_path, "rb") as file:
            tokenizer = joblib.load(file)
            tfidf = joblib.load(file)
        self.tokenizer = tokenizer
        self.tfidf = tfidf
        
    def embeddings(self, text: str) -> list:
        return np.squeeze(np.array(self.tfidf.transform([text]).todense())).tolist()
    
    
class TextBertEmbedder:
    def __init__(self,
                 device: torch.device = DEVICE
                 ) -> None:

        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        model.eval()
    
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        
    def embeddings(self, text: str) -> list:
        tokens = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            last_layer_cls = self.model(input_ids, attention_mask).last_hidden_state[:,0,:]
        return last_layer_cls.squeeze(0).squeeze(0).cpu().tolist()
