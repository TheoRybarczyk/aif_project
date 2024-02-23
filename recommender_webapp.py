import argparse
from PIL import Image
import numpy as np
import requests
import json
import gradio as gr

from embedders import ImageEmbedder, TextBertEmbedder, TextBOWEmbedder, StemTokenizer

TXT_METHODS = ["Bag Of Word", "DistilBert"]

img_embedder = ImageEmbedder()
bow_embedder = TextBOWEmbedder()
bert_embedder = TextBertEmbedder()

def recommend_by_img(img: np.ndarray | None = None,
                     k: int = 5) -> list[Image.Image]:
    """Recommend images from an input image.

    Args:
        img (np.ndarray | None, optional): image array. Defaults to None.
        k (int, optional): number of recommendations. Defaults to 5.

    Returns:
        list[Image.Image]: list of recommended images
    """
    global api_adress
    
    if img is None:
        return []
    img = Image.fromarray(img.astype('uint8')) # type: ignore
    address = "http://" + api_address + "/predict_image"
    paths = _embed_send_and_receive(img_embedder, address, img, k)
    imgs_recom = [Image.open(p) for p in paths]
    return imgs_recom

def recommend_by_txt(txt: str | None = None,
                     k: int = 5,
                     method: str = TXT_METHODS[0]) -> str:
    """Recommend movies from a description.

    Args:
        txt (str | None, optional): input text description. Defaults to None.
        k (int, optional): number of recommendations.. Defaults to 5.
        method (str, optional): embedder method. Defaults to TXT_METHODS[0].

    Returns:
        str: recommended movies as a single string
    """
    global api_adress
    
    if txt is None:
        return ""
    if method == TXT_METHODS[0]: # BOW
        address = "http://" + api_address + "/predict_bow"
        titles_overviews = _embed_send_and_receive(bow_embedder,
                                                   address,
                                                   txt,
                                                   k)
    elif method == TXT_METHODS[1]: # BERT
        address = "http://" + api_address + "/predict_bert"
        titles_overviews = _embed_send_and_receive(bert_embedder,
                                                   address,
                                                   txt,
                                                   k)
    else:
        print(f"Invalid text embedding method {method}")
        return ""
    txt_recom = ""
    for ttl, ovw in titles_overviews:
        txt_recom += f"===== {ttl} =====\n{ovw}\n"
    return txt_recom

def _embed_send_and_receive(embedder, address, data, k) -> list:
    global api_adress
    
    try:
        emb = embedder.embeddings(data)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return []
    message = json.dumps({"emb": emb, "k": k})
    # Send request to the API
    response = requests.post(address, data=message)
    # Get response
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        print(f"Could not get a valid response from API "
              "({response.status_code})")
        return []

if __name__=='__main__':
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--docker", help="Docker mode", action="store_true")
    args = argParser.parse_args()
    
    if args.docker:
        server_name = "0.0.0.0"
        server_port = 7860
        debug = False
        share = True
        api_address = "annoy-api:5000"
    else:
        server_name = "127.0.0.1"
        server_port = 7860
        debug = True
        share = True
        api_address = "127.0.0.1:5000"
        
    # Define two input components
    input_component1 = gr.Image()
    input_component2 = gr.Gallery()

    # Define two output components
    output_component1 = gr.Image()
    output_component2 = gr.Gallery()
    
    imgrec = gr.Interface(fn=recommend_by_img, 
                inputs=[gr.Image(),
                        gr.Number(value=5, minimum=1, maximum=42, precision=0)], 
                outputs=gr.Gallery(),
                description="Select or drop an image to get a recommended movie based on its poster.",
                )
    
    txtrec = gr.Interface(fn=recommend_by_txt, 
                inputs=[gr.Textbox(),
                        gr.Number(value=5, minimum=1, maximum=42, precision=0),
                        gr.Dropdown(choices=TXT_METHODS, value=TXT_METHODS[0])], 
                outputs=gr.Textbox(),
                description="Write a movie description to get some recommendations.",
                )
    
    rec = gr.TabbedInterface([imgrec, txtrec], ["Movie recommender by iamge", "Movie recommender by description"])
    rec.launch(server_name=server_name, server_port=server_port, debug=debug, share=share)


