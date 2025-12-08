from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .viclip import ViCLIP
import requests
import os

def download_file(url, local_path):
    """
    Downloads a file from the specified URL to the given local path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code.
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded file to {local_path}")

def get_viclip(size='l', checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth")

    # If the pretrained file is not found at the specified path, download it.
    if not os.path.exists(checkpoint_path):
        download_url = "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"
        print(f"Pretrained file not found at {checkpoint_path}. Downloading from {download_url}...")
        download_file(download_url, checkpoint_path)
    
    tokenizer = _Tokenizer()
    vclip = ViCLIP(tokenizer=tokenizer, size=size, pretrain=checkpoint_path)
    m = {'viclip': vclip, 'tokenizer': tokenizer}
    
    return m
