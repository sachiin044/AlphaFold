import requests
import os
import urllib3
from urllib.parse import urlparse

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_URL = "https://scop.berkeley.edu/downloads/scopeseq-2.07/astral-scopedom-seqres-gd-all-2.07-stable.fa"
DEFAULT_FOLDER = r"C:\Users\aryan\OneDrive\Desktop\hack-nation\hacknation\src\data"

# DEFAULT_URL = "https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.07-stable.txt"
# DEFAULT_FOLDER = r"C:\Users\aryan\OneDrive\Desktop\hack-nation\hacknation\src\data"

def download_file(url, folder_path):
    try:
        response = requests.get(url, stream=True, verify=False)  # SSL check disabled
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url} : {e}")
        return False

    filename = os.path.basename(urlparse(url).path)
    save_path = os.path.join(folder_path, filename)
    os.makedirs(folder_path, exist_ok=True)

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded successfully and saved to {save_path}")
    return True

if __name__ == "__main__":
    print(f"Downloading from: {DEFAULT_URL}")
    print(f"Saving to folder: {DEFAULT_FOLDER}")
    download_file(DEFAULT_URL, DEFAULT_FOLDER)
