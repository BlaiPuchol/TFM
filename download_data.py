import os
import zipfile
import requests
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    total = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        leave=True
    ) as pbar:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        with tqdm(total=len(members), desc=f"Extracting {os.path.basename(zip_path)}", unit="file", leave=True) as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)

if __name__ == "__main__":
    # List of (Google Drive file id, output filename)
    files = [
        ("1YmPcGZOinxqUl-ggotYPLtUHNnnUJnuW", "de-en-images.zip"),
        ("1Su5MjtMCbEieerq7lN-LlzT46zg2ty7E", "en-de.zip"),
        ("1RyKp4Fz1rgHewXVjulCKTUf_F350L4il", "fr-en-images.zip"),
        ("1H32hzLtXbyLMeKeBUgDDnHzvULNeSzlN", "fr-en.zip"),
    ]

    download_dir = "downloads"
    extract_dir = "extracted"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    for file_id, filename in files:
        zip_path = os.path.join(download_dir, filename)
        print(f"Downloading {filename}...")
        download_file_from_google_drive(file_id, zip_path)
        print(f"Extracting {filename}...")
        unzip_file(zip_path, extract_dir)
        print(f"Done with {filename}.")

    print("All files downloaded and extracted.")