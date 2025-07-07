import os
import zipfile
from tqdm import tqdm
import gdown
import argparse

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        with tqdm(total=len(members), desc=f"Extracting {os.path.basename(zip_path)}", unit="file", leave=True) as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract datasets from Google Drive.")
    parser.add_argument("--download_dir", type=str, default="downloads", help="Directory to save downloaded zip files.")
    parser.add_argument("--extract_dir", type=str, default="extracted", help="Directory to extract files to.")
    args = parser.parse_args()

    # List of (Google Drive file id, output filename)
    files = [
        ("1YmPcGZOinxqUl-ggotYPLtUHNnnUJnuW", "de-en-images.zip"),
        ("1Su5MjtMCbEieerq7lN-LlzT46zg2ty7E", "en-de.zip"),
        ("1RyKp4Fz1rgHewXVjulCKTUf_F350L4il", "fr-en-images.zip"),
        ("1H32hzLtXbyLMeKeBUgDDnHzvULNeSzlN", "fr-en.zip"),
    ]

    download_dir = args.download_dir if args.download_dir else "downloads"
    extract_dir = args.extract_dir if args.extract_dir else "extracted"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    for file_id, filename in files:
        zip_path = os.path.join(download_dir, filename)
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)
        print(f"Extracting {filename}...")
        unzip_file(zip_path, extract_dir)
        print(f"Done with {filename}.")

    print("All files downloaded and extracted.")