import argparse

# Import necessary libraries for evaluation metrics
from jiwer import wer, cer, mer, wil, wip, Compose, RemovePunctuation, ReduceToListOfListOfWords, ReduceToListOfListOfChars
from tqdm import tqdm

def parse_lines(file_path):
    texts = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Expecting format: "Image {n}: some text"
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                header = parts[0]
                text = parts[1].strip()
                try:
                    img_num = header.split(' ')[1]
                    texts[img_num] = text
                except (ValueError, IndexError):
                    print(f"Skipping malformed line (header): {line.strip()}")
            else:
                if line.strip(): # Avoid warning for empty lines
                    print(f"Skipping malformed line (no colon): {line.strip()}")
    return texts

def parse_ref_lines(file_path):
    texts = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            texts[str(i)] = line.strip()
    return texts

def main():
    parser = argparse.ArgumentParser(description="Compute OCR metrics (WER, CER, etc) using jiwer.")
    parser.add_argument('--hyp', required=True, help='Path to hypothesis file (recognized text)')
    parser.add_argument('--ref', required=True, help='Path to reference file (ground truth text)')
    args = parser.parse_args()

    hyp_texts_dict = parse_lines(args.hyp)
    ref_texts_dict = parse_ref_lines(args.ref)

    hyp_texts = []
    ref_texts = []

    for img_num, hyp_text in tqdm(hyp_texts_dict.items(), desc="Aligning texts"):
        if img_num in ref_texts_dict:
            hyp_texts.append(hyp_text)
            ref_texts.append(ref_texts_dict[img_num])
        else:
            print(f"Warning: No reference found for image {img_num}. Skipping.")

    if not hyp_texts:
        print("No matching lines found between hypothesis and reference files. Exiting.")
        return

    # Define transformations for metrics
    wer_transform = Compose([RemovePunctuation(), ReduceToListOfListOfWords()])
    cer_transform = Compose([RemovePunctuation(), ReduceToListOfListOfChars()])
    # Compute metrics
    print("Computing metrics...")
    print(f"Number of sentences: {len(hyp_texts)}")
    wer_metric = wer(ref_texts, hyp_texts, reference_transform=wer_transform, hypothesis_transform=wer_transform)
    cer_metric = cer(ref_texts, hyp_texts, reference_transform=cer_transform, hypothesis_transform=cer_transform)
    mer_metric = mer(ref_texts, hyp_texts, reference_transform=wer_transform, hypothesis_transform=wer_transform)
    wil_metric = wil(ref_texts, hyp_texts, reference_transform=wer_transform, hypothesis_transform=wer_transform)
    wip_metric = wip(ref_texts, hyp_texts, reference_transform=wer_transform, hypothesis_transform=wer_transform)

    print(f"WER: {wer_metric:.2%}")
    print(f"CER: {cer_metric:.2%}")
    print(f"MER: {mer_metric:.2%}")
    print(f"WIL: {wil_metric:.2%}")
    print(f"WIP: {wip_metric:.2%}")

if __name__ == "__main__":
    main()