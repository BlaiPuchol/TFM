import argparse
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.metrics.ter import TER

def main():
    parser = argparse.ArgumentParser(description="Evaluate MT between translation, OCR, and reference files.")
    parser.add_argument("--trans_file", type=str, required=True, help="Translation file with one sentence per line (aligned with OCR file)")
    parser.add_argument("--ocr_file", type=str, required=True, help="OCR file (with format 'Image {n}:\\tsentence')")
    parser.add_argument("--ref_file", type=str, required=True, help="Reference file with one sentence per line")
    parser.add_argument("-n", type=int, default=None, help="Number of sentences to evaluate")
    parser.add_argument("--lowercase_trans", action="store_true", help="Lowercase translation sentences")
    parser.add_argument("--lowercase_reference", action="store_true", help="Lowercase reference sentences")
    args = parser.parse_args()

    translations = []
    image_indices = []

    # Read OCR file to get image indices
    with open(args.ocr_file, 'r', encoding='utf-8') as f:
        for line in f:
            if args.n is not None and len(image_indices) >= args.n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                header, _ = line.split(':\t', 1)
                img_num_str = header.split(' ')[1]
                image_indices.append(int(img_num_str))
            except (ValueError, IndexError):
                print(f"Skipping malformed line in OCR file: {line}")
                continue

    # Read translations (already aligned with OCR file)
    with open(args.trans_file, 'r', encoding='utf-8') as f:
        for line in f:
            if args.n is not None and len(translations) >= args.n:
                break
            line = line.strip()
            if not line:
                continue
            if args.lowercase_trans:
                line = line.lower()
            translations.append(line)

    # Read reference file as plain sentences
    reference_sentences = []
    with open(args.ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if args.lowercase_reference:
                line = line.lower()
            reference_sentences.append(line)

    # Align references using image_indices from OCR file
    references = []
    for idx in image_indices:
        if idx < len(reference_sentences):
            references.append(reference_sentences[idx])
        else:
            references.append("")

    # Compute metrics using sacrebleu
    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=2)
    ter = TER()

    bleu_score = bleu.corpus_score(translations, [references]).score
    chrf_score = chrf.corpus_score(translations, [references]).score
    ter_score = ter.corpus_score(translations, [references]).score

    print(f"Evaluation for {len(translations)} sentences:")
    print(f"BLEU: {bleu_score:f}")
    print(f"CHRF: {chrf_score:f}")
    print(f"TER: {ter_score:f}")

if __name__ == "__main__":
    main()