# MT translation and evaluation using OCR CSV files

import argparse
import csv
import json
import os

from mt_evaluation import MTEvaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT translation and evaluation using OCR CSV files or gold-aligned text files.")
    parser.add_argument("--src_lang", type=str, required=True, help="Source language code (e.g. de, en, fr)")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code (e.g. de, en, fr)")
    parser.add_argument("--ocr_csv", type=str, help="CSV file with image_id and transcription columns (for OCR mode)")
    parser.add_argument("--src_file", type=str, help="Plain text file with source sentences, one per line (for gold mode)")
    parser.add_argument("--ref_file", type=str, required=True, help="Reference file (one sentence per line)")
    parser.add_argument("-n", type=int, default=None, help="Maximum number of sentences to translate")
    parser.add_argument("--save_trans", action="store_true", help="Save translations to disk")
    parser.add_argument("--trans_folder", type=str, default="translations", help="Folder to save translations")
    parser.add_argument("--save_eval", action="store_true", help="Save evaluation results to disk")
    parser.add_argument("--eval_folder", type=str, default="evaluations", help="Folder to save evaluation results")
    parser.add_argument("--print_trans", action="store_true", help="Print translations to console")
    parser.add_argument("--print_results", action="store_true", help="Print evaluation results to console")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase source, reference, and translations")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of in-context examples for prompting (0 = zero-shot)")
    parser.add_argument("--shots_seed", type=int, default=13, help="Random seed for selecting in-context examples")
    parser.add_argument("--gold", action="store_true", help="Use gold-aligned plain text files as source (no image_id alignment)")
    args = parser.parse_args()

    # Load reference file (one sentence per line)
    with open(args.ref_file, 'r', encoding='utf-8') as f:
        all_references = [line.rstrip('\n') for line in f]

    source_sentences = []
    reference_sentences = []
    image_ids = []

    if args.gold:
        # Gold mode: src_file and ref_file are aligned, use line index
        if not args.src_file:
            raise ValueError("--src_file must be provided when using --gold mode.")
        with open(args.src_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if args.n is not None and len(source_sentences) >= args.n:
                    break
                src = line.rstrip('\n')
                ref = all_references[idx] if idx < len(all_references) else ""
                image_ids.append(idx)
                source_sentences.append(src.lower() if args.lowercase else src)
                reference_sentences.append(ref.lower() if args.lowercase else ref)
        print(f"Loaded {len(source_sentences)} sentences from {args.src_file} (gold mode)")
    else:
        # OCR mode: use CSV with image_id
        if not args.ocr_csv:
            raise ValueError("--ocr_csv must be provided unless using --gold mode.")
        with open(args.ocr_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.n is not None and len(source_sentences) >= args.n:
                    break
                img_id = int(row['image_id'])
                transcription = row['transcription'].strip()
                if not transcription:
                    continue
                if img_id >= len(all_references):
                    print(f"Warning: image_id {img_id} out of range ({len(all_references)} lines), skipping")
                    continue
                ref = all_references[img_id]
                image_ids.append(img_id)
                source_sentences.append(transcription.lower() if args.lowercase else transcription)
                reference_sentences.append(ref.lower() if args.lowercase else ref)
        print(f"Loaded {len(source_sentences)} sentences from {args.ocr_csv} (ocr mode)")

    # Engines to evaluate
    engines = {
        'LLaMA-3.2-1B': 'meta-llama/Llama-3.2-1B',
        'LLaMA-3.2-3B': 'meta-llama/Llama-3.2-3B',
        'LLaMA-3.1-8B': 'meta-llama/Llama-3.1-8B',
        'LLaMA-3.2-1B-Instruct': 'meta-llama/Llama-3.2-1B-Instruct',
        'LLaMA-3.2-3B-Instruct': 'meta-llama/Llama-3.2-3B-Instruct',
        'LLaMA-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
        'M2M100': 'facebook/m2m100_1.2B',
        'EuroLLM-1.7B': 'utter-project/EuroLLM-1.7B',
        'EuroLLM-9B-2512': 'utter-project/EuroLLM-9B-2512',
        'EuroLLM-22B-2512': 'utter-project/EuroLLM-22B-2512',
        'EuroLLM-1.7B-Instruct': 'utter-project/EuroLLM-1.7B-Instruct',
        'EuroLLM-9B-2512-Instruct': 'utter-project/EuroLLM-9B-2512-Instruct',
        'EuroLLM-22B-2512-Instruct': 'utter-project/EuroLLM-22B-2512-Instruct',
    }

    direction = f"{args.src_lang}_{args.tgt_lang}"
    mt_eval = MTEvaluation(
        args.src_lang, args.tgt_lang,
        engines=engines,
        dataset=f"ocr_onnxtr_{direction}",
    )
    mt_eval.set_source_from_list(source_sentences)
    mt_eval.set_references_from_list(reference_sentences)

    # Translate
    mt_eval.translate(
        save=args.save_trans,
        folder=args.trans_folder,
        lowercase=args.lowercase,
        num_shots=args.num_shots,
        shots_seed=args.shots_seed,
    )

    # Print translations
    if args.print_trans:
        for engine, corpus in mt_eval.mt.items():
            print(f"\nTranslations for {engine}:")
            for img_id, translation in zip(image_ids, corpus.segments()):
                print(f"  Image {img_id}: {translation}")

    # Evaluate and optionally save full report
    mt_eval.full_report(save=args.save_eval, folder=args.eval_folder, to_json=True)

    # Print corpus-level results if requested
    if args.print_results:
        mt_eval.corpus_evaluate(save=False, to_json=False, print_results=True)
