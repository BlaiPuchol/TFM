# Evaluación de traducción de imágenes

# Imports
import argparse
from random import shuffle
import os

# Hugging Face login

from huggingface_hub import login

# Login to Hugging Face Hub
# login()

# Import the MTEvaluation class from my TFG
from mt_evaluation import MTEvaluation

# Makea list of the number of images to evaluate
def get_image_numbers(n, file_list, shuf=False):
    if shuf:
        shuffle(file_list)
    image_numbers = []
    for i, filename in enumerate(file_list):
        if i >= n:
            break
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_numbers.append(filename.split('.')[0])
    return image_numbers

# Make a list of image paths and save it in a dict, where the keys are the image names and the values are the image paths
def get_image_paths(image_dir, numbers):
    image_paths = {}
    file_list = os.listdir(image_dir)
    for filename in file_list:
        if filename.split('.')[0] in numbers:
            image_paths[filename.split('.')[0]] = os.path.join(image_dir, filename)
    return image_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de traducción de imágenes con OCR y MT.")
    parser.add_argument("--src_lang", type=str, default="en", help="Idioma de origen (source language)")
    parser.add_argument("--tgt_lang", type=str, default="de", help="Idioma de destino (target language)")
    parser.add_argument("--input_file", type=str, default=None, help="Archivo de texto con las frases a traducir.")
    parser.add_argument("--corpus_src", type=str, default=None, help="Archivo corpus en idioma origen")
    parser.add_argument("--corpus_tgt", type=str, default=None, help="Archivo corpus en idioma destino")
    parser.add_argument("--n", type=int, default=None, help="Número de imágenes a evaluar")
    parser.add_argument("--save_trans", action="store_true", help="Guardar traducciones en disco")
    parser.add_argument("--trans_folder", type=str, default="translations", help="Carpeta para guardar traducciones")
    parser.add_argument("--save_eval", action="store_true", help="Guardar resultados de evaluación")
    parser.add_argument("--print_trans", action="store_true", help="Imprimir traducciones en consola")
    parser.add_argument("--print_results", action="store_true", help="Imprimir resultados de evaluación")
    args = parser.parse_args()

    # Set language pairs
    source_lang = args.src_lang
    target_lang = args.tgt_lang

    # Dataset directories and files
    INPUT_FILE = args.input_file
    CORPUS_SRC = args.corpus_src
    CORPUS_TGT = args.corpus_tgt

    # Engines to evaluate
    engines = {
        'euroLLM': 'utter-project/EuroLLM-9B',
        'LLaMA': 'meta-llama/Llama-3.2-1B-Instruct',
        'M2M100': 'facebook/m2m100_1.2B',
    }

    # Create an instance of the MTEvaluation class
    mt_eval = MTEvaluation(source_lang, target_lang, engines=engines, source=CORPUS_SRC, references=CORPUS_TGT)

    # Read sentences from input file
    source_sentences = []
    reference_sentences = []
    image_numbers = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if args.n is not None and len(source_sentences) >= args.n:
                break
            line = line.strip()
            if not line:
                continue
            
            try:
                # Format: "Image {n}:\tsome text"
                header, sentence = line.split(':\t', 1)
                img_num_str = header.split(' ')[1]
                image_numbers.append(img_num_str)
                source_sentences.append(sentence)
                reference_sentences.append(mt_eval.get_references()[int(img_num_str)])
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line: {line}")
                continue

    print(f"Read {len(source_sentences)} sentences from {INPUT_FILE}")

    # Set new source and reference sentences for the MTEvaluation class
    mt_eval.set_source_from_list(source_sentences)
    mt_eval.set_references_from_list(reference_sentences)

    # Translate the source sentences using the engines specified in the MTEvaluation class
    mt_eval.translate(save=args.save_trans, folder=args.trans_folder)

    # Print the translations
    if args.print_trans:    
        for engine, translations in mt_eval.mt.items():
            print(f"Translations for {engine}:")
            for i, translation in zip(image_numbers, translations.segments()):
                print(f"Image {i}: {translation}")
            print()

    # Evaluate the results of each model
    mt_eval.corpus_evaluate(to_json=False, save=args.save_eval, print_results=args.print_results)