# Evaluación de traducción de imágenes

# Imports
import argparse
import time
from random import shuffle
import os
from tqdm import tqdm

# Hugging Face login

from huggingface_hub import login

# Login to Hugging Face Hub
# login()

import matplotlib.pyplot as plt

# Import docTR libraries
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor


# Import necessary libraries for evaluation metrics
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.metrics.ter import TER

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
    parser.add_argument("--img_src_dir", type=str, default=None, help="Directorio de imágenes en idioma origen")
    parser.add_argument("--img_tgt_dir", type=str, default=None, help="Directorio de imágenes en idioma destino")
    parser.add_argument("--corpus_src", type=str, default=None, help="Archivo corpus en idioma origen")
    parser.add_argument("--corpus_tgt", type=str, default=None, help="Archivo corpus en idioma destino")
    parser.add_argument("--n", type=int, default=None, help="Número de imágenes a evaluar")
    parser.add_argument("--shuffle", action="store_true", help="Mezclar las imágenes antes de la traducción")
    parser.add_argument("--save_ocr", action="store_true", help="Guardar resultados de OCR")
    parser.add_argument("--save_trans", action="store_true", help="Guardar traducciones en disco")
    parser.add_argument("--trans_folder", type=str, default="translations", help="Carpeta para guardar traducciones")
    parser.add_argument("--save_eval", action="store_true", help="Guardar resultados de evaluación")
    parser.add_argument("--print_trans", action="store_true", help="Imprimir traducciones en consola")
    parser.add_argument("--print_results", action="store_true", help="Imprimir resultados de evaluación en consola")
    args = parser.parse_args()

    # Set language pairs
    source_lang = args.src_lang
    target_lang = args.tgt_lang

    # Dataset directories and files
    IMAGES_SRC = args.img_src_dir
    IMAGES_TGT = args.img_tgt_dir
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

    # Instantiate a pretrained model
    predictor = ocr_predictor('fast_base', # Text Detection
                              'crnn_vgg16_bn', # Text Recognition
                              # Document related parameters
                              assume_straight_pages=False,  # set to `False` if the pages are not straight (rotation, perspective, etc.) (default: True)
                              straighten_pages=False,  # set to `True` if the pages should be straightened before final processing (default: False)
                              export_as_straight_boxes=False,  # set to `True` if the boxes should be exported as if the pages were straight (default: False)
                              # Preprocessing related parameters
                              preserve_aspect_ratio=True,  # set to `False` if the aspect ratio should not be preserved (default: True)
                              symmetric_pad=True,  # set to `False` to disable symmetric padding (default: True)
                              # Additional parameters - meta information
                              detect_orientation=False,  # set to `True` if the orientation of the pages should be detected (default: False)
                              detect_language=False, # set to `True` if the language of the pages should be detected (default: False)
                              # Orientation specific parameters in combination with `assume_straight_pages=False` and/or `straighten_pages=True`
                              disable_crop_orientation=False,  # set to `True` if the crop orientation classification should be disabled (default: False)
                              disable_page_orientation=False,  # set to `True` if the general page orientation classification should be disabled (default: False)
                              # DocumentBuilder specific parameters
                              resolve_lines=True,  # whether words should be automatically grouped into lines (default: True)
                              resolve_blocks=False,  # whether lines should be automatically grouped into blocks (default: False)
                              paragraph_break=0.035,  # relative length of the minimum space separating paragraphs (default: 0.035)
                              )
    
    # Number the images to evaluate
    if args.n is None:
        # If no number is specified, use all images in the source directory
        args.n = len(os.listdir(IMAGES_SRC))

    # If a number is specified, use that number of images
    numbers = get_image_numbers(args.n, os.listdir(IMAGES_SRC), shuf=args.shuffle)

    # Get image paths
    images_paths_en = get_image_paths(IMAGES_SRC, numbers)
    images_paths_de = get_image_paths(IMAGES_TGT, numbers)
    print(f"Selected {len(images_paths_en)} images from {IMAGES_SRC}")
    print(f"Selected {len(images_paths_de)} images from {IMAGES_TGT}")

    # Read the images and run OCR on them with tqdm
    images_en = [images_paths_en[name] for name in images_paths_en.keys()]

    total_time = 0
    results = []
    for path in tqdm(images_en, desc="Making OCR to the images"):
        images = DocumentFile.from_images(path)
        start_time = time.time()
        results.append(predictor(images).pages[0])
        end_time = time.time()
        total_time += end_time - start_time

    print(f"Processed {len(results)} images in {total_time:.2f} seconds.")

    if args.save_ocr:
        # Save the OCR results to a file
        output_file = 'ocr_results.txt'
        with open(output_file, 'w') as f:
            for i, page in enumerate(results):
                f.write(f"Image {i+1}:\t")
                f.write(page.render().replace('\n', ' ') + '\n')
        print(f"OCR results saved to {output_file}")

    # Prepare sentences for evaluation
    extracted_sentences = []
    source_sentences = []
    reference_sentences = []
    for i, (n, page) in enumerate(zip(images_paths_en.keys(), results)):
        if i >= args.n:
            break
        extracted_sentences.append(page.render().replace('\n', ' '))
        source_sentences.append(mt_eval.get_source()[int(n)])
        reference_sentences.append(mt_eval.get_references()[int(n)])

    # Compute evaluation metrics
    bleu = BLEU()
    chrf = CHRF()
    ter = TER()
    bleu_score = bleu.corpus_score(extracted_sentences, [source_sentences])
    chrf_score = chrf.corpus_score(extracted_sentences, [source_sentences])
    ter_score = ter.corpus_score(extracted_sentences, [source_sentences])

    # Print evaluation results
    print(f"BLEU: {bleu_score.score:.2f}")
    print(f"CHRF: {chrf_score.score:.2f}")
    print(f"TER: {ter_score.score:.2f}")

    # Set new source and reference sentences for the MTEvaluation class
    mt_eval.set_source_from_list(extracted_sentences)
    mt_eval.set_references_from_list(reference_sentences)

    # Translate the source sentences using the engines specified in the MTEvaluation class
    mt_eval.translate(save=args.save_trans, folder=args.trans_folder)

    # Print the translations
    if args.print_trans:    
        for engine, translations in mt_eval.mt.items():
            print(f"Translations for {engine}:")
            for i, translation in zip(images_paths_en.keys(), translations.segments()):
                print(f"Image {i}: {translation}")
            print()

    # Evaluate the results of each model
    mt_eval.corpus_evaluate(to_json=False, save=args.save_eval)