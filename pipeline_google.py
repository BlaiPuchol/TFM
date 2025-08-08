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

# Import Google GenAI
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials

# Import the MTEvaluation class from my TFG
from mt_evaluation import MTEvaluation
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.metrics.ter import TER

# Dotenv variables
from dotenv import load_dotenv
load_dotenv()

lang_mapping = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ro": "Romanian",
}

# Ensure flags to use Vertex AI mode
os.environ["GOOGLE_GENERATION_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "erudite-coast-467916-t0"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

# Path environment
service_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
scopes = ["https://www.googleapis.com/auth/cloud-platform"]

creds = Credentials.from_service_account_file(service_path, scopes=scopes)

# Make a list of the number of images to evaluate
def get_image_numbers(n, file_list):
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
    parser.add_argument("--src_lang", type=str, default=None, help="Idioma de origen (source language)")
    parser.add_argument("--tgt_lang", type=str, default=None, help="Idioma de destino (target language)")
    parser.add_argument("--img_src_dir", type=str, default=None, help="Directorio de imágenes en idioma origen")
    parser.add_argument("--img_tgt_dir", type=str, default=None, help="Directorio de imágenes en idioma destino")
    parser.add_argument("--corpus_src", type=str, default=None, help="Archivo corpus en idioma origen")
    parser.add_argument("--corpus_tgt", type=str, default=None, help="Archivo corpus en idioma destino")
    parser.add_argument("-n", type=int, default=None, help="Número de imágenes a evaluar")
    parser.add_argument("--start_index", type=int, default=None, help="Índice de inicio para la selección de imágenes")
    parser.add_argument("--end_index", type=int, default=None, help="Índice de fin para la selección de imágenes")
    parser.add_argument("--shuffle", action="store_true", help="Mezclar las imágenes antes de la traducción")
    parser.add_argument("--save_trans", action="store_true", help="Guardar traducciones en disco")
    parser.add_argument("--trans_folder", type=str, default="translations", help="Carpeta para guardar traducciones")
    parser.add_argument("--save_eval", action="store_true", help="Guardar resultados de evaluación")
    parser.add_argument("--print_trans", action="store_true", help="Imprimir traducciones en consola")
    parser.add_argument("--print_results", action="store_true", help="Imprimir resultados de evaluación en consola")
    parser.add_argument("--lowercase_source", action="store_true", help="Pasar a minúsculas las frases del archivo fuente")
    parser.add_argument("--lowercase_trans", action="store_true", help="Pasar a minúsculas las frases del archivo de traducción")
    parser.add_argument("--lowercase_reference", action="store_true", help="Pasar a minúsculas las frases del archivo referencia")
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
        'gemini-2.5-flash': 'gemini-2.5-flash',
    }

    # Create an instance of the MTEvaluation class
    mt_eval = MTEvaluation(source_lang, target_lang, engines=engines, source=CORPUS_SRC, references=CORPUS_TGT)
    
    # Number the images to evaluate
    file_list = os.listdir(IMAGES_SRC)
    if args.shuffle:
        shuffle(file_list)

    if args.start_index is not None and args.end_index is not None:
        file_list = file_list[args.start_index:args.end_index]
        args.n = len(file_list)
    elif args.n is None:
        # If no number is specified, use all images in the source directory
        args.n = len(file_list)

    # If a number is specified, use that number of images
    numbers = get_image_numbers(args.n, file_list)

    # Get image paths
    images_paths_en = get_image_paths(IMAGES_SRC, numbers)
    images_paths_de = get_image_paths(IMAGES_TGT, numbers)
    print(f"Selected {len(images_paths_en)} images from {IMAGES_SRC}")
    print(f"Selected {len(images_paths_de)} images from {IMAGES_TGT}")

    # Read the images and run OCR on them with tqdm
    total_time = 0
    results = {}
    prompt = f"First, extract the {lang_mapping[source_lang].lower()} text from this image. Then, translate the extracted text to {lang_mapping[target_lang]}. Provide only the final {lang_mapping[target_lang]} translation."
    for engine_name, engine in mt_eval.engines.items():

        client = genai.Client(
            vertexai=True,
            project="erudite-coast-467916-t0",
            location="global",
            credentials=creds,  # Use the credentials from the service account
        )

        model = engine
        print(f"Using model: {model}")
        results[engine_name] = []

        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 65535,
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
            thinking_config=types.ThinkingConfig(
            thinking_budget=0,
            ),
        )

        # Open files for saving translations and indices if needed
        trans_file = None
        indices_file = None
        if args.save_trans:
            output_file = os.path.join(args.trans_folder, f"trans_{engine_name}_{source_lang}_{target_lang}.txt")
            trans_file = open(output_file, 'w')
            indices_output_file = os.path.join(args.trans_folder, f"image_indices_{engine_name}_{source_lang}_{target_lang}.txt")
            indices_file = open(indices_output_file, 'w')

        for n, path in tqdm(images_paths_en.items(), desc="Translating images using the engine"):
            with open(path, "rb") as img_file:
                img_bytes = img_file.read()
            msg_img = types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg",
            )
            msg_text = types.Part.from_text(
                text=prompt,
            )
            contents = types.Content(
                role="user",
                parts=[
                    msg_img,
                    msg_text
                ]
            )
            translation = ""
            retries = 5
            delay = 5  # seconds
            while retries > 0:
                try:
                    translation_stream = client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    )
                    temp_translation = ""
                    for chunk in translation_stream:
                        if chunk.text:
                            result = chunk.text
                            if result:
                                temp_translation += result
                    translation = temp_translation
                    break  # Success
                except Exception as e:
                    print(f"\nAn error occurred for image {n}: {e}")
                    retries -= 1
                    if retries > 0:
                        print(f"Retrying in {delay} seconds... ({retries} retries left)")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        print(f"Failed to process image {n} after multiple retries.")
                        translation = "ERROR: FAILED TO TRANSLATE"
            
            clean_translation = translation.replace('\n', ' ').strip()
            results[engine_name].append(clean_translation)

            if args.save_trans:
                # Save the translation and index to files
                trans_file.write(f"{clean_translation.lower() if args.lowercase_trans else clean_translation}\n")
                indices_file.write(f"{n}\n")

        if args.save_trans:
            trans_file.close()
            indices_file.close()
            print(f"Translations saved to {os.path.join(args.trans_folder, f'trans_{engine_name}_{source_lang}_{target_lang}.txt')}")
            print(f"Image indices saved to {os.path.join(args.trans_folder, f'image_indices_{engine_name}_{source_lang}_{target_lang}.txt')}")
                        

    # Prepare sentences for evaluation
    translated_sentences = {}
    source_sentences = {}
    reference_sentences = {}
    for engine_name, translations in results.items():
        translated_sentences[engine_name] = []
        source_sentences[engine_name] = []
        reference_sentences[engine_name] = []
         # Print the translations
        if args.print_trans:
            print(f"Translations for {engine_name}:")
        for i, (n, translation) in enumerate(zip(images_paths_en.keys(), translations)):
            if i >= args.n:
                break
            translated_sentences[engine_name].append(translation.lower() if args.lowercase_trans else translation)
            source_sentences[engine_name].append(mt_eval.get_source()[int(n)].lower() if args.lowercase_source else mt_eval.get_source()[int(n)])
            reference_sentences[engine_name].append(mt_eval.get_references()[int(n)].lower() if args.lowercase_reference else mt_eval.get_references()[int(n)])
            if args.print_trans:
                print(f"Image {i}: {translation}")
        print()

        # Evaluate the results of each model
        bleu = BLEU(effective_order=True)
        chrf = CHRF(word_order=2)
        ter = TER()

        bleu_score = bleu.corpus_score(translated_sentences[engine_name], [reference_sentences[engine_name]]).score
        chrf_score = chrf.corpus_score(translated_sentences[engine_name], [reference_sentences[engine_name]]).score
        ter_score = ter.corpus_score(translated_sentences[engine_name], [reference_sentences[engine_name]]).score

        if args.print_results:
            print(f"Evaluation for {len(translated_sentences[engine_name])} sentences:")
            print(f"BLEU: {bleu_score:f}")
            print(f"CHRF: {chrf_score:f}")
            print(f"TER: {ter_score:f}")

        if args.save_eval:
            # Save the evaluation results to a file
            output_file = os.path.join(args.trans_folder, f"eval_results_{source_lang}_{target_lang}.txt")
            with open(output_file, 'w') as f:
                f.write(f"BLEU: {bleu_score:f}\n")
                f.write(f"CHRF: {chrf_score:f}\n")
                f.write(f"TER: {ter_score:f}\n")
            print(f"Evaluation results saved to {output_file}")