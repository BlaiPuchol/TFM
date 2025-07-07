# Evaluación de traducción de imágenes


# Hugging Face login

from huggingface_hub import login

# Login to Hugging Face Hub
# login()

# Imports

%matplotlib inline
import matplotlib.pyplot as plt

# Import docTR libraries
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor

# Import the MTEvaluation class from my TFG
from mt_evaluation import MTEvaluation

# Datos

# Obtain the data

# Set language pairs
source_lang = 'en'
target_lang = 'de'

# Main directory of the dataset
DATASET_DIR = '/home/blai/TFM/Datasets/IWSLT/'

# Set the path to the corpus files
IMAGES_EN = DATASET_DIR + 'iwslt14.de-en-images/test_en/'
IMAGES_DE = DATASET_DIR + 'iwslt14.de-en-images/test_de/'
CORPUS_EN = DATASET_DIR + 'iwslt14.de-en/test.en'
CORPUS_DE = DATASET_DIR + 'iwslt14.de-en/test.de'

# Engines to evaluate
engines = {
    'euroLLM': 'utter-project/EuroLLM-1.7B',
    'LLaMA': 'meta-llama/Llama-3.2-1B-Instruct',
    'M2M100': 'facebook/m2m100_1.2B',
}

# Create an instance of the MTEvaluation class
mt_eval = MTEvaluation(source_lang, target_lang, engines=engines, source=CORPUS_EN, references=CORPUS_DE)

# Extracción del texto de las imagenes

# %%
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

# %%
from random import shuffle
import os

# Number the images to evaluate
N = 100

# Makea list of the number of images to evaluate
def get_image_numbers(n):
    file_list = os.listdir(IMAGES_EN)
    shuffle(file_list)
    image_numbers = []
    for i, filename in enumerate(file_list):
        if i >= n:
            break
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_numbers.append(filename.split('.')[0])
    return image_numbers

numbers = get_image_numbers(N)

# Make a list of image paths and save it in a dict, where the keys are the image names and the values are the image paths
def get_image_paths(image_dir, numbers):
    image_paths = {}
    file_list = os.listdir(image_dir)
    for filename in file_list:
        if filename.split('.')[0] in numbers:
            image_paths[filename.split('.')[0]] = os.path.join(image_dir, filename)
    return image_paths

images_paths_en = {}
image_paths_en = get_image_paths(IMAGES_EN, numbers)
print(f"Selected {len(image_paths_en)} images from {IMAGES_EN}")

images_paths_de = {}
image_paths_de = get_image_paths(IMAGES_DE, numbers)
print(f"Selected {len(image_paths_de)} images from {IMAGES_DE}")

# Read the images and run OCR on them
images_en = DocumentFile.from_images([image_paths_en[name] for name in image_paths_en.keys()])

# %%
import time

# Process the images with the OCR predictor
start_time = time.time()
results = predictor(images_en)
end_time = time.time()
print(f"Processed {len(results.pages)} images in {end_time - start_time:.2f} seconds.")

# Print the results of the first N images
i = 0
for n, page in zip(image_paths_en.keys(), results.pages):
    if i >= 10:
        break
    print(f"Image {n}: {image_paths_en[n]}")
    print("Text:", page.render().replace('\n', ' '))
    print()
    i += 1

# %% [markdown]
# ## Comprobar texto extraido con el del dataset

# %%
# Import necessary libraries for evaluation metrics
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.metrics.ter import TER

# Initialize evaluation metrics
bleu = BLEU()
chrf = CHRF()
ter = TER()

# Prepare the extracted sentences, source sentences and reference sentences for evaluation
extracted_sentences = []
source_sentences = []
reference_sentences = []

# Select from the extracted sentences, the i source sentences that correspond to the images
for i, (n, page) in enumerate(zip(image_paths_en.keys(), results.pages)):
    if i >= N:
        break
    # Get the corresponding sentence from the results
    extracted_sentences.append(page.render().replace('\n', ' '))
    # Get the corresponding sentence from the source corpus
    source_sentences.append(mt_eval.get_source()[int(n)])
    # Get the corresponding sentence from the reference corpus
    reference_sentences.append(mt_eval.get_references()[int(n)])

#  Compute the evaluation metrics
bleu = bleu.corpus_score(extracted_sentences, [source_sentences])
chrf = chrf.corpus_score(extracted_sentences, [source_sentences])
ter = ter.corpus_score(extracted_sentences, [source_sentences])

# Print the evaluation results
print(f"BLEU: {bleu.score:.2f}")
print(f"CHRF: {chrf.score:.2f}")
print(f"TER: {ter.score:.2f}")

# Set new source and reference sentences for the MTEvaluation class
mt_eval.set_source_from_list(extracted_sentences)
mt_eval.set_references_from_list(reference_sentences)


# %% [markdown]
# ## Traducción del texto extraido

# %%
# Translate the source sentences using the engines specified in the MTEvaluation class

mt_eval.translate(save=True, folder='translations')

# %%
# Print the translations
for engine, translations in mt_eval.mt.items():
    print(f"Translations for {engine}:")
    for i, translation in zip(image_paths_en.keys(), translations.segments()):
        print(f"Image {i}: {translation}")
    print()

# %% [markdown]
# ## Evaluar los resultados de cada modelo

# %%
mt_eval.corpus_evaluate(to_json=False)


