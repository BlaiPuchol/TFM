#!/bin/bash
#SBATCH -p docencia                  # Cola (partición)
#SBATCH --gres=gpu:2             # --- 6 GPUs logicas ---
#SBATCH --cpus-per-task=12        # 8 CPUs (4 por GPU)
#SBATCH --mem=64G                # 64GB de RAM
#SBATCH --job-name=translate   # ¡Nuevo nombre de trabajo!
#SBATCH -o translate_salida_%j.log     # ¡Nuevos logs!
#SBATCH -e translate_error_%j.log      # ¡Nuevos logs!

# Translate all 4 directions using OCR CSV files and IWSLT references

set -e

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OCR_DIR="${SCRIPT_DIR}/ocr/test"
IWSLT_DE_EN="${SCRIPT_DIR}/IWSLT/iwslt14.de-en"
IWSLT_FR_EN="${SCRIPT_DIR}/IWSLT/iwslt17.fr-en"
TRANS_FOLDER="${SCRIPT_DIR}/translations_eurollm"
EVAL_FOLDER="${SCRIPT_DIR}/evaluations_eurollm"

cd "$SCRIPT_DIR"

# Optional: pass --num_shots N as argument to the shell script
NUM_SHOTS=0

echo "=============================="
echo " Translation - all directions"
echo " num_shots=${NUM_SHOTS}"
echo "=============================="

echo ""
echo "=== de -> en ==="
python TFM/only_translate_eurollm.py \
    --src_lang de --tgt_lang en \
    --ocr_csv "${OCR_DIR}/ocr_results_onnxtr_de_en.csv" \
    --ref_file "${IWSLT_DE_EN}/test.en" \
    --save_trans --trans_folder "${TRANS_FOLDER}" \
    --save_eval  --eval_folder  "${EVAL_FOLDER}" \
    --print_results \
    --num_shots "${NUM_SHOTS}"

echo ""
echo "=== en -> de ==="
python TFM/only_translate_eurollm.py \
    --src_lang en --tgt_lang de \
    --ocr_csv "${OCR_DIR}/ocr_results_onnxtr_en_de.csv" \
    --ref_file "${IWSLT_DE_EN}/test.de" \
    --save_trans --trans_folder "${TRANS_FOLDER}" \
    --save_eval  --eval_folder  "${EVAL_FOLDER}" \
    --print_results \
    --num_shots "${NUM_SHOTS}"

echo ""
echo "=== en -> fr ==="
python TFM/only_translate_eurollm.py \
    --src_lang en --tgt_lang fr \
    --ocr_csv "${OCR_DIR}/ocr_results_onnxtr_en_fr.csv" \
    --ref_file "${IWSLT_FR_EN}/test.fr" \
    --save_trans --trans_folder "${TRANS_FOLDER}" \
    --save_eval  --eval_folder  "${EVAL_FOLDER}" \
    --print_results \
    --num_shots "${NUM_SHOTS}"

echo ""
echo "=== fr -> en ==="
python TFM/only_translate_eurollm.py \
    --src_lang fr --tgt_lang en \
    --ocr_csv "${OCR_DIR}/ocr_results_onnxtr_fr_en.csv" \
    --ref_file "${IWSLT_FR_EN}/test.en" \
    --save_trans --trans_folder "${TRANS_FOLDER}" \
    --save_eval  --eval_folder  "${EVAL_FOLDER}" \
    --print_results \
    --num_shots "${NUM_SHOTS}"

echo ""
echo "=============================="
echo " Done. Results saved to:"
echo "  Translations: ${TRANS_FOLDER}"
echo "  Evaluations:  ${EVAL_FOLDER}"
echo "=============================="
