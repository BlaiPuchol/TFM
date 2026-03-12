import argparse
import pandas as pd
import re
import sys

def load_lines(filepath):
    """Reads a file and returns a list of stripped lines."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

def parse_ocr_line(line):
    """
    Extracts the Index and the actual Text from the OCR line.
    Format expected: "Image 6552: some text here"
    Returns: (6552, "some text here")
    """
    # Regex to find "Image " followed by digits and a colon
    match = re.search(r"Image (\d+):", line)
    if match:
        index = int(match.group(1))
        # Split on the first colon to get the text part
        # limit split to 1 in case the text itself contains colons
        parts = line.split(':', 1)
        if len(parts) > 1:
            text = parts[1].strip()
        else:
            text = "" # Handle case where line might be empty after ID
        return index, text
    else:
        return None, line # Fallback if format is weird

def main():
    parser = argparse.ArgumentParser(description="Generate a Qualitative Analysis CSV by parsing OCR indices.")
    
    # Input Files
    parser.add_argument("--ocr_file", type=str, required=True, help="OCR Output file (contains 'Image ID: ...')")
    parser.add_argument("--hyp_file", type=str, required=True, help="Hypothesis file (MT Output, aligns 1-to-1 with OCR file)")
    parser.add_argument("--src_file", type=str, required=True, help="Master Source file (contains ALL source sentences)")
    parser.add_argument("--ref_file", type=str, required=True, help="Master Reference file (contains ALL reference sentences)")
    
    # Output
    parser.add_argument("--output", type=str, default="qualitative_analysis_full.csv", help="Output CSV filename")

    args = parser.parse_args()

    print("--- Loading Files ---")
    ocr_lines = load_lines(args.ocr_file)
    hyp_lines = load_lines(args.hyp_file)
    master_src = load_lines(args.src_file)
    master_ref = load_lines(args.ref_file)

    # Validation: OCR and Hypothesis must be 1-to-1
    if len(ocr_lines) != len(hyp_lines):
        print(f"[ERROR] Mismatch in line counts!")
        print(f"OCR File: {len(ocr_lines)} lines")
        print(f"Hyp File: {len(hyp_lines)} lines")
        print("These two files must be generated from the same pipeline run.")
        sys.exit(1)

    print(f"Processing {len(ocr_lines)} samples...")
    print("--- Aligning Data ---")
    
    aligned_data = []
    
    for i in range(len(ocr_lines)):
        raw_ocr_line = ocr_lines[i]
        hyp_text = hyp_lines[i]
        
        # 1. Extract Index from OCR Line
        idx, ocr_text_only = parse_ocr_line(raw_ocr_line)
        
        if idx is None:
            print(f"Warning: Could not parse index in line {i+1}: '{raw_ocr_line}'")
            # Create a placeholder row or skip
            continue

        # 2. Fetch Source and Reference using the extracted Index
        # Check bounds to avoid crashing if Index is huge
        if idx < len(master_src):
            src_text = master_src[idx]
        else:
            src_text = "[INDEX OUT OF BOUNDS]"

        if idx < len(master_ref):
            ref_text = master_ref[idx]
        else:
            ref_text = "[INDEX OUT OF BOUNDS]"

        # 3. Build the Row
        row = {
            'Index': idx,
            'Source (Master)': src_text,
            'OCR Text (Input)': ocr_text_only,
            'Hypothesis (Output)': hyp_text,
            'Reference (Master)': ref_text
        }
        aligned_data.append(row)

    # 4. Export
    df = pd.DataFrame(aligned_data)
    
    # Reorder columns for logical reading
    cols = ['Index', 'Source (Master)', 'OCR Text (Input)', 'Hypothesis (Output)', 'Reference (Master)']
    df = df[cols]
    
    print(f"--- Saving to {args.output} ---")
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print("Done! You can now analyze the CSV.")

if __name__ == "__main__":
    main()