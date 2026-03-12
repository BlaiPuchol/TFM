import argparse
import pandas as pd
import sys

def load_lines(filepath):
    """Reads a file and returns a list of stripped lines."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

def load_indices(filepath):
    """Reads the index file and returns a list of integers."""
    indices = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    indices.append(int(line))
                except ValueError:
                    print(f"Warning: Skipping malformed line in index file: {line}")
    except FileNotFoundError:
        print(f"Error: Index file not found: {filepath}")
        sys.exit(1)
    return indices

def main():
    parser = argparse.ArgumentParser(description="Generate Qualitative Analysis CSV for Gemini (Index-based alignment).")
    
    # Input Files
    parser.add_argument("--index_file", type=str, required=True, help="File containing one image index per line")
    parser.add_argument("--hyp_file", type=str, required=True, help="Gemini Hypothesis file (aligned line-by-line with index_file)")
    parser.add_argument("--src_file", type=str, required=True, help="Master Source file (contains ALL source sentences)")
    parser.add_argument("--ref_file", type=str, required=True, help="Master Reference file (contains ALL reference sentences)")
    
    # Output
    parser.add_argument("--output", type=str, default="qualitative_analysis_gemini.csv", help="Output CSV filename")

    args = parser.parse_args()

    print("--- Loading Files ---")
    indices = load_indices(args.index_file)
    hyp_lines = load_lines(args.hyp_file)
    master_src = load_lines(args.src_file)
    master_ref = load_lines(args.ref_file)

    # Validation: Index and Hypothesis must be 1-to-1
    if len(indices) != len(hyp_lines):
        print(f"[ERROR] Mismatch in line counts!")
        print(f"Index File: {len(indices)} lines")
        print(f"Hypothesis File: {len(hyp_lines)} lines")
        print("The hypothesis file must have exactly one line per index in the index file.")
        sys.exit(1)

    print(f"Processing {len(indices)} samples...")
    print("--- Aligning Data ---")
    
    aligned_data = []
    
    for i, idx in enumerate(indices):
        hyp_text = hyp_lines[i]
        
        # Fetch Source and Reference using the Index
        # Check bounds to avoid crashing if Index is huge
        if idx < len(master_src):
            src_text = master_src[idx]
        else:
            src_text = "[INDEX OUT OF BOUNDS]"

        if idx < len(master_ref):
            ref_text = master_ref[idx]
        else:
            ref_text = "[INDEX OUT OF BOUNDS]"

        # Build the Row
        # Since Gemini translates from Image -> Text, 'OCR Text' is not applicable. 
        # We compare Ground Truth Source vs Gemini Output.
        row = {
            'Index': idx,
            'Source (Ground Truth)': src_text,
            'Hypothesis (Gemini)': hyp_text,
            'Reference (Human)': ref_text
        }
        aligned_data.append(row)

    # Export
    df = pd.DataFrame(aligned_data)
    
    # Reorder columns
    cols = ['Index', 'Source (Ground Truth)', 'Hypothesis (Gemini)', 'Reference (Human)']
    df = df[cols]
    
    print(f"--- Saving to {args.output} ---")
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print("Done! You can now analyze the Gemini results.")

if __name__ == "__main__":
    main()