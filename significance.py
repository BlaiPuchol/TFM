import argparse
import numpy as np
import sacrebleu
import sys

def load_lines(filepath):
    """Reads a file and returns a list of stripped lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_indices(filepath):
    """Reads the index file and returns a list of integers."""
    indices = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                indices.append(int(line))
            except ValueError:
                print(f"Warning: Skipping malformed line in index file: {line}")
    return indices

def align_references(all_refs, indices):
    """Aligns the master reference list to the specific subset using indices."""
    aligned_refs = []
    for idx in indices:
        if idx < len(all_refs):
            aligned_refs.append(all_refs[idx])
        else:
            # Fallback for out of bounds (though this shouldn't happen in clean data)
            aligned_refs.append("") 
    return aligned_refs

def bootstrap_test(args):
    # 1. Load all files
    print(f"Loading Index file A: {args.index1}")
    indices1 = load_indices(args.index1)

    print(f"Loading Index file B: {args.index2}")
    indices2 = load_indices(args.index2)
    
    print(f"Loading Reference file: {args.ref_file}")
    full_refs = load_lines(args.ref_file)
    
    print(f"Loading System A: {args.sys1_file}")
    sys1 = load_lines(args.sys1_file)
    
    print(f"Loading System B: {args.sys2_file}")
    sys2 = load_lines(args.sys2_file)

    # 2. Validate lengths and truncate indices if necessary
    if len(sys1) < len(indices1):
        print(f"\n[WARNING] System A has fewer sentences ({len(sys1)}) than Index A ({len(indices1)}). Truncating indices.")
        indices1 = indices1[:len(sys1)]

    if len(sys2) < len(indices2):
        print(f"\n[WARNING] System B has fewer sentences ({len(sys2)}) than Index B ({len(indices2)}). Truncating indices.")
        indices2 = indices2[:len(sys2)]

    # Check if indices match for paired testing
    if indices1 == indices2:
        indices = indices1
    else:
        print("\n[INFO] Index order mismatch or subset detected. Aligning systems by index...")
        
        set1 = set(indices1)
        set2 = set(indices2)
        
        common_indices = set1.intersection(set2)
        
        if not common_indices:
            print("\n[ERROR] No common indices found between the two systems!")
            sys.exit(1)
            
        if len(common_indices) < len(set1) or len(common_indices) < len(set2):
            print(f"[WARNING] Using intersection of indices. A: {len(set1)}, B: {len(set2)} -> Common: {len(common_indices)}")
            
        # Create mappings: Index -> Sentence
        map1 = dict(zip(indices1, sys1))
        map2 = dict(zip(indices2, sys2))
        
        # Sort indices to ensure deterministic alignment
        indices = sorted(list(common_indices))
        
        # Reconstruct aligned lists
        sys1 = [map1[idx] for idx in indices]
        sys2 = [map2[idx] for idx in indices]
        
        print(f"Aligned {len(indices)} sentences based on sorted indices.")

    # 3. Perform Alignment
    # Select only the references that match the indices
    aligned_refs = align_references(full_refs, indices)
    
    # Optional: Lowercasing (standard for some metrics, but usually BLEU is case-sensitive)
    if args.lowercase:
        sys1 = [s.lower() for s in sys1]
        sys2 = [s.lower() for s in sys2]
        aligned_refs = [r.lower() for r in aligned_refs]

    # 4. Calculate Actual Scores
    # sacrebleu expects references as a list of lists [[ref1, ref2...]]
    refs_formatted = [aligned_refs]
    
    bleu1 = sacrebleu.corpus_bleu(sys1, refs_formatted).score
    bleu2 = sacrebleu.corpus_bleu(sys2, refs_formatted).score
    
    print(f"\n--- Base Evaluation (on {len(indices)} aligned sentences) ---")
    print(f"System A BLEU: {bleu1:.2f}")
    print(f"System B BLEU: {bleu2:.2f}")
    
    diff_orig = abs(bleu1 - bleu2)
    if diff_orig == 0:
        print("Scores are identical. No significance test needed.")
        return

    # 5. Bootstrap Resampling
    print(f"\nRunning {args.samples} bootstrap samples...")
    
    wins_sys1 = 0
    wins_sys2 = 0
    n_sentences = len(sys1)
    
    # Convert to numpy for speed
    np_sys1 = np.array(sys1)
    np_sys2 = np.array(sys2)
    np_refs = np.array(aligned_refs)

    for i in range(args.samples):
        # Resample indices with replacement
        # We are resampling the *aligned* lists, not the original file indices
        resample_idx = np.random.choice(n_sentences, size=n_sentences, replace=True)
        
        sample_sys1 = np_sys1[resample_idx].tolist()
        sample_sys2 = np_sys2[resample_idx].tolist()
        sample_ref = [np_refs[resample_idx].tolist()] # Wrap in list for sacrebleu

        s_bleu1 = sacrebleu.corpus_bleu(sample_sys1, sample_ref).score
        s_bleu2 = sacrebleu.corpus_bleu(sample_sys2, sample_ref).score

        if s_bleu1 > s_bleu2:
            wins_sys1 += 1
        elif s_bleu2 > s_bleu1:
            wins_sys2 += 1

        # Optional progress indicator
        if (i+1) % 100 == 0:
            sys.stdout.write(f"\rProgress: {i+1}/{args.samples}")
            sys.stdout.flush()
            
    print("\n")

    # 6. Calculate P-Value
    # Null hypothesis: Both systems are equal.
    # If Sys A > Sys B, p-value is the fraction of times Sys B won in the resampling.
    
    if bleu1 > bleu2:
        better_system = "System A"
        p_value = 1.0 - (wins_sys1 / args.samples)
    else:
        better_system = "System B"
        p_value = 1.0 - (wins_sys2 / args.samples)

    print(f"--- Statistical Result ---")
    print(f"Winner: {better_system}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"CONCLUSION: SIGNIFICANT DIFFERENCE (p < 0.05)")
    else:
        print(f"CONCLUSION: NOT SIGNIFICANT (p >= 0.05)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Statistical Significance with Aligned References")
    
    # File arguments
    parser.add_argument("--ref_file", type=str, required=True, help="Master Reference file (contains all sentences)")
    parser.add_argument("--index1", type=str, required=True, help="File of indices for System A")
    parser.add_argument("--index2", type=str, required=True, help="File of indices for System B")
    parser.add_argument("--sys1_file", type=str, required=True, help="Output file from Model A (subset)")
    parser.add_argument("--sys2_file", type=str, required=True, help="Output file from Model B (subset)")
    
    # Options
    parser.add_argument("--samples", type=int, default=1000, help="Number of bootstrap samples (default 1000)")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase texts before computing BLEU")

    args = parser.parse_args()
    bootstrap_test(args)