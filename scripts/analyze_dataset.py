#!/usr/bin/env python3
"""
Analysis script for geometry_clauses30_samples10M.jsonl
Analyzes llm_input_renamed column for:
0. Auxiliary content analysis (ratio of samples containing '<aux>') + predicate distribution in aux content
1. Point distribution 
2. Predicate distribution (before and after ?)
3. Proof length (number of ; in <proof> </proof>)

Example data structure:
{
  "llm_input_renamed": "<problem>a01 a02 a03 : x00 g : coll a01 a02 g [001] perp a01 a02 g a03 [002] ; ? perp a01 a03 g</problem>",
  "llm_output_renamed": "<aux> x00 g : coll a b g [006] perp a b g d [007] ; </aux><proof>coll a01 a02 g [001]; perp a01 a02 g a03 [002]; ...</proof>"
}

Analysis process demonstration:
1. Point extraction: From "a01 a02 a03 : x00 g :" → points = {a01, a02, a03, x00, g} → count = 5
2. Predicates before '?': From "coll a01 a02 g [001] perp a01 a02 g a03 [002]" → [coll, perp]
3. Predicates after '?': From "perp a01 a03 g" → [perp]
4. Auxiliary predicate combinations: From "<aux> x00 g : coll a b g [006] perp a b g d [007] ; p1 [008] ; p2 p3 [009] ; </aux>" → [('coll', 'perp'), ('p1',), ('p2', 'p3')]
5. Proof length: Count semicolons in <proof> content
"""

import json
import re
from collections import Counter, defaultdict
import sys
from tqdm import tqdm

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_points(problem_text):
    """
    Extract point names from the problem text using proper segmentation.
    
    Example: "a01 a02 a03 : x00 g : coll a01 a02 g [001]" → {a01, a02, a03, x00, g}
    """
    points = set()
    
    # Split by semicolons first
    segments = problem_text.split(';')
    
    for segment in segments:
        segment = segment.strip()
        if ':' in segment:
            # Extract points before ":"
            before_colon = segment.split(':')[0].strip()
            # Get single letter points
            point_letters = before_colon.split(" ")
            points.update(point_letters)
    
    return points

def extract_predicates_before_after_question(problem_text):
    """
    Extract predicates before and after the ? mark using proper segmentation.
    
    Example: "coll a01 a02 g [001] perp a01 a02 g a03 [002] ; ? perp a01 a03 g"
    Returns: ([coll, perp], [perp])
    """
    parts = problem_text.split('?')
    if len(parts) != 2:
        return [], []
    
    before_part = parts[0]
    after_part = parts[1]
    
    def extract_predicates_from_part(part):
        predicates = []
        # Split by semicolons
        segments = part.split(';')
        
        for segment in segments:
            segment = segment.strip()
            
            # Handle segments that may contain both point definitions and predicates
            if ':' in segment:
                # Check if there's content after the colon that contains predicates
                colon_parts = segment.split(':', 1)
                if len(colon_parts) > 1:
                    after_colon = colon_parts[1].strip()
                    # Look for predicates in the part after the colon - pattern: word + args + [number]
                    predicate_matches = re.findall(r'([a-z]+)\s+[a-z\s]+\s*\[\d+\]', after_colon)
                    predicates.extend(predicate_matches)
                continue
            
            # Look for patterns like "xxxx [number]"
            if '[' in segment:
                # Extract everything before [number]
                before_bracket = segment.split('[')[0].strip()
                if before_bracket:
                    # Split by space and get first token (predicate)
                    tokens = before_bracket.split()
                    if tokens:
                        predicate = tokens[0]
                        predicates.append(predicate)
            else:
                # Handle predicates without brackets (like after ?)
                tokens = segment.split()
                if tokens:
                    predicate = tokens[0]
                    predicates.append(predicate)
        
        return predicates
    
    before_preds = extract_predicates_from_part(before_part)
    after_preds = extract_predicates_from_part(after_part)
    
    return before_preds, after_preds

def extract_aux_predicates(llm_output):
    """
    Extract predicate combinations from auxiliary content between <aux> and </aux> tags.
    Each semicolon-separated segment represents a separate combination.
    
    Example: "<aux> x00 g : coll a b g [006] perp a b g d [007] ; p1 p2 [008] ; p3 [009] ; </aux>"
    Returns: [('coll', 'perp'), ('p1',), ('p3',)] - list of combination tuples
    """
    aux_match = re.search(r'<aux>(.*?)</aux>', llm_output, re.DOTALL)
    if not aux_match:
        return []
    
    aux_content = aux_match.group(1).strip()
    combinations = []
    
    # Split by semicolons to get separate combinations
    segments = aux_content.split(';')
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        combinations = []
        # Handle segments that may contain both point definitions and predicates
        if ':' in segment:
            # Check if there's content after the colon that contains predicates
            colon_parts = segment.split(':', 1)
            if len(colon_parts) > 1:
                after_colon = colon_parts[1].strip()
                # Look for predicates in the part after the colon - pattern: word + args + [number]
                predicate_matches = re.findall(r'([a-z]+)\s+[a-z\s]+\s*\[\d+\]', after_colon)
                combinations.append(tuple(sorted(predicate_matches)))
    
    return combinations

# ============================================================================
# ANALYSIS FUNCTIONS  
# ============================================================================

def count_proof_semicolons(llm_output):
    """
    Count semicolons in the <proof> </proof> section to measure proof length.
    
    Example: "<proof>coll a01 a02 g [001]; perp a01 a02 g a03 [002];</proof>" → 2
    """
    proof_match = re.search(r'<proof>(.*?)</proof>', llm_output, re.DOTALL)
    if proof_match:
        proof_content = proof_match.group(1)
        return proof_content.count(';')
    return 0

def analyze_jsonl_file(file_path):
    """
    Main analysis function that processes the entire JSONL file.
    
    Returns: (point_counts, predicates_before, predicates_after, aux_predicates, proof_lengths, aux_count)
    """
    print(f"Analyzing {file_path}...")
    
    point_counts = Counter()
    predicates_before = Counter()
    predicates_after = Counter()
    aux_predicate_combinations = Counter()
    proof_lengths = []
    aux_count = 0
    
    # Count total lines first for progress bar
    print("Counting total lines...")
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines:,}")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Processing lines")):
            
            try:
                data = json.loads(line.strip())
                llm_input = data['llm_input_renamed']
                llm_output = data['llm_output_renamed']
                
                # Check for <aux> in llm_output_renamed and extract predicate combinations
                if '<aux>' in llm_output:
                    aux_count += 1
                    # Extract predicate combinations from auxiliary content
                    aux_pred_combos = extract_aux_predicates(llm_output)
                    # Count each combination separately
                    for combo in aux_pred_combos:
                        if combo:  # Only count non-empty combinations
                            aux_predicate_combinations[combo] += 1
                
                # Extract problem text from <problem> tags
                problem_match = re.search(r'<problem>(.*?)</problem>', llm_input, re.DOTALL)
                if problem_match:
                    problem_text = problem_match.group(1)
                    
                    # 1. Point distribution
                    points = extract_points(problem_text)
                    point_counts[len(points)] += 1
                    
                    # 2. Predicate distribution
                    before_preds, after_preds = extract_predicates_before_after_question(problem_text)
                    predicates_before.update(before_preds)
                    predicates_after.update(after_preds)
                
                # 3. Proof length
                proof_length = count_proof_semicolons(llm_output)
                proof_lengths.append(proof_length)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue
    
    return point_counts, predicates_before, predicates_after, aux_predicate_combinations, proof_lengths, aux_count

# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

def generate_report(point_counts, predicates_before, predicates_after, aux_predicate_combinations, proof_lengths, aux_count):
    """
    Generate comprehensive analysis report with all statistics.
    
    Displays: auxiliary content analysis, point distribution, predicate distribution, and proof length analysis.
    """
    print("\n" + "="*80)
    print("GEOMETRY DATASET ANALYSIS REPORT")
    print("="*80)
    
    # 0. Auxiliary Content Analysis
    print("\n0. AUXILIARY CONTENT ANALYSIS")
    print("-" * 40)
    total_samples = sum(point_counts.values())
    aux_ratio = (aux_count / total_samples) * 100 if total_samples > 0 else 0
    print(f"Total samples analyzed: {total_samples:,}")
    print(f"Samples containing '<aux>': {aux_count:,} ({aux_ratio:.2f}%)")
    print(f"Samples without '<aux>': {total_samples - aux_count:,} ({100 - aux_ratio:.2f}%)")
    
    # Auxiliary predicate combination distribution
    if aux_predicate_combinations:
        print(f"\nAuxiliary Predicate Combination Distribution:")
        total_aux_combinations = sum(aux_predicate_combinations.values())
        for combo, count in aux_predicate_combinations.most_common():
            percentage = (count / total_aux_combinations) * 100
            # Format combination as [pred1, pred2, ...]
            combo_str = '[' + ', '.join(combo) + ']'
            print(f"  {combo_str}: {count:,} ({percentage:.2f}%)")
    
    # 1. Point Distribution Analysis
    print("\n1. POINT DISTRIBUTION ANALYSIS")
    print("-" * 40)
    print(f"Point count distribution:")
    for num_points in sorted(point_counts.keys()):
        count = point_counts[num_points]
        percentage = (count / total_samples) * 100
        print(f"  {num_points} points: {count:,} samples ({percentage:.2f}%)")
    
    # 2. Predicate Distribution Analysis
    print("\n2. PREDICATE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    print("\nPredicates BEFORE '?' (Given conditions):")
    total_before = sum(predicates_before.values())
    for pred, count in predicates_before.most_common():
        percentage = (count / total_before) * 100
        print(f"  {pred}: {count:,} ({percentage:.2f}%)")
    
    print("\nPredicates AFTER '?' (Goals to prove):")
    total_after = sum(predicates_after.values())
    for pred, count in predicates_after.most_common():
        percentage = (count / total_after) * 100
        print(f"  {pred}: {count:,} ({percentage:.2f}%)")
    
    # 3. Proof Length Analysis
    print("\n3. PROOF LENGTH ANALYSIS")
    print("-" * 40)
    if proof_lengths:
        import numpy as np
        proof_lengths_array = np.array(proof_lengths)
        print(f"Total proofs analyzed: {len(proof_lengths):,}")
        print(f"Average proof length: {np.mean(proof_lengths_array):.2f} steps")
        print(f"Median proof length: {np.median(proof_lengths_array):.1f} steps")
        print(f"Min proof length: {np.min(proof_lengths_array)} steps")
        print(f"Max proof length: {np.max(proof_lengths_array)} steps")
        print(f"Standard deviation: {np.std(proof_lengths_array):.2f}")
        
        # Proof length distribution
        proof_length_dist = Counter(proof_lengths)
        min_length = min(proof_length_dist.keys())
        max_length = max(proof_length_dist.keys())
        
        print(f"\nProof length distribution (full range: {min_length}-{max_length}):")
        
        # Create bins based on proof length values
        bin_counts = defaultdict(int)
        
        for length, count in proof_length_dist.items():
            if length < 20:
                # Bin size = 1 for lengths < 20
                bin_counts[(length, length)] += count
            elif length < 100:
                # Bin size = 10 for lengths < 100
                bin_start = (length // 10) * 10
                bin_end = bin_start + 9
                bin_counts[(bin_start, bin_end)] += count
            elif length < 200:
                # Bin size = 20 for lengths < 200
                bin_start = (length // 20) * 20
                bin_end = bin_start + 19
                bin_counts[(bin_start, bin_end)] += count
            else:
                # Bin size = 50 for lengths >= 200
                bin_start = (length // 50) * 50
                bin_end = bin_start + 49
                bin_counts[(bin_start, bin_end)] += count
        
        # Display binned distribution
        for (start, end), count in sorted(bin_counts.items()):
            percentage = (count / len(proof_lengths)) * 100
            if start == end:
                print(f"  {start} steps: {count:,} proofs ({percentage:.2f}%)")
            else:
                print(f"  {start}-{end} steps: {count:,} proofs ({percentage:.2f}%)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point for the analysis script.
    
    Usage: python analyze_dataset.py <path_to_jsonl_file>
    """
    if len(sys.argv) != 2:
        print("Usage: python analyze_dataset.py <path_to_jsonl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Analyze the data
    point_counts, predicates_before, predicates_after, aux_predicate_combinations, proof_lengths, aux_count = analyze_jsonl_file(file_path)
    
    # Generate report
    generate_report(point_counts, predicates_before, predicates_after, aux_predicate_combinations, proof_lengths, aux_count)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()