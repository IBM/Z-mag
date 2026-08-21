#!/usr/bin/env python3
"""
Validation script to compare BEM matrix results from two different implementations.

Usage:
    python validate_results.py dir1 dir2
    
Example:
    python validate_results.py standalone/BEM_computed_cpp_py standalone/BEM_computed_cpp_zerocopy
"""

import sys
import os
import glob
import numpy as np
from pathlib import Path


def load_matrix(filename):
    """Load matrix from text file (first line is dimensions)."""
    with open(filename, 'r') as f:
        # Read dimensions from first line
        first_line = f.readline().strip()
        rows, cols = map(int, first_line.split())
        
        # Handle empty matrices
        if rows == 0 or cols == 0:
            return np.zeros((rows, cols))
        
        # Load the matrix data
        matrix = np.loadtxt(f)
        
        # Reshape if necessary
        if matrix.ndim == 1:
            matrix = matrix.reshape(rows, cols)
        
    return matrix


def compare_matrices(mat1, mat2, rtol=1e-12, atol=1e-15):
    """Compare two matrices and return statistics."""
    if mat1.shape != mat2.shape:
        return {
            'match': False,
            'error': f'Shape mismatch: {mat1.shape} vs {mat2.shape}'
        }
    
    # Handle empty matrices (both are empty = match)
    if mat1.size == 0 and mat2.size == 0:
        return {
            'match': True,
            'shape': mat1.shape,
            'max_abs_diff': 0.0,
            'mean_abs_diff': 0.0,
            'max_rel_diff': 0.0,
            'num_mismatches': 0,
            'total_elements': 0,
            'mismatch_percentage': 0.0
        }
    
    # Compute differences
    abs_diff = np.abs(mat1 - mat2)
    rel_diff = np.abs((mat1 - mat2) / (np.abs(mat2) + 1e-20))
    
    # Check if they match within tolerance
    matches = np.allclose(mat1, mat2, rtol=rtol, atol=atol)
    
    # Find mismatches
    mismatch_mask = ~np.isclose(mat1, mat2, rtol=rtol, atol=atol)
    num_mismatches = np.sum(mismatch_mask)
    
    stats = {
        'match': matches,
        'shape': mat1.shape,
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'max_rel_diff': np.max(rel_diff),
        'num_mismatches': num_mismatches,
        'total_elements': mat1.size,
        'mismatch_percentage': 100.0 * num_mismatches / mat1.size if mat1.size > 0 else 0.0
    }
    
    # Find worst mismatches
    if num_mismatches > 0:
        worst_indices = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        stats['worst_mismatch'] = {
            'index': worst_indices,
            'value1': mat1[worst_indices],
            'value2': mat2[worst_indices],
            'abs_diff': abs_diff[worst_indices],
            'rel_diff': rel_diff[worst_indices]
        }
    
    return stats


def validate_directories(dir1, dir2, rtol=1e-12, atol=1e-15):
    """Compare all B_matrix_slice files in two directories."""
    
    # Find all matrix files in both directories
    pattern = "B_matrix_slice_rank*.txt"
    files1 = sorted(glob.glob(os.path.join(dir1, pattern)))
    files2 = sorted(glob.glob(os.path.join(dir2, pattern)))
    
    if not files1:
        print(f"❌ No files matching {pattern} found in {dir1}")
        return False
    
    if not files2:
        print(f"❌ No files matching {pattern} found in {dir2}")
        return False
    
    # Extract rank numbers
    def get_rank(filename):
        basename = os.path.basename(filename)
        # Extract number between 'rank' and '.txt'
        rank_str = basename.split('rank')[1].split('.')[0]
        return int(rank_str)
    
    ranks1 = {get_rank(f): f for f in files1}
    ranks2 = {get_rank(f): f for f in files2}
    
    common_ranks = sorted(set(ranks1.keys()) & set(ranks2.keys()))
    only_in_1 = set(ranks1.keys()) - set(ranks2.keys())
    only_in_2 = set(ranks2.keys()) - set(ranks1.keys())
    
    print("="*70)
    print("BEM Matrix Validation")
    print("="*70)
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print(f"Tolerance:   rtol={rtol}, atol={atol}")
    print(f"\nFiles found: {len(files1)} in dir1, {len(files2)} in dir2")
    print(f"Common ranks: {len(common_ranks)}")
    
    if only_in_1:
        print(f"⚠️  Only in dir1: ranks {sorted(only_in_1)}")
    if only_in_2:
        print(f"⚠️  Only in dir2: ranks {sorted(only_in_2)}")
    
    print("\n" + "-"*70)
    print("Comparing matrices...")
    print("-"*70)
    
    all_match = True
    stats_summary = []
    
    for rank in common_ranks:
        file1 = ranks1[rank]
        file2 = ranks2[rank]
        
        try:
            mat1 = load_matrix(file1)
            mat2 = load_matrix(file2)
            
            stats = compare_matrices(mat1, mat2, rtol, atol)
            stats['rank'] = rank
            stats_summary.append(stats)
            
            if stats['match']:
                shape_str = f"{stats['shape'][0]}×{stats['shape'][1]}"
                if stats['total_elements'] == 0:
                    print(f"✓ Rank {rank:3d}: MATCH  ({shape_str}, empty matrix)")
                else:
                    print(f"✓ Rank {rank:3d}: MATCH  ({shape_str}, "
                          f"max_diff={stats['max_abs_diff']:.2e})")
            else:
                print(f"✗ Rank {rank:3d}: MISMATCH  ({stats.get('error', 'See details below')})")
                all_match = False
                
                if 'worst_mismatch' in stats:
                    wm = stats['worst_mismatch']
                    print(f"    Max abs diff: {stats['max_abs_diff']:.6e}")
                    print(f"    Max rel diff: {stats['max_rel_diff']:.6e}")
                    print(f"    Mismatches: {stats['num_mismatches']}/{stats['total_elements']} "
                          f"({stats['mismatch_percentage']:.2f}%)")
                    print(f"    Worst at [{wm['index'][0]}, {wm['index'][1]}]: "
                          f"{wm['value1']:.6e} vs {wm['value2']:.6e}")
        
        except Exception as e:
            print(f"✗ Rank {rank:3d}: ERROR - {str(e)}")
            all_match = False
    
    # Summary statistics
    if stats_summary and all([s['match'] for s in stats_summary]):
        print("\n" + "="*70)
        print("Summary Statistics")
        print("="*70)
        
        all_max_diff = [s['max_abs_diff'] for s in stats_summary]
        all_mean_diff = [s['mean_abs_diff'] for s in stats_summary]
        
        print(f"Overall max absolute difference: {max(all_max_diff):.6e}")
        print(f"Overall mean absolute difference: {np.mean(all_mean_diff):.6e}")
        print(f"Median max difference: {np.median(all_max_diff):.6e}")
    
    # Final result
    print("\n" + "="*70)
    if all_match and len(common_ranks) > 0:
        print("✅ VALIDATION PASSED - All matrices match!")
    else:
        print("❌ VALIDATION FAILED - Differences found!")
    print("="*70 + "\n")
    
    return all_match


def main():
    if len(sys.argv) != 3:
        print("Usage: python validate_results.py <dir1> <dir2>")
        print("\nExample:")
        print("  python validate_results.py standalone/BEM_computed_cpp_py standalone/BEM_computed_cpp_zerocopy")
        return 1
    
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    
    if not os.path.isdir(dir1):
        print(f"Error: Directory not found: {dir1}")
        return 1
    
    if not os.path.isdir(dir2):
        print(f"Error: Directory not found: {dir2}")
        return 1
    
    # Run validation
    success = validate_directories(dir1, dir2, rtol=1e-6, atol=1e-8)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

