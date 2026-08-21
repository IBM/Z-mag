#!/usr/bin/env python3
"""
Compare full global BEM matrices assembled from distributed rank slices.

This script handles different MPI decompositions by:
1. Loading boundary node coordinates from .npy files
2. Matching rows by physical coordinates (not rank ownership)
3. Assembling and comparing the full global matrices

Usage:
    python compare_global_matrices.py dir1 dir2
    
Example:
    python compare_global_matrices.py ./output ./standalone/BEM_computed_cpp_debug
"""

import sys
import os
import glob
import numpy as np
from scipy.spatial import cKDTree


def load_matrix_slice(txt_file):
    """Load matrix slice from text file."""
    with open(txt_file, 'r') as f:
        first_line = f.readline().strip()
        rows, cols = map(int, first_line.split())
        
        if rows == 0 or cols == 0:
            return np.zeros((rows, cols))
        
        matrix = np.loadtxt(f)
        if matrix.ndim == 1:
            matrix = matrix.reshape(rows, cols)
    
    return matrix


def load_coordinates(npy_file):
    """Load boundary node coordinates from .npy file."""
    try:
        with open(npy_file, 'rb') as f:
            _ = np.load(f)  # BEM matrix (skip)
            _ = np.load(f)  # global_n_bnd_nodes (skip)
            _ = np.load(f)  # gnodes_to_bnodes (skip)
            _ = np.load(f)  # bnd_nodes (skip)
            coordinates = np.load(f)  # Boundary node coordinates
        return coordinates
    except (FileNotFoundError, EOFError, ValueError) as e:
        raise RuntimeError(
            f"Could not load coordinates from {npy_file}.\n"
            f"Error: {e}\n"
            f"Make sure the BEM matrices were saved with the updated save() method\n"
            f"that includes coordinates. You may need to re-run the assembly."
        )


def assemble_global_matrix_with_coords(directory):
    """
    Assemble global matrix and coordinates from distributed rank slices.
    
    Returns
    -------
    tuple
        (global_matrix, global_coords, rank_info)
    """
    # Find matrix text files
    txt_pattern = "B_matrix_slice_rank*.txt"
    txt_files = sorted(glob.glob(os.path.join(directory, txt_pattern)))
    
    if not txt_files:
        raise ValueError(f"No files matching {txt_pattern} found in {directory}")
    
    # Extract rank numbers
    def get_rank(filename):
        basename = os.path.basename(filename)
        rank_str = basename.split('rank')[1].split('.')[0]
        return int(rank_str)
    
    rank_txt_files = [(get_rank(f), f) for f in txt_files]
    rank_txt_files.sort(key=lambda x: x[0])
    
    print(f"  Loading {len(rank_txt_files)} rank slices from {directory}")
    
    # Load matrices and coordinates for each rank
    matrices = []
    coords_list = []
    ranks = []
    
    for rank, txt_file in rank_txt_files:
        # Load matrix
        matrix = load_matrix_slice(txt_file)
        
        # Load coordinates from corresponding .npy file
        npy_file = txt_file.replace('.txt', '.npy')
        coords = load_coordinates(npy_file)
        
        if coords.shape[0] != matrix.shape[0]:
            raise ValueError(
                f"Rank {rank}: Mismatch between matrix rows ({matrix.shape[0]}) "
                f"and coordinates ({coords.shape[0]})"
            )
        
        matrices.append(matrix)
        coords_list.append(coords)
        ranks.append(rank)
        
        print(f"    Rank {rank:3d}: {matrix.shape[0]:4d} rows × {matrix.shape[1]:4d} cols")
    
    # Verify all matrices have same number of columns
    n_global_cols = matrices[0].shape[1]
    for rank, mat in zip(ranks, matrices):
        if mat.shape[1] != n_global_cols:
            raise ValueError(
                f"Rank {rank}: Column mismatch! Expected {n_global_cols}, got {mat.shape[1]}"
            )
    
    # Concatenate all coordinates and matrices
    # Note: This stacks by rank order, but we'll use coordinates to reorder later
    global_coords = np.vstack(coords_list)
    global_matrix = np.vstack(matrices)
    
    rank_info = {
        'n_ranks': len(rank_txt_files),
        'ranks': ranks,
        'rows_per_rank': [m.shape[0] for m in matrices],
        'n_global_rows': global_matrix.shape[0],
        'n_global_cols': global_matrix.shape[1]
    }
    
    print(f"  ✓ Assembled: {global_matrix.shape[0]} × {global_matrix.shape[1]} with coordinates")
    
    return global_matrix, global_coords, rank_info


def match_and_reorder_matrices(mat1, coords1, mat2, coords2, tol=1e-10):
    """
    Match and reorder matrices based on physical coordinates.
    
    Parameters
    ----------
    mat1, mat2 : np.ndarray
        Matrices to compare
    coords1, coords2 : np.ndarray
        Coordinates of rows in mat1 and mat2
    tol : float
        Coordinate matching tolerance
    
    Returns
    -------
    tuple
        (mat1_reordered, mat2_reordered, match_info)
    """
    print("\n  Matching rows by coordinates...")
    
    n1, n2 = coords1.shape[0], coords2.shape[0]
    
    if n1 != n2:
        raise ValueError(
            f"Different number of boundary nodes: {n1} vs {n2}\n"
            f"This suggests different meshes or boundary conditions!"
        )
    
    # Build KD-tree for efficient coordinate matching
    tree = cKDTree(coords2)
    distances, indices = tree.query(coords1, k=1)
    
    # Check if all points matched within tolerance
    max_dist = np.max(distances)
    unmatched = np.sum(distances > tol)
    
    print(f"    Max coordinate distance: {max_dist:.6e}")
    print(f"    Unmatched nodes (dist > {tol}): {unmatched}")
    
    if unmatched > 0:
        print(f"    ⚠️  Warning: {unmatched} nodes could not be matched!")
        print(f"       This may indicate different meshes or numerical precision issues.")
    
    # Reorder mat2 to match mat1's coordinate ordering
    mat2_reordered = mat2[indices, :]
    
    match_info = {
        'max_coord_dist': max_dist,
        'unmatched_nodes': unmatched,
        'reorder_indices': indices
    }
    
    print(f"  ✓ Matrices reordered to common coordinate ordering")
    
    return mat1, mat2_reordered, match_info


def compare_matrices(mat1, mat2, rtol=1e-12, atol=1e-15):
    """Compare two matrices."""
    if mat1.shape != mat2.shape:
        return {
            'match': False,
            'error': f'Shape mismatch: {mat1.shape} vs {mat2.shape}'
        }
    
    abs_diff = np.abs(mat1 - mat2)
    rel_diff = np.abs((mat1 - mat2) / (np.abs(mat2) + 1e-20))
    
    matches = np.allclose(mat1, mat2, rtol=rtol, atol=atol)
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


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_global_matrices.py <dir1> <dir2>")
        print("\nExample:")
        print("  python compare_global_matrices.py ./output ./standalone/BEM_computed_cpp_debug")
        return 1
    
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    
    if not os.path.isdir(dir1):
        print(f"Error: Directory not found: {dir1}")
        return 1
    
    if not os.path.isdir(dir2):
        print(f"Error: Directory not found: {dir2}")
        return 1
    
    print("="*70)
    print("Global BEM Matrix Comparison (Coordinate-Based Matching)")
    print("="*70)
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print(f"Tolerance:   rtol=1e-12, atol=1e-15")
    print("="*70)
    print()
    
    # Assemble global matrices with coordinates
    try:
        print("Assembling Matrix 1:")
        mat1, coords1, info1 = assemble_global_matrix_with_coords(dir1)
        print()
        
        print("Assembling Matrix 2:")
        mat2, coords2, info2 = assemble_global_matrix_with_coords(dir2)
        print()
    except Exception as e:
        print(f"❌ ERROR during assembly: {e}")
        return 1
    
    # Print decomposition info
    print("-"*70)
    print("Decomposition Information:")
    print("-"*70)
    print(f"Matrix 1: {info1['n_ranks']} ranks, {info1['n_global_rows']} total rows")
    print(f"  Rows per rank: {info1['rows_per_rank']}")
    print(f"Matrix 2: {info2['n_ranks']} ranks, {info2['n_global_rows']} total rows")
    print(f"  Rows per rank: {info2['rows_per_rank']}")
    print()
    
    # Match and reorder matrices based on coordinates
    print("-"*70)
    print("Matching Rows by Coordinates:")
    print("-"*70)
    try:
        mat1_ordered, mat2_ordered, match_info = match_and_reorder_matrices(
            mat1, coords1, mat2, coords2, tol=1e-10
        )
    except Exception as e:
        print(f"❌ ERROR during matching: {e}")
        return 1
    
    # Compare matrices
    print()
    print("-"*70)
    print("Comparing Global Matrices:")
    print("-"*70)
    
    stats = compare_matrices(mat1_ordered, mat2_ordered, rtol=1e-12, atol=1e-15)
    
    if 'error' in stats:
        print(f"❌ {stats['error']}")
        print("="*70)
        return 1
    
    # Print results
    print(f"Matrix shape:     {stats['shape'][0]} × {stats['shape'][1]}")
    print(f"Total elements:   {stats['total_elements']:,}")
    print(f"Max abs diff:     {stats['max_abs_diff']:.6e}")
    print(f"Mean abs diff:    {stats['mean_abs_diff']:.6e}")
    print(f"Max rel diff:     {stats['max_rel_diff']:.6e}")
    
    if stats['num_mismatches'] > 0:
        print(f"Mismatches:       {stats['num_mismatches']:,} ({stats['mismatch_percentage']:.4f}%)")
        
        if 'worst_mismatch' in stats:
            wm = stats['worst_mismatch']
            print(f"Worst at [{wm['index'][0]}, {wm['index'][1]}]:")
            print(f"  Value 1:  {wm['value1']:.12e}")
            print(f"  Value 2:  {wm['value2']:.12e}")
            print(f"  Abs diff: {wm['abs_diff']:.12e}")
            print(f"  Rel diff: {wm['rel_diff']:.12e}")
    
    print()
    print("="*70)
    
    if stats['match']:
        print("✅ VALIDATION PASSED - Global matrices match!")
        print("   (Different MPI decompositions produce identical results)")
    else:
        print("❌ VALIDATION FAILED - Matrices differ beyond tolerance!")
    
    print("="*70)
    print()
    
    return 0 if stats['match'] else 1


if __name__ == "__main__":
    sys.exit(main())
