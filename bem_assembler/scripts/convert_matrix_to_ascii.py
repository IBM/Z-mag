#!/usr/bin/env python3
"""
Convert BEM matrix .npy files to ASCII format for C++ verification.

Usage:
    python convert_matrix_to_ascii.py --rank 0
    python convert_matrix_to_ascii.py --size 80
"""

import numpy as np
import os
import argparse
import sys


def convert_matrix_to_ascii(npy_path, txt_path):
    """Convert a single .npy matrix to ASCII format."""
    print(f"Converting {npy_path} -> {txt_path}")
    
    if not os.path.exists(npy_path):
        print(f"  ⚠ Warning: {npy_path} not found, skipping")
        return False
    
    # Load matrix
    B = np.load(npy_path)
    
    # Write to ASCII
    with open(txt_path, 'w') as f:
        # Write dimensions
        f.write(f"{B.shape[0]} {B.shape[1]}\n")
        
        # Write matrix data (one row per line)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                f.write(f"{B[i, j]:.16e}")
                if j < B.shape[1] - 1:
                    f.write(" ")
            f.write("\n")
    
    print(f"  ✓ Converted {B.shape[0]}x{B.shape[1]} matrix")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert BEM matrix .npy files to ASCII format'
    )
    parser.add_argument('--rank', type=int, help='Convert specific rank')
    parser.add_argument('--size', type=int, help='Convert all ranks 0 to size-1')
    parser.add_argument('--input-dir', default='standalone/BEM_computed',
                       help='Input directory with .npy files')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for ASCII files (default: same as input)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print("=" * 60)
    print("BEM Matrix NPY to ASCII Converter")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found!")
        sys.exit(1)
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Determine which ranks to convert
    if args.rank is not None:
        ranks = [args.rank]
    elif args.size is not None:
        ranks = range(args.size)
    else:
        # Auto-detect
        ranks = []
        for f in os.listdir(args.input_dir):
            if f.startswith('B_matrix_slice_rank') and f.endswith('.npy'):
                rank_num = int(f.replace('B_matrix_slice_rank', '').replace('.npy', ''))
                ranks.append(rank_num)
        ranks = sorted(set(ranks))
        
        if not ranks:
            print(f"Error: No B_matrix_slice_rank*.npy files found in {args.input_dir}")
            sys.exit(1)
        
        print(f"Auto-detected {len(ranks)} ranks")
    
    # Convert each rank
    success_count = 0
    skip_count = 0
    
    for rank in ranks:
        npy_path = os.path.join(args.input_dir, f'B_matrix_slice_rank{rank}.npy')
        txt_path = os.path.join(args.output_dir, f'B_matrix_slice_rank{rank}.txt')
        
        if convert_matrix_to_ascii(npy_path, txt_path):
            success_count += 1
        else:
            skip_count += 1
    
    print("\n" + "=" * 60)
    print(f"Conversion complete!")
    print(f"  Successfully converted: {success_count} matrices")
    if skip_count > 0:
        print(f"  Skipped: {skip_count} files")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

